// ============================================================================
// Tests for c10::CPUAllocator
// ============================================================================
//
// Tests covering:
//   - allocate() returns non-null, correct device, correct alignment
//   - Zero-size allocation is safe
//   - Memory is writable and zero-initialized
//   - raw_deleter() returns deleteCPU
//   - Self-registration into AllocatorRegistry
//   - GetAllocator(DeviceType::CPU) returns the CPUAllocator
//   - GetCPUAllocator() returns a valid singleton
//   - raw_allocate/raw_deallocate work correctly
//   - Multiple allocations are independent and correctly aligned
//   - Large allocation works
//   - DataPtr from allocate() frees correctly on destruction (ASan)

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "c10/core/AllocatorRegistry.h"
#include "c10/core/CPUAllocator.h"

namespace {

using namespace c10;

// ============================================================================
// 1. allocate(64) returns non-null DataPtr
// ============================================================================
TEST(CPUAllocator, AllocateReturnsNonNull) {
  auto* alloc = GetCPUAllocator();
  auto dp = alloc->allocate(64);
  EXPECT_NE(dp.get(), nullptr);
}

// ============================================================================
// 2. allocate returns CPU device
// ============================================================================
TEST(CPUAllocator, AllocateReturnsCPUDevice) {
  auto* alloc = GetCPUAllocator();
  auto dp = alloc->allocate(128);
  EXPECT_TRUE(dp.device().is_cpu());
  EXPECT_EQ(dp.device().index, 0);
}

// ============================================================================
// 3. allocate returns 64-byte-aligned pointer
// ============================================================================
TEST(CPUAllocator, AllocateIsAligned) {
  auto* alloc = GetCPUAllocator();
  auto dp = alloc->allocate(100);
  auto addr = reinterpret_cast<uintptr_t>(dp.get());
  EXPECT_EQ(addr % kCPUAlignment, 0u)
      << "Pointer " << dp.get() << " is not 64-byte aligned";
}

// ============================================================================
// 4. Zero-size allocation does not crash
// ============================================================================
TEST(CPUAllocator, AllocateZeroSize) {
  auto* alloc = GetCPUAllocator();
  auto dp = alloc->allocate(0);
  // data_ may be null for zero-size — that's correct.
  EXPECT_EQ(dp.get(), nullptr);
  EXPECT_TRUE(dp.device().is_cpu());
}

// ============================================================================
// 5. Memory is writable
// ============================================================================
TEST(CPUAllocator, MemoryIsWritable) {
  auto* alloc = GetCPUAllocator();
  auto dp = alloc->allocate(256);
  ASSERT_NE(dp.get(), nullptr);
  // Write a pattern — ASan validates the memory is real.
  std::memset(dp.get(), 0xCD, 256);
  auto* bytes = static_cast<unsigned char*>(dp.get());
  EXPECT_EQ(bytes[0], 0xCD);
  EXPECT_EQ(bytes[255], 0xCD);
}

// ============================================================================
// 6. Memory is zero-initialized
// ============================================================================
TEST(CPUAllocator, MemoryIsZeroInitialized) {
  auto* alloc = GetCPUAllocator();
  auto dp = alloc->allocate(128);
  ASSERT_NE(dp.get(), nullptr);
  auto* bytes = static_cast<unsigned char*>(dp.get());
  for (size_t i = 0; i < 128; ++i) {
    EXPECT_EQ(bytes[i], 0) << "Byte " << i << " is not zero";
  }
}

// ============================================================================
// 7. raw_deleter() returns deleteCPU
// ============================================================================
TEST(CPUAllocator, RawDeleterIsDeleteCPU) {
  auto* alloc = GetCPUAllocator();
  EXPECT_EQ(alloc->raw_deleter(), deleteCPU);
}

// ============================================================================
// 8. Self-registration — GetAllocator(DeviceType::CPU) works
// ============================================================================
TEST(CPUAllocator, SelfRegistration) {
  Allocator* alloc = GetAllocator(DeviceType::CPU);
  ASSERT_NE(alloc, nullptr);
  // Verify it's the same object as GetCPUAllocator().
  EXPECT_EQ(alloc, GetCPUAllocator());
}

// ============================================================================
// 9. GetAllocator(Device::cpu()) convenience overload
// ============================================================================
TEST(CPUAllocator, GetAllocatorDeviceOverload) {
  Allocator* alloc = GetAllocator(Device::cpu());
  ASSERT_NE(alloc, nullptr);
  EXPECT_EQ(alloc, GetCPUAllocator());
}

// ============================================================================
// 10. Allocate through the registry — end-to-end
// ============================================================================
TEST(CPUAllocator, AllocateThroughRegistry) {
  Allocator* alloc = GetAllocator(DeviceType::CPU);
  ASSERT_NE(alloc, nullptr);
  auto dp = alloc->allocate(64);
  EXPECT_NE(dp.get(), nullptr);
  EXPECT_TRUE(dp.device().is_cpu());
}

// ============================================================================
// 11. GetCPUAllocator returns a stable singleton
// ============================================================================
TEST(CPUAllocator, SingletonIsStable) {
  auto* a = GetCPUAllocator();
  auto* b = GetCPUAllocator();
  EXPECT_EQ(a, b);
}

// ============================================================================
// 12. raw_allocate / raw_deallocate roundtrip
// ============================================================================
TEST(CPUAllocator, RawAllocateDeallocateRoundtrip) {
  auto* alloc = GetCPUAllocator();
  void* ptr = alloc->raw_allocate(512);
  ASSERT_NE(ptr, nullptr);
  // Write to prove it's real memory.
  std::memset(ptr, 0xAB, 512);
  // Free via the allocator's deleter — ASan validates.
  alloc->raw_deallocate(ptr);
}

// ============================================================================
// 13. Multiple allocations are independent
// ============================================================================
TEST(CPUAllocator, MultipleAllocationsIndependent) {
  auto* alloc = GetCPUAllocator();
  auto dp1 = alloc->allocate(64);
  auto dp2 = alloc->allocate(64);
  auto dp3 = alloc->allocate(64);

  EXPECT_NE(dp1.get(), nullptr);
  EXPECT_NE(dp2.get(), nullptr);
  EXPECT_NE(dp3.get(), nullptr);
  EXPECT_NE(dp1.get(), dp2.get());
  EXPECT_NE(dp1.get(), dp3.get());
  EXPECT_NE(dp2.get(), dp3.get());
}

// ============================================================================
// 14. Multiple allocations are all aligned
// ============================================================================
TEST(CPUAllocator, MultipleAllocationsAllAligned) {
  auto* alloc = GetCPUAllocator();
  for (int i = 0; i < 10; ++i) {
    auto dp = alloc->allocate(100 + i * 7);  // varying sizes
    ASSERT_NE(dp.get(), nullptr);
    auto addr = reinterpret_cast<uintptr_t>(dp.get());
    EXPECT_EQ(addr % kCPUAlignment, 0u)
        << "Allocation " << i << " at " << dp.get()
        << " is not 64-byte aligned";
  }
}

// ============================================================================
// 15. Large allocation works
// ============================================================================
TEST(CPUAllocator, LargeAllocation) {
  auto* alloc = GetCPUAllocator();
  constexpr size_t kSize = 64 * 1024 * 1024;  // 64 MiB
  auto dp = alloc->allocate(kSize);
  ASSERT_NE(dp.get(), nullptr);
  EXPECT_TRUE(dp.device().is_cpu());
  // Touch first and last byte — ASan validates.
  auto* bytes = static_cast<unsigned char*>(dp.get());
  bytes[0] = 0xFF;
  bytes[kSize - 1] = 0xFE;
  EXPECT_EQ(bytes[0], 0xFF);
  EXPECT_EQ(bytes[kSize - 1], 0xFE);
}

// ============================================================================
// 16. DataPtr from allocate frees correctly on destruction — ASan validates
// ============================================================================
TEST(CPUAllocator, DataPtrFreesOnDestruction) {
  auto* alloc = GetCPUAllocator();
  {
    auto dp = alloc->allocate(1024);
    EXPECT_NE(dp.get(), nullptr);
    std::memset(dp.get(), 0xAB, 1024);
  }  // dp destructor fires — ASan catches any leak.
}

}  // namespace
