// ============================================================================
// Tests for c10::Allocator and c10::AllocatorRegistry
// ============================================================================
//
// 15 test cases covering:
//   - Interface implementability
//   - allocate() correctness (non-null, device, zero-size)
//   - raw_deleter() returns correct function pointer
//   - raw_allocate() / raw_deallocate() convenience
//   - Registry: register, retrieve, nullptr for unregistered, overwrite
//   - Non-copyability via type traits
//   - Multiple independent allocations

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "c10/core/Allocator.h"
#include "c10/core/AllocatorRegistry.h"

namespace {

using namespace c10;

// ---------------------------------------------------------------------------
// Mock allocator — concrete implementation for testing only
// ---------------------------------------------------------------------------

class MockCPUAllocator : public Allocator {
 public:
  DataPtr allocate(size_t nbytes) const override {
    void* ptr = nbytes > 0 ? std::malloc(nbytes) : nullptr;
    return DataPtr(ptr, deleteCPU, Device::cpu());
  }

  Deleter raw_deleter() const override { return deleteCPU; }
};

// A second mock to test registry overwriting.
class MockCPUAllocator2 : public Allocator {
 public:
  DataPtr allocate(size_t nbytes) const override {
    void* ptr = nbytes > 0 ? std::malloc(nbytes) : nullptr;
    return DataPtr(ptr, deleteCPU, Device::cpu());
  }

  Deleter raw_deleter() const override { return deleteCPU; }
};

// ============================================================================
// 1. Interface is implementable — compiles and links
// ============================================================================
TEST(Allocator, InterfaceIsImplementable) {
  MockCPUAllocator alloc;
  // If this compiles, the interface is correctly implementable.
  Allocator* base = &alloc;
  EXPECT_NE(base, nullptr);
}

// ============================================================================
// 2. allocate(64) returns non-null DataPtr
// ============================================================================
TEST(Allocator, AllocateReturnsNonNull) {
  MockCPUAllocator alloc;
  auto dp = alloc.allocate(64);
  EXPECT_NE(dp.get(), nullptr);
  // dp destructor frees — ASan validates.
}

// ============================================================================
// 3. allocate(64) returns CPU device DataPtr
// ============================================================================
TEST(Allocator, AllocateReturnsCPUDevice) {
  MockCPUAllocator alloc;
  auto dp = alloc.allocate(64);
  EXPECT_TRUE(dp.device().is_cpu());
  EXPECT_EQ(dp.device().index, 0);
}

// ============================================================================
// 4. allocate(0) does not crash — zero size allocation
// ============================================================================
TEST(Allocator, AllocateZeroSizeDoesNotCrash) {
  MockCPUAllocator alloc;
  auto dp = alloc.allocate(0);
  // data_ may be null for zero-size, that's fine.
  // Just must not crash.
  EXPECT_TRUE(dp.device().is_cpu());
}

// ============================================================================
// 5. raw_deleter() returns correct function pointer
// ============================================================================
TEST(Allocator, RawDeleterReturnsCorrectPointer) {
  MockCPUAllocator alloc;
  EXPECT_EQ(alloc.raw_deleter(), deleteCPU);
}

// ============================================================================
// 6. raw_allocate() returns non-null raw pointer
// ============================================================================
TEST(Allocator, RawAllocateReturnsNonNull) {
  MockCPUAllocator alloc;
  void* ptr = alloc.raw_allocate(128);
  ASSERT_NE(ptr, nullptr);
  // Caller took ownership — must free manually.
  alloc.raw_deallocate(ptr);
}

// ============================================================================
// 7. raw_deallocate() frees without crash — ASan validates
// ============================================================================
TEST(Allocator, RawDeallocateFreesCleanly) {
  MockCPUAllocator alloc;
  void* ptr = std::malloc(32);
  ASSERT_NE(ptr, nullptr);
  std::memset(ptr, 0xCD, 32);
  // raw_deallocate should call free() — ASan validates no leak/double-free.
  alloc.raw_deallocate(ptr);
}

// ============================================================================
// 8. RegisterAllocator stores allocator correctly
// ============================================================================
TEST(AllocatorRegistry, RegisterStoresCorrectly) {
  MockCPUAllocator alloc;
  RegisterAllocator(DeviceType::CPU, &alloc);
  EXPECT_EQ(GetAllocator(DeviceType::CPU), &alloc);
}

// ============================================================================
// 9. GetAllocator(DeviceType::CPU) returns registered allocator
// ============================================================================
TEST(AllocatorRegistry, GetAllocatorReturnsRegistered) {
  MockCPUAllocator alloc;
  RegisterAllocator(DeviceType::CPU, &alloc);
  Allocator* retrieved = GetAllocator(DeviceType::CPU);
  ASSERT_NE(retrieved, nullptr);

  // Verify it's actually usable — allocate through the retrieved pointer.
  auto dp = retrieved->allocate(16);
  EXPECT_NE(dp.get(), nullptr);
  EXPECT_TRUE(dp.device().is_cpu());
}

// ============================================================================
// 10. GetAllocator for unregistered device returns nullptr
// ============================================================================
TEST(AllocatorRegistry, UnregisteredDeviceReturnsNull) {
  // CUDA is not registered in our test setup (unless a previous test did).
  // Use a high device type index that definitely isn't registered.
  // We'll test by registering CPU then checking CUDA is still whatever
  // it was (likely nullptr from the zero-initialized array or a prior test).
  // Better: check the return for an invalid DeviceType.
  Allocator* result = GetAllocator(static_cast<DeviceType>(7));
  EXPECT_EQ(result, nullptr);
}

// ============================================================================
// 11. GetAllocator(Device) convenience overload works
// ============================================================================
TEST(AllocatorRegistry, GetAllocatorDeviceOverload) {
  MockCPUAllocator alloc;
  RegisterAllocator(DeviceType::CPU, &alloc);

  Allocator* retrieved = GetAllocator(Device::cpu());
  EXPECT_EQ(retrieved, &alloc);
}

// ============================================================================
// 12. Second registration overwrites first
// ============================================================================
TEST(AllocatorRegistry, SecondRegistrationOverwrites) {
  MockCPUAllocator alloc1;
  MockCPUAllocator2 alloc2;

  RegisterAllocator(DeviceType::CPU, &alloc1);
  EXPECT_EQ(GetAllocator(DeviceType::CPU), &alloc1);

  RegisterAllocator(DeviceType::CPU, &alloc2);
  EXPECT_EQ(GetAllocator(DeviceType::CPU), &alloc2);
  EXPECT_NE(GetAllocator(DeviceType::CPU), &alloc1);
}

// ============================================================================
// 13. DataPtr returned by allocate() frees correctly on destruction
// ============================================================================
TEST(Allocator, AllocateFreesOnDestruction) {
  MockCPUAllocator alloc;
  {
    auto dp = alloc.allocate(256);
    EXPECT_NE(dp.get(), nullptr);
    // Write to it — ASan validates the memory is real.
    std::memset(dp.get(), 0xAB, 256);
  }  // dp destructor fires — ASan validates no leak.
}

// ============================================================================
// 14. Multiple allocations return independent DataPtrs
// ============================================================================
TEST(Allocator, MultipleAllocationsAreIndependent) {
  MockCPUAllocator alloc;
  auto dp1 = alloc.allocate(32);
  auto dp2 = alloc.allocate(32);
  auto dp3 = alloc.allocate(32);

  // All non-null and distinct.
  EXPECT_NE(dp1.get(), nullptr);
  EXPECT_NE(dp2.get(), nullptr);
  EXPECT_NE(dp3.get(), nullptr);
  EXPECT_NE(dp1.get(), dp2.get());
  EXPECT_NE(dp1.get(), dp3.get());
  EXPECT_NE(dp2.get(), dp3.get());
  // Each has its own deleter — all three freed independently by destructors.
}

// ============================================================================
// 15. Allocator is not copyable — verified via type traits
// ============================================================================
TEST(Allocator, NotCopyable) {
  EXPECT_FALSE(std::is_copy_constructible<MockCPUAllocator>::value);
  EXPECT_FALSE(std::is_copy_assignable<MockCPUAllocator>::value);
  EXPECT_FALSE(std::is_move_constructible<MockCPUAllocator>::value);
  EXPECT_FALSE(std::is_move_assignable<MockCPUAllocator>::value);
}

}  // namespace
