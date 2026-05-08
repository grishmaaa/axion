// ============================================================================
// Tests for c10::StorageImpl
// ============================================================================
//
// 14 test cases covering:
//   - Construction with allocator (non-null, correct size, correct device)
//   - data() returns correct raw pointer
//   - nbytes() returns correct size
//   - device() returns correct device
//   - Not copyable (type traits)
//   - Movable (ownership transfers)
//   - Shared ownership via intrusive_ptr (refcount, lifetime)
//   - Destroyed when last intrusive_ptr dies
//   - Resize larger (data preserved)
//   - Resize smaller
//   - Resize on non-resizable storage asserts
//   - External DataPtr construction
//   - Empty/zero-size storage
//   - Allocator pointer stored correctly

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "c10/core/CPUAllocator.h"
#include "c10/core/StorageImpl.h"
#include "c10/util/intrusive_ptr.h"

namespace {

using namespace c10;

// ============================================================================
// 1. Construction with CPUAllocator — data non-null, size correct, device CPU
// ============================================================================
TEST(StorageImpl, ConstructWithAllocator) {
  auto storage = make_intrusive<StorageImpl>(256, GetCPUAllocator());
  EXPECT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->nbytes(), 256u);
  EXPECT_TRUE(storage->device().is_cpu());
}

// ============================================================================
// 2. data() returns the correct raw pointer
// ============================================================================
TEST(StorageImpl, DataReturnsRawPointer) {
  auto storage = make_intrusive<StorageImpl>(64, GetCPUAllocator());
  void* ptr = storage->data();
  ASSERT_NE(ptr, nullptr);
  // Write and read back to prove it's real.
  std::memset(ptr, 0xCD, 64);
  EXPECT_EQ(static_cast<unsigned char*>(ptr)[0], 0xCD);
  EXPECT_EQ(static_cast<unsigned char*>(ptr)[63], 0xCD);
}

// ============================================================================
// 3. nbytes() returns the allocated size
// ============================================================================
TEST(StorageImpl, NbytesReturnsCorrectSize) {
  auto s1 = make_intrusive<StorageImpl>(100, GetCPUAllocator());
  auto s2 = make_intrusive<StorageImpl>(4096, GetCPUAllocator());
  EXPECT_EQ(s1->nbytes(), 100u);
  EXPECT_EQ(s2->nbytes(), 4096u);
}

// ============================================================================
// 4. device() returns CPU
// ============================================================================
TEST(StorageImpl, DeviceIsCPU) {
  auto storage = make_intrusive<StorageImpl>(32, GetCPUAllocator());
  EXPECT_TRUE(storage->device().is_cpu());
  EXPECT_EQ(storage->device().index, 0);
}

// ============================================================================
// 5. Not copyable — verified via type traits
// ============================================================================
TEST(StorageImpl, NotCopyable) {
  EXPECT_FALSE(std::is_copy_constructible<StorageImpl>::value);
  EXPECT_FALSE(std::is_copy_assignable<StorageImpl>::value);
}

// ============================================================================
// 6. Movable — ownership transfers, source left empty
// ============================================================================
TEST(StorageImpl, Movable) {
  StorageImpl src(64, GetCPUAllocator());
  void* orig_ptr = src.data();
  EXPECT_NE(orig_ptr, nullptr);

  StorageImpl dst(std::move(src));
  EXPECT_EQ(dst.data(), orig_ptr);
  EXPECT_EQ(dst.nbytes(), 64u);
  // Source should be empty after move.
  EXPECT_EQ(src.data(), nullptr);   // NOLINT: intentional use-after-move
  EXPECT_EQ(src.nbytes(), 0u);     // NOLINT
}

// ============================================================================
// 7. Shared ownership via intrusive_ptr — refcount correct
// ============================================================================
TEST(StorageImpl, SharedOwnershipRefcount) {
  auto s1 = make_intrusive<StorageImpl>(128, GetCPUAllocator());
  EXPECT_EQ(s1.use_count(), 1u);

  auto s2 = s1;  // copy intrusive_ptr
  EXPECT_EQ(s1.use_count(), 2u);
  EXPECT_EQ(s2.use_count(), 2u);
  EXPECT_EQ(s1->data(), s2->data());  // same underlying memory
}

// ============================================================================
// 8. Destroyed when last intrusive_ptr dies
// ============================================================================
TEST(StorageImpl, DestroyedWhenLastRefDies) {
  intrusive_ptr<StorageImpl> outer;
  {
    auto inner = make_intrusive<StorageImpl>(64, GetCPUAllocator());
    outer = inner;  // refcount = 2
    EXPECT_EQ(outer.use_count(), 2u);
  }  // inner dies — refcount = 1
  EXPECT_EQ(outer.use_count(), 1u);
  EXPECT_NE(outer->data(), nullptr);
  // When outer dies, the StorageImpl and its DataPtr are freed.
  // ASan validates — no leak.
}

// ============================================================================
// 9. Resize larger — data preserved, new size correct
// ============================================================================
TEST(StorageImpl, ResizeLarger) {
  auto storage = make_intrusive<StorageImpl>(64, GetCPUAllocator());
  // Write a known pattern.
  std::memset(storage->data(), 0xAB, 64);

  storage->resize(256);
  EXPECT_EQ(storage->nbytes(), 256u);
  EXPECT_NE(storage->data(), nullptr);

  // Old data should be preserved in the first 64 bytes.
  auto* bytes = static_cast<unsigned char*>(storage->data());
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(bytes[i], 0xAB) << "Byte " << i << " not preserved after resize";
  }
}

// ============================================================================
// 10. Resize smaller — new size correct
// ============================================================================
TEST(StorageImpl, ResizeSmaller) {
  auto storage = make_intrusive<StorageImpl>(256, GetCPUAllocator());
  std::memset(storage->data(), 0xCD, 256);

  storage->resize(32);
  EXPECT_EQ(storage->nbytes(), 32u);
  EXPECT_NE(storage->data(), nullptr);

  // First 32 bytes should be preserved.
  auto* bytes = static_cast<unsigned char*>(storage->data());
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(bytes[i], 0xCD) << "Byte " << i << " not preserved after shrink";
  }
}

// ============================================================================
// 11. Resize on non-resizable storage — death test
// ============================================================================
TEST(StorageImplDeathTest, ResizeNonResizable) {
  auto storage = make_intrusive<StorageImpl>(64, GetCPUAllocator());
  storage->set_resizable(false);
  EXPECT_DEATH(storage->resize(128), "");
}

// ============================================================================
// 12. External DataPtr construction — wrapping foreign memory
// ============================================================================
TEST(StorageImpl, ExternalDataPtr) {
  // Simulate externally provided memory.
  void* external = std::malloc(512);
  ASSERT_NE(external, nullptr);
  std::memset(external, 0x42, 512);

  {
    DataPtr dp(external, deleteCPU, Device::cpu());
    auto storage = make_intrusive<StorageImpl>(512, std::move(dp), nullptr);

    EXPECT_EQ(storage->data(), external);
    EXPECT_EQ(storage->nbytes(), 512u);
    EXPECT_EQ(storage->allocator(), nullptr);
    EXPECT_TRUE(storage->device().is_cpu());
    // Not resizable when allocator is null.
    EXPECT_FALSE(storage->resizable());
  }
  // storage dies → DataPtr dies → free(external). ASan validates.
}

// ============================================================================
// 13. Zero-size storage
// ============================================================================
TEST(StorageImpl, ZeroSizeStorage) {
  auto storage = make_intrusive<StorageImpl>(0, GetCPUAllocator());
  EXPECT_EQ(storage->nbytes(), 0u);
  // data() may be null for zero-size — that's fine.
  EXPECT_TRUE(storage->device().is_cpu());
}

// ============================================================================
// 14. Allocator pointer stored correctly
// ============================================================================
TEST(StorageImpl, AllocatorStoredCorrectly) {
  auto* alloc = GetCPUAllocator();
  auto storage = make_intrusive<StorageImpl>(128, alloc);
  EXPECT_EQ(storage->allocator(), alloc);
}

}  // namespace
