// ============================================================================
// Tests for c10::Storage
// ============================================================================
//
// 16 test cases covering:
//   - Default construction (invalid)
//   - Factory construction (create / wrap)
//   - Forwarding accessors (data, nbytes, device, allocator, resizable)
//   - Copy semantics (refcount increment, shared ownership)
//   - Move semantics (ownership transfer, source invalid)
//   - Equality (same impl == equal, different impl != equal)
//   - valid() / bool conversion
//   - resize() forwarding
//   - use_count() / unique()
//   - Memory freed when last Storage dies (ASan validates)

#include <gtest/gtest.h>

#include <cstring>

#include "c10/core/CPUAllocator.h"
#include "c10/core/Storage.h"

namespace {

using namespace c10;

// ============================================================================
// 1. Default construction is invalid
// ============================================================================
TEST(Storage, DefaultConstructionIsInvalid) {
  Storage s;
  EXPECT_FALSE(s.valid());
  EXPECT_FALSE(static_cast<bool>(s));
}

// ============================================================================
// 2. Factory create — allocates fresh storage
// ============================================================================
TEST(Storage, CreateAllocates) {
  auto s = Storage::create(256, GetCPUAllocator());
  EXPECT_TRUE(s.valid());
  EXPECT_NE(s.data(), nullptr);
  EXPECT_EQ(s.nbytes(), 256u);
}

// ============================================================================
// 3. Factory wrap — wraps external memory
// ============================================================================
TEST(Storage, WrapExternalMemory) {
  void* raw = std::malloc(128);
  ASSERT_NE(raw, nullptr);
  std::memset(raw, 0x42, 128);

  auto s = Storage::wrap(128, DataPtr(raw, deleteCPU, Device::cpu()));
  EXPECT_TRUE(s.valid());
  EXPECT_EQ(s.data(), raw);
  EXPECT_EQ(s.nbytes(), 128u);
  EXPECT_TRUE(s.device().is_cpu());
  // Storage destructor frees raw. ASan validates.
}

// ============================================================================
// 4. data() returns correct pointer
// ============================================================================
TEST(Storage, DataAccessor) {
  auto s = Storage::create(64, GetCPUAllocator());
  void* ptr = s.data();
  ASSERT_NE(ptr, nullptr);
  // Write and read to prove it's real.
  std::memset(ptr, 0xAB, 64);
  EXPECT_EQ(static_cast<unsigned char*>(ptr)[0], 0xAB);
}

// ============================================================================
// 5. nbytes() returns correct size
// ============================================================================
TEST(Storage, NbytesAccessor) {
  auto s = Storage::create(1024, GetCPUAllocator());
  EXPECT_EQ(s.nbytes(), 1024u);
}

// ============================================================================
// 6. device() returns CPU
// ============================================================================
TEST(Storage, DeviceAccessor) {
  auto s = Storage::create(32, GetCPUAllocator());
  EXPECT_TRUE(s.device().is_cpu());
  EXPECT_EQ(s.device().index, 0);
}

// ============================================================================
// 7. allocator() returns correct pointer
// ============================================================================
TEST(Storage, AllocatorAccessor) {
  auto* alloc = GetCPUAllocator();
  auto s = Storage::create(64, alloc);
  EXPECT_EQ(s.allocator(), alloc);
}

// ============================================================================
// 8. resizable() default is true for allocator-constructed
// ============================================================================
TEST(Storage, ResizableDefault) {
  auto s = Storage::create(64, GetCPUAllocator());
  EXPECT_TRUE(s.resizable());
}

// ============================================================================
// 9. Copy increments refcount — shared ownership
// ============================================================================
TEST(Storage, CopySharesOwnership) {
  auto s1 = Storage::create(128, GetCPUAllocator());
  EXPECT_EQ(s1.use_count(), 1u);
  EXPECT_TRUE(s1.unique());

  auto s2 = s1;  // copy
  EXPECT_EQ(s1.use_count(), 2u);
  EXPECT_EQ(s2.use_count(), 2u);
  EXPECT_FALSE(s1.unique());
  EXPECT_EQ(s1.data(), s2.data());  // same underlying memory
  EXPECT_EQ(s1, s2);                // identity equal
}

// ============================================================================
// 10. Move transfers ownership — source becomes invalid
// ============================================================================
TEST(Storage, MoveTransfersOwnership) {
  auto s1 = Storage::create(64, GetCPUAllocator());
  void* orig_ptr = s1.data();

  auto s2 = std::move(s1);
  EXPECT_TRUE(s2.valid());
  EXPECT_EQ(s2.data(), orig_ptr);
  EXPECT_EQ(s2.use_count(), 1u);
  EXPECT_FALSE(s1.valid());  // NOLINT: intentional use-after-move
}

// ============================================================================
// 11. Equality — same impl is equal, different impl is not
// ============================================================================
TEST(Storage, EqualityComparison) {
  auto s1 = Storage::create(64, GetCPUAllocator());
  auto s2 = s1;  // copy — same impl
  auto s3 = Storage::create(64, GetCPUAllocator());  // different impl

  EXPECT_EQ(s1, s2);
  EXPECT_NE(s1, s3);
}

// ============================================================================
// 12. resize() forwards to StorageImpl
// ============================================================================
TEST(Storage, ResizeForwards) {
  auto s = Storage::create(64, GetCPUAllocator());
  std::memset(s.data(), 0xCD, 64);

  s.resize(256);
  EXPECT_EQ(s.nbytes(), 256u);
  // First 64 bytes preserved.
  auto* bytes = static_cast<unsigned char*>(s.data());
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(bytes[i], 0xCD) << "Byte " << i << " not preserved";
  }
}

// ============================================================================
// 13. set_resizable() forwards correctly
// ============================================================================
TEST(Storage, SetResizable) {
  auto s = Storage::create(64, GetCPUAllocator());
  EXPECT_TRUE(s.resizable());
  s.set_resizable(false);
  EXPECT_FALSE(s.resizable());
}

// ============================================================================
// 14. use_count() and unique()
// ============================================================================
TEST(Storage, UseCountAndUnique) {
  auto s1 = Storage::create(32, GetCPUAllocator());
  EXPECT_EQ(s1.use_count(), 1u);
  EXPECT_TRUE(s1.unique());

  {
    auto s2 = s1;
    EXPECT_EQ(s1.use_count(), 2u);
    EXPECT_FALSE(s1.unique());
  }

  EXPECT_EQ(s1.use_count(), 1u);
  EXPECT_TRUE(s1.unique());
}

// ============================================================================
// 15. Memory freed when last Storage dies — ASan validates
// ============================================================================
TEST(Storage, MemoryFreedOnLastDeath) {
  {
    auto s1 = Storage::create(512, GetCPUAllocator());
    std::memset(s1.data(), 0xFF, 512);
    {
      auto s2 = s1;  // refcount = 2
      EXPECT_EQ(s2.use_count(), 2u);
    }  // s2 dies — refcount = 1, memory still alive
    EXPECT_NE(s1.data(), nullptr);
  }  // s1 dies — refcount = 0, memory freed. ASan validates.
}

// ============================================================================
// 16. Construct from raw StorageImpl pointer
// ============================================================================
TEST(Storage, ConstructFromRawPointer) {
  auto* raw = new StorageImpl(64, GetCPUAllocator());
  Storage s(raw);
  EXPECT_TRUE(s.valid());
  EXPECT_EQ(s.nbytes(), 64u);
  EXPECT_EQ(s.use_count(), 1u);
  // s destructor destroys the StorageImpl. ASan validates.
}

}  // namespace
