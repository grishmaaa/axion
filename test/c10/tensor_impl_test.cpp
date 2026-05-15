#include <gtest/gtest.h>
#include "c10/core/TensorImpl.h"
#include "c10/core/CPUAllocator.h"

namespace {

using namespace c10;

// ============================================================================
// 1. Construction with explicit sizes and strides
// ============================================================================
TEST(TensorImpl, ExplicitConstruction) {
  auto storage = Storage::create(64 * 4, GetCPUAllocator());
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, sizes, strides, 5);

  EXPECT_EQ(impl->sizes(), sizes);
  EXPECT_EQ(impl->strides(), strides);
  EXPECT_EQ(impl->dtype(), ScalarType::Float32);
  EXPECT_EQ(impl->storage_offset(), 5);
}

// ============================================================================
// 2. Construction with sizes only — strides computed automatically
// ============================================================================
TEST(TensorImpl, AutomaticStrides) {
  auto storage = Storage::create(64 * 4, GetCPUAllocator());
  std::vector<int64_t> sizes = {3, 4};
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, sizes);

  std::vector<int64_t> expected_strides = {4, 1};
  EXPECT_EQ(impl->strides(), expected_strides);
  EXPECT_TRUE(impl->is_contiguous());
}

// ============================================================================
// 3. ndim() returns correct number of dimensions
// ============================================================================
TEST(TensorImpl, Ndim) {
  auto storage = Storage::create(64 * 4, GetCPUAllocator());
  auto impl1 = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{2, 3, 4});
  EXPECT_EQ(impl1->ndim(), 3);

  auto impl2 = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{10});
  EXPECT_EQ(impl2->ndim(), 1);

  auto impl3 = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{});
  EXPECT_EQ(impl3->ndim(), 0);
}

// ============================================================================
// 4. numel() returns correct product of sizes
// ============================================================================
TEST(TensorImpl, Numel) {
  auto storage = Storage::create(256, GetCPUAllocator());
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{3, 4, 5});
  EXPECT_EQ(impl->numel(), 60);
}

// ============================================================================
// 5. is_contiguous() true for freshly allocated row-major tensor
// ============================================================================
TEST(TensorImpl, IsContiguousFresh) {
  auto storage = Storage::create(256, GetCPUAllocator());
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{2, 5, 3});
  EXPECT_TRUE(impl->is_contiguous());
}

// ============================================================================
// 6. Strides for a 2D contiguous tensor are [ncols, 1]
// ============================================================================
TEST(TensorImpl, 2DStrides) {
  auto storage = Storage::create(256, GetCPUAllocator());
  int64_t rows = 3;
  int64_t cols = 7;
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{rows, cols});
  EXPECT_EQ(impl->strides()[0], cols);
  EXPECT_EQ(impl->strides()[1], 1);
}

// ============================================================================
// 7. data_ptr() with zero offset returns same pointer as storage.data()
// ============================================================================
TEST(TensorImpl, DataPtrZeroOffset) {
  auto storage = Storage::create(256, GetCPUAllocator());
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{10}, 0);
  EXPECT_EQ(impl->data(), storage.data());
}

// ============================================================================
// 8. data_ptr() with non-zero offset returns correctly shifted pointer
// ============================================================================
TEST(TensorImpl, DataPtrNonZeroOffset) {
  auto storage = Storage::create(256, GetCPUAllocator());
  int64_t offset = 10;
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{5}, offset);
  
  void* expected = static_cast<char*>(storage.data()) + offset * sizeof(float);
  EXPECT_EQ(impl->data(), expected);
}

// ============================================================================
// 9. ScalarType stored and retrieved correctly
// ============================================================================
TEST(TensorImpl, ScalarTypeCheck) {
  auto storage = Storage::create(256, GetCPUAllocator());
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Int64, std::vector<int64_t>{2});
  EXPECT_EQ(impl->dtype(), ScalarType::Int64);
}

// ============================================================================
// 10. elementSize returns correct byte count
// ============================================================================
TEST(TensorImpl, ElementSizeCheck) {
  EXPECT_EQ(elementSize(ScalarType::Float32), 4);
  EXPECT_EQ(elementSize(ScalarType::Int8), 1);
  EXPECT_EQ(elementSize(ScalarType::Float16), 2);
  EXPECT_EQ(elementSize(ScalarType::Int64), 8);
}

// ============================================================================
// 11. Device forwarded correctly from storage
// ============================================================================
TEST(TensorImpl, DeviceForwarding) {
  auto storage = Storage::create(256, GetCPUAllocator());
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{2});
  EXPECT_EQ(impl->device(), storage.device());
  EXPECT_TRUE(impl->device().is_cpu());
}

// ============================================================================
// 12. Storage sharing — refcount is 2
// ============================================================================
TEST(TensorImpl, StorageSharing) {
  auto storage = Storage::create(256, GetCPUAllocator());
  {
    auto impl1 = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{10});
    auto impl2 = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{5}, 5);
    EXPECT_EQ(storage.use_count(), 3); // storage, impl1->storage, impl2->storage
  }
  EXPECT_EQ(storage.use_count(), 1);
}

// ============================================================================
// 13. Scalar tensor — zero dimensions, one element, numel is 1
// ============================================================================
TEST(TensorImpl, ScalarTensor) {
  auto storage = Storage::create(4, GetCPUAllocator());
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{});
  EXPECT_EQ(impl->ndim(), 0);
  EXPECT_EQ(impl->numel(), 1);
  EXPECT_TRUE(impl->is_contiguous());
}

// ============================================================================
// 14. 1D tensor — sizes [n], strides [1], is_contiguous true
// ============================================================================
TEST(TensorImpl, 1DTensorContiguous) {
  auto storage = Storage::create(40, GetCPUAllocator());
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{10});
  EXPECT_EQ(impl->strides()[0], 1);
  EXPECT_TRUE(impl->is_contiguous());
}

// ============================================================================
// 15. Non-contiguous strides — manually set strides returns false
// ============================================================================
TEST(TensorImpl, NonContiguous) {
  auto storage = Storage::create(100, GetCPUAllocator());
  // 2x2 matrix, but strides are [1, 2] instead of [2, 1] (column-major)
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Float32, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 2});
  EXPECT_FALSE(impl->is_contiguous());
}

// ============================================================================
// 16. Offset preservation during construction
// ============================================================================
TEST(TensorImpl, OffsetPreservation) {
  auto storage = Storage::create(100, GetCPUAllocator());
  auto impl = make_intrusive<TensorImpl>(storage, ScalarType::Int32, std::vector<int64_t>{4}, 7);
  EXPECT_EQ(impl->storage_offset(), 7);
}

} // namespace
