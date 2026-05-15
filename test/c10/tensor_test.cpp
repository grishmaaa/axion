// ============================================================================
// Tests for c10::Tensor — the user-facing tensor handle
// ============================================================================
//
// 16 test cases covering:
//   1.  Default construction is undefined
//   2.  Tensor::empty creates valid tensor with correct shape
//   3.  sizes() returns correct values
//   4.  strides() returns correct values (auto-computed, contiguous)
//   5.  dtype() returns correct type
//   6.  numel() returns correct product of sizes
//   7.  ndim() returns correct dimensions
//   8.  is_contiguous() true for freshly allocated tensor
//   9.  data_ptr() returns non-null for valid tensor
//  10.  Typed data_ptr<float>() read/write
//  11.  device() returns CPU
//  12.  Copy shares ownership (refcount increment)
//  13.  Move transfers ownership (source becomes undefined)
//  14.  Equality comparison (same impl equal, different not)
//  15.  from_blob wraps external memory correctly
//  16.  size(dim) and stride(dim) per-dimension accessors

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "c10/core/CPUAllocator.h"
#include "c10/core/Tensor.h"

namespace {

using namespace c10;

// ============================================================================
// 1. Default construction is undefined
// ============================================================================
TEST(Tensor, DefaultConstructionIsUndefined) {
  Tensor t;
  EXPECT_FALSE(t.defined());
  EXPECT_FALSE(static_cast<bool>(t));
}

// ============================================================================
// 2. Tensor::empty creates valid tensor with correct shape
// ============================================================================
TEST(Tensor, EmptyCreatesValid) {
  auto t = Tensor::empty({3, 4}, ScalarType::Float32);
  EXPECT_TRUE(t.defined());
  EXPECT_NE(t.data_ptr(), nullptr);
  EXPECT_EQ(t.sizes().size(), 2u);
  EXPECT_EQ(t.sizes()[0], 3);
  EXPECT_EQ(t.sizes()[1], 4);
}

// ============================================================================
// 3. sizes() returns correct values
// ============================================================================
TEST(Tensor, SizesAccessor) {
  auto t = Tensor::empty({2, 5, 3}, ScalarType::Float32);
  std::vector<int64_t> expected = {2, 5, 3};
  EXPECT_EQ(t.sizes(), expected);
}

// ============================================================================
// 4. strides() auto-computed for contiguous layout
// ============================================================================
TEST(Tensor, StridesAutoComputed) {
  auto t = Tensor::empty({3, 4}, ScalarType::Float32);
  // Row-major contiguous: strides = [4, 1]
  std::vector<int64_t> expected = {4, 1};
  EXPECT_EQ(t.strides(), expected);
}

// ============================================================================
// 5. dtype() returns correct type
// ============================================================================
TEST(Tensor, DtypeAccessor) {
  auto t1 = Tensor::empty({2}, ScalarType::Float32);
  EXPECT_EQ(t1.dtype(), ScalarType::Float32);

  auto t2 = Tensor::empty({2}, ScalarType::Int64);
  EXPECT_EQ(t2.dtype(), ScalarType::Int64);
}

// ============================================================================
// 6. numel() returns correct product of sizes
// ============================================================================
TEST(Tensor, NumelAccessor) {
  auto t = Tensor::empty({3, 4, 5}, ScalarType::Float32);
  EXPECT_EQ(t.numel(), 60);
}

// ============================================================================
// 7. ndim() returns correct dimensions
// ============================================================================
TEST(Tensor, NdimAccessor) {
  auto t0 = Tensor::empty({}, ScalarType::Float32);       // scalar
  auto t1 = Tensor::empty({10}, ScalarType::Float32);     // 1D
  auto t3 = Tensor::empty({2, 3, 4}, ScalarType::Float32); // 3D

  EXPECT_EQ(t0.ndim(), 0);
  EXPECT_EQ(t1.ndim(), 1);
  EXPECT_EQ(t3.ndim(), 3);
}

// ============================================================================
// 8. is_contiguous() true for freshly allocated tensor
// ============================================================================
TEST(Tensor, IsContiguousFresh) {
  auto t = Tensor::empty({4, 5, 6}, ScalarType::Float32);
  EXPECT_TRUE(t.is_contiguous());
}

// ============================================================================
// 9. data_ptr() returns non-null for valid tensor
// ============================================================================
TEST(Tensor, DataPtrNonNull) {
  auto t = Tensor::empty({8}, ScalarType::Float32);
  EXPECT_NE(t.data_ptr(), nullptr);
}

// ============================================================================
// 10. Typed data_ptr<float>() read/write round-trip
// ============================================================================
TEST(Tensor, TypedDataPtrReadWrite) {
  auto t = Tensor::empty({4}, ScalarType::Float32);
  float* p = t.data_ptr<float>();
  ASSERT_NE(p, nullptr);

  p[0] = 1.0f;
  p[1] = 2.0f;
  p[2] = 3.0f;
  p[3] = 4.0f;

  EXPECT_FLOAT_EQ(t.data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(t.data_ptr<float>()[1], 2.0f);
  EXPECT_FLOAT_EQ(t.data_ptr<float>()[2], 3.0f);
  EXPECT_FLOAT_EQ(t.data_ptr<float>()[3], 4.0f);
}

// ============================================================================
// 11. device() returns CPU
// ============================================================================
TEST(Tensor, DeviceIsCPU) {
  auto t = Tensor::empty({2, 3}, ScalarType::Float32);
  EXPECT_TRUE(t.device().is_cpu());
  EXPECT_EQ(t.device().index, 0);
}

// ============================================================================
// 12. Copy shares ownership — refcount increments
// ============================================================================
TEST(Tensor, CopySharesOwnership) {
  auto t1 = Tensor::empty({3, 4}, ScalarType::Float32);
  EXPECT_EQ(t1.use_count(), 1u);
  EXPECT_TRUE(t1.unique());

  auto t2 = t1;  // copy
  EXPECT_EQ(t1.use_count(), 2u);
  EXPECT_EQ(t2.use_count(), 2u);
  EXPECT_FALSE(t1.unique());

  // Same underlying data
  EXPECT_EQ(t1.data_ptr(), t2.data_ptr());
  EXPECT_EQ(t1, t2);  // identity equal
}

// ============================================================================
// 13. Move transfers ownership — source becomes undefined
// ============================================================================
TEST(Tensor, MoveTransfersOwnership) {
  auto t1 = Tensor::empty({5}, ScalarType::Float32);
  void* orig = t1.data_ptr();

  auto t2 = std::move(t1);
  EXPECT_TRUE(t2.defined());
  EXPECT_EQ(t2.data_ptr(), orig);
  EXPECT_EQ(t2.use_count(), 1u);
  EXPECT_FALSE(t1.defined());  // NOLINT: intentional use-after-move
}

// ============================================================================
// 14. Equality — same impl equal, different impl not equal
// ============================================================================
TEST(Tensor, EqualityComparison) {
  auto t1 = Tensor::empty({4}, ScalarType::Float32);
  auto t2 = t1;  // copy — same TensorImpl
  auto t3 = Tensor::empty({4}, ScalarType::Float32);  // different impl

  EXPECT_EQ(t1, t2);
  EXPECT_NE(t1, t3);
}

// ============================================================================
// 15. from_blob wraps external memory correctly
// ============================================================================
TEST(Tensor, FromBlobWrapsMemory) {
  float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto t = Tensor::from_blob(data, {2, 3}, ScalarType::Float32);

  EXPECT_TRUE(t.defined());
  EXPECT_EQ(t.data_ptr(), static_cast<void*>(data));
  EXPECT_EQ(t.numel(), 6);
  EXPECT_TRUE(t.is_contiguous());

  // Read through tensor matches original array
  float* p = t.data_ptr<float>();
  EXPECT_FLOAT_EQ(p[0], 1.0f);
  EXPECT_FLOAT_EQ(p[5], 6.0f);

  // Write through tensor modifies original array
  p[0] = 42.0f;
  EXPECT_FLOAT_EQ(data[0], 42.0f);
}

// ============================================================================
// 16. size(dim) and stride(dim) per-dimension accessors
// ============================================================================
TEST(Tensor, PerDimensionAccessors) {
  auto t = Tensor::empty({2, 3, 4}, ScalarType::Float32);

  // sizes
  EXPECT_EQ(t.size(0), 2);
  EXPECT_EQ(t.size(1), 3);
  EXPECT_EQ(t.size(2), 4);

  // strides — contiguous: [3*4, 4, 1] = [12, 4, 1]
  EXPECT_EQ(t.stride(0), 12);
  EXPECT_EQ(t.stride(1), 4);
  EXPECT_EQ(t.stride(2), 1);
}

}  // namespace
