// ============================================================================
// Tests for aten::TensorIterator (cpu_kernel_unary / cpu_kernel_binary)
// ============================================================================
//
// 10 test cases covering:
//   1.  Unary kernel on Float32 — correct output values
//   2.  Unary kernel on Float64 — correct output values
//   3.  Binary kernel on Float32 — correct output values
//   4.  Binary kernel on Float64 — correct output values
//   5.  Output is a new tensor, not the same as input
//   6.  Output shape matches input shape
//   7.  Output dtype matches input dtype
//   8.  Binary kernel with mismatched shapes — death
//   9.  Binary kernel with mismatched dtypes — death
//  10.  Unary kernel on undefined tensor — death

#include <gtest/gtest.h>

#include "aten/core/TensorIterator.h"
#include "c10/core/CPUAllocator.h"
#include "c10/core/Tensor.h"

namespace {

using namespace c10;

// ============================================================================
// 1. Unary kernel on Float32 — correct output values
// ============================================================================
TEST(TensorIterator, UnaryFloat32) {
  auto t = Tensor::empty({4}, ScalarType::Float32);
  float* p = t.data_ptr<float>();
  p[0] = 1.0f; p[1] = -2.0f; p[2] = 3.0f; p[3] = -4.0f;

  auto out = aten::cpu_kernel_unary(t, [](float x) { return -x; });

  float* o = out.data_ptr<float>();
  EXPECT_FLOAT_EQ(o[0], -1.0f);
  EXPECT_FLOAT_EQ(o[1],  2.0f);
  EXPECT_FLOAT_EQ(o[2], -3.0f);
  EXPECT_FLOAT_EQ(o[3],  4.0f);
}

// ============================================================================
// 2. Unary kernel on Float64 — correct output values
// ============================================================================
TEST(TensorIterator, UnaryFloat64) {
  auto t = Tensor::empty({3}, ScalarType::Float64);
  double* p = t.data_ptr<double>();
  p[0] = 1.5; p[1] = -2.5; p[2] = 0.0;

  auto out = aten::cpu_kernel_unary(t, [](double x) { return x * 2.0; });

  double* o = out.data_ptr<double>();
  EXPECT_DOUBLE_EQ(o[0], 3.0);
  EXPECT_DOUBLE_EQ(o[1], -5.0);
  EXPECT_DOUBLE_EQ(o[2], 0.0);
}

// ============================================================================
// 3. Binary kernel on Float32 — correct output values
// ============================================================================
TEST(TensorIterator, BinaryFloat32) {
  auto a = Tensor::empty({3}, ScalarType::Float32);
  auto b = Tensor::empty({3}, ScalarType::Float32);
  float* ap = a.data_ptr<float>();
  float* bp = b.data_ptr<float>();
  ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;
  bp[0] = 10.0f; bp[1] = 20.0f; bp[2] = 30.0f;

  auto out = aten::cpu_kernel_binary(a, b, [](float x, float y) { return x + y; });

  float* o = out.data_ptr<float>();
  EXPECT_FLOAT_EQ(o[0], 11.0f);
  EXPECT_FLOAT_EQ(o[1], 22.0f);
  EXPECT_FLOAT_EQ(o[2], 33.0f);
}

// ============================================================================
// 4. Binary kernel on Float64 — correct output values
// ============================================================================
TEST(TensorIterator, BinaryFloat64) {
  auto a = Tensor::empty({2}, ScalarType::Float64);
  auto b = Tensor::empty({2}, ScalarType::Float64);
  double* ap = a.data_ptr<double>();
  double* bp = b.data_ptr<double>();
  ap[0] = 1.5; ap[1] = 2.5;
  bp[0] = 0.5; bp[1] = 0.5;

  auto out = aten::cpu_kernel_binary(a, b, [](double x, double y) { return x * y; });

  double* o = out.data_ptr<double>();
  EXPECT_DOUBLE_EQ(o[0], 0.75);
  EXPECT_DOUBLE_EQ(o[1], 1.25);
}

// ============================================================================
// 5. Output is a new tensor, not the same as input
// ============================================================================
TEST(TensorIterator, OutputIsNewTensor) {
  auto a = Tensor::empty({4}, ScalarType::Float32);
  auto b = Tensor::empty({4}, ScalarType::Float32);

  auto out = aten::cpu_kernel_binary(a, b, [](float x, float y) { return x + y; });

  // Identity check — different TensorImpl, not same handle
  EXPECT_NE(out, a);
  EXPECT_NE(out, b);
}

// ============================================================================
// 6. Output shape matches input shape
// ============================================================================
TEST(TensorIterator, OutputShapeMatches) {
  auto t = Tensor::empty({2, 3}, ScalarType::Float32);
  auto out = aten::cpu_kernel_unary(t, [](float x) { return x; });

  EXPECT_EQ(out.sizes(), t.sizes());
}

// ============================================================================
// 7. Output dtype matches input dtype
// ============================================================================
TEST(TensorIterator, OutputDtypeMatches) {
  auto t32 = Tensor::empty({4}, ScalarType::Float32);
  auto out32 = aten::cpu_kernel_unary(t32, [](float x) { return x; });
  EXPECT_EQ(out32.dtype(), ScalarType::Float32);

  auto t64 = Tensor::empty({4}, ScalarType::Float64);
  auto out64 = aten::cpu_kernel_unary(t64, [](double x) { return x; });
  EXPECT_EQ(out64.dtype(), ScalarType::Float64);
}

// ============================================================================
// 8. Binary kernel with mismatched shapes — death
// ============================================================================
TEST(TensorIterator, BinaryShapeMismatchDeath) {
  auto a = Tensor::empty({3}, ScalarType::Float32);
  auto b = Tensor::empty({4}, ScalarType::Float32);

  EXPECT_DEATH(
      aten::cpu_kernel_binary(a, b, [](float x, float y) { return x + y; }),
      "shape mismatch");
}

// ============================================================================
// 9. Binary kernel with mismatched dtypes — death
// ============================================================================
TEST(TensorIterator, BinaryDtypeMismatchDeath) {
  auto a = Tensor::empty({3}, ScalarType::Float32);
  auto b = Tensor::empty({3}, ScalarType::Float64);

  EXPECT_DEATH(
      aten::cpu_kernel_binary(a, b, [](float x, float y) { return x + y; }),
      "dtype mismatch");
}

// ============================================================================
// 10. Unary kernel on undefined tensor — death
// ============================================================================
TEST(TensorIterator, UnaryUndefinedDeath) {
  Tensor t;  // undefined

  EXPECT_DEATH(
      aten::cpu_kernel_unary(t, [](float x) { return x; }),
      "undefined");
}

}  // namespace
