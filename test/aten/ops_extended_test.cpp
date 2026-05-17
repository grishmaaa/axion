// ============================================================================
// Tests for Phase 2 extended ATen ops
// ============================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "aten/ops/Ops.h"
#include "c10/core/Dispatcher.h"

namespace {

// Helper: get float value at index i
inline float f(const c10::Tensor& t, int64_t i) {
  return t.data_ptr<float>()[i];
}

// ============================================================================
// Unary ops
// ============================================================================

TEST(OpsExtended, Exp) {
  auto a = aten::full({4}, 1.0f);
  auto r = aten::exp(a);
  EXPECT_NEAR(f(r, 0), std::exp(1.0f), 1e-5f);
}

TEST(OpsExtended, Log) {
  auto a = aten::full({4}, 2.718281828f);
  auto r = aten::log(a);
  EXPECT_NEAR(f(r, 0), std::log(2.718281828f), 1e-5f);
}

TEST(OpsExtended, Sqrt) {
  auto a = aten::full({4}, 9.0f);
  auto r = aten::sqrt(a);
  EXPECT_NEAR(f(r, 0), 3.0f, 1e-5f);
}

TEST(OpsExtended, Tanh) {
  auto a = aten::full({4}, 0.5f);
  auto r = aten::tanh(a);
  EXPECT_NEAR(f(r, 0), std::tanh(0.5f), 1e-5f);
}

TEST(OpsExtended, Sigmoid) {
  auto a = aten::zeros({4});
  auto r = aten::sigmoid(a);
  EXPECT_NEAR(f(r, 0), 0.5f, 1e-5f);  // sigmoid(0) = 0.5
}

TEST(OpsExtended, Gelu) {
  auto a = aten::zeros({4});
  auto r = aten::gelu(a);
  EXPECT_NEAR(f(r, 0), 0.0f, 1e-5f);  // gelu(0) = 0

  auto b = aten::full({4}, 1.0f);
  auto r2 = aten::gelu(b);
  float expected = 1.0f * 0.5f * (1.0f + std::erf(1.0f / std::sqrt(2.0f)));
  EXPECT_NEAR(f(r2, 0), expected, 1e-5f);
}

// ============================================================================
// Binary ops
// ============================================================================

TEST(OpsExtended, Div) {
  auto a = aten::full({4}, 6.0f);
  auto b = aten::full({4}, 2.0f);
  auto r = aten::div(a, b);
  EXPECT_NEAR(f(r, 0), 3.0f, 1e-5f);
}

TEST(OpsExtended, Pow) {
  auto a = aten::full({4}, 2.0f);
  auto b = aten::full({4}, 3.0f);
  auto r = aten::pow(a, b);
  EXPECT_NEAR(f(r, 0), 8.0f, 1e-5f);
}

// ============================================================================
// Scalar ops
// ============================================================================

TEST(OpsExtended, AddScalar) {
  auto a = aten::full({4}, 3.0f);
  auto r = aten::add_scalar(a, 2.0);
  EXPECT_NEAR(f(r, 0), 5.0f, 1e-5f);
}

TEST(OpsExtended, SubScalar) {
  auto a = aten::full({4}, 5.0f);
  auto r = aten::sub_scalar(a, 2.0);
  EXPECT_NEAR(f(r, 0), 3.0f, 1e-5f);
}

TEST(OpsExtended, MulScalar) {
  auto a = aten::full({4}, 3.0f);
  auto r = aten::mul_scalar(a, 4.0);
  EXPECT_NEAR(f(r, 0), 12.0f, 1e-5f);
}

TEST(OpsExtended, DivScalar) {
  auto a = aten::full({4}, 12.0f);
  auto r = aten::div_scalar(a, 3.0);
  EXPECT_NEAR(f(r, 0), 4.0f, 1e-5f);
}

// ============================================================================
// Reduce ops
// ============================================================================

TEST(OpsExtended, Mean) {
  auto a = aten::full({4}, 3.0f);
  auto r = aten::mean(a);
  EXPECT_NEAR(f(r, 0), 3.0f, 1e-5f);

  // Different values
  auto b = c10::Tensor::empty({4}, c10::ScalarType::Float32);
  b.data_ptr<float>()[0] = 1.0f;
  b.data_ptr<float>()[1] = 2.0f;
  b.data_ptr<float>()[2] = 3.0f;
  b.data_ptr<float>()[3] = 4.0f;
  auto r2 = aten::mean(b);
  EXPECT_NEAR(f(r2, 0), 2.5f, 1e-5f);
}

// ============================================================================
// Creation ops
// ============================================================================

TEST(OpsExtended, Arange) {
  auto r = aten::arange(0.0, 5.0, 1.0);
  EXPECT_EQ(r.numel(), 5);
  EXPECT_NEAR(f(r, 0), 0.0f, 1e-5f);
  EXPECT_NEAR(f(r, 4), 4.0f, 1e-5f);
}

TEST(OpsExtended, ArangeStep) {
  auto r = aten::arange(1.0, 4.0, 0.5);
  EXPECT_EQ(r.numel(), 6);
  EXPECT_NEAR(f(r, 0), 1.0f, 1e-5f);
  EXPECT_NEAR(f(r, 5), 3.5f, 1e-5f);
}

TEST(OpsExtended, Eye) {
  auto r = aten::eye(3);
  EXPECT_EQ(r.ndim(), 2);
  EXPECT_EQ(r.size(0), 3);
  EXPECT_EQ(r.size(1), 3);
  // Diagonal = 1, off-diagonal = 0
  float* p = r.data_ptr<float>();
  EXPECT_NEAR(p[0], 1.0f, 1e-5f);  // (0,0)
  EXPECT_NEAR(p[1], 0.0f, 1e-5f);  // (0,1)
  EXPECT_NEAR(p[4], 1.0f, 1e-5f);  // (1,1)
  EXPECT_NEAR(p[8], 1.0f, 1e-5f);  // (2,2)
}

// ============================================================================
// Shape ops
// ============================================================================

TEST(OpsExtended, Transpose) {
  // Create a 2x3 tensor: [[1,2,3],[4,5,6]]
  auto a = c10::Tensor::empty({2, 3}, c10::ScalarType::Float32);
  float* p = a.data_ptr<float>();
  for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i + 1);

  auto t = aten::transpose(a, 0, 1);
  // Should be 3x2
  EXPECT_EQ(t.size(0), 3);
  EXPECT_EQ(t.size(1), 2);
  // Not contiguous (strides are swapped)
  EXPECT_FALSE(t.is_contiguous());
  // Same storage
  EXPECT_EQ(a.data_ptr(), t.data_ptr());
}

TEST(OpsExtended, Reshape) {
  auto a = aten::arange(0.0, 6.0, 1.0);  // [0,1,2,3,4,5]
  auto r = aten::reshape(a, {2, 3});
  EXPECT_EQ(r.size(0), 2);
  EXPECT_EQ(r.size(1), 3);
  EXPECT_TRUE(r.is_contiguous());
  EXPECT_NEAR(f(r, 0), 0.0f, 1e-5f);
  EXPECT_NEAR(f(r, 5), 5.0f, 1e-5f);
}

TEST(OpsExtended, ReshapeInfer) {
  auto a = aten::arange(0.0, 12.0, 1.0);
  auto r = aten::reshape(a, {3, -1});
  EXPECT_EQ(r.size(0), 3);
  EXPECT_EQ(r.size(1), 4);
}

TEST(OpsExtended, View) {
  auto a = aten::arange(0.0, 6.0, 1.0);
  auto r = aten::view(a, {2, 3});
  EXPECT_EQ(r.size(0), 2);
  EXPECT_EQ(r.size(1), 3);
}

TEST(OpsExtended, Contiguous) {
  auto a = aten::arange(0.0, 6.0, 1.0);
  auto r = aten::reshape(a, {2, 3});
  auto t = aten::transpose(r, 0, 1);  // 3x2, non-contiguous

  auto c = aten::contiguous(t);
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_EQ(c.size(0), 3);
  EXPECT_EQ(c.size(1), 2);

  // Data should be: [[0,3],[1,4],[2,5]]
  float* p = c.data_ptr<float>();
  EXPECT_NEAR(p[0], 0.0f, 1e-5f);  // (0,0) of transposed
  EXPECT_NEAR(p[1], 3.0f, 1e-5f);  // (0,1) of transposed
  EXPECT_NEAR(p[2], 1.0f, 1e-5f);  // (1,0) of transposed
}

TEST(OpsExtended, ContiguousNoop) {
  auto a = aten::ones({3, 4});
  auto c = aten::contiguous(a);
  EXPECT_EQ(a.data_ptr(), c.data_ptr());  // Same pointer — no copy
}

// ============================================================================
// Dispatch routing still works
// ============================================================================

TEST(OpsExtended, AllNewOpsRegistered) {
  auto& d = c10::Dispatcher::singleton();
  EXPECT_TRUE(d.hasOp("aten::exp"));
  EXPECT_TRUE(d.hasOp("aten::log"));
  EXPECT_TRUE(d.hasOp("aten::sqrt"));
  EXPECT_TRUE(d.hasOp("aten::tanh"));
  EXPECT_TRUE(d.hasOp("aten::sigmoid"));
  EXPECT_TRUE(d.hasOp("aten::gelu"));
  EXPECT_TRUE(d.hasOp("aten::div"));
  EXPECT_TRUE(d.hasOp("aten::pow"));
  EXPECT_TRUE(d.hasOp("aten::add_scalar"));
  EXPECT_TRUE(d.hasOp("aten::mean"));
  EXPECT_TRUE(d.hasOp("aten::arange"));
  EXPECT_TRUE(d.hasOp("aten::eye"));
  EXPECT_TRUE(d.hasOp("aten::contiguous"));
}

}  // namespace
