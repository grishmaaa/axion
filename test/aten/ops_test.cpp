// ============================================================================
// Tests for aten::Ops — the ATen public API
// ============================================================================
//
// 16 test cases covering:
//   1.  zeros — all elements are 0
//   2.  ones — all elements are 1
//   3.  full — all elements match the fill value
//   4.  rand — all values in [0,1), not all-zero
//   5.  add — element-wise correctness
//   6.  sub — element-wise correctness
//   7.  mul — element-wise correctness
//   8.  neg — element-wise correctness
//   9.  relu — negatives become 0, positives unchanged
//  10.  abs — negatives flipped, positives unchanged
//  11.  sum — correct scalar result, ndim==0, numel==1
//  12.  matmul — 2x3 times 3x2, verify all four output elements
//  13.  matmul — output shape is {a.size(0), b.size(1)}
//  14.  Shape mismatch on binary op — death
//  15.  Output is always a new tensor — identity check
//  16.  Dtype preserved through ops

#include <gtest/gtest.h>

#include "aten/ops/Ops.h"
#include "c10/core/CPUAllocator.h"

namespace {

using namespace c10;

// Helper: create a tensor from a vector of floats.
static Tensor from_vec(const std::vector<float>& v) {
  auto t = Tensor::empty({static_cast<int64_t>(v.size())}, ScalarType::Float32);
  float* p = t.data_ptr<float>();
  for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return t;
}

// ============================================================================
// 1. zeros — all elements are 0
// ============================================================================
TEST(Ops, Zeros) {
  auto t = aten::zeros({3, 4}, ScalarType::Float32);
  EXPECT_EQ(t.numel(), 12);
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ(p[i], 0.0f) << "Element " << i;
  }
}

// ============================================================================
// 2. ones — all elements are 1
// ============================================================================
TEST(Ops, Ones) {
  auto t = aten::ones({2, 3}, ScalarType::Float32);
  EXPECT_EQ(t.numel(), 6);
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(p[i], 1.0f) << "Element " << i;
  }
}

// ============================================================================
// 3. full — all elements match the fill value
// ============================================================================
TEST(Ops, Full) {
  auto t = aten::full({5}, 3.14, ScalarType::Float32);
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(p[i], 3.14f) << "Element " << i;
  }
}

// ============================================================================
// 4. rand — all values in [0, 1), not all-zero
// ============================================================================
TEST(Ops, Rand) {
  auto t = aten::rand({100}, ScalarType::Float32);
  float* p = t.data_ptr<float>();
  bool any_nonzero = false;
  for (int64_t i = 0; i < 100; ++i) {
    EXPECT_GE(p[i], 0.0f) << "Element " << i << " below 0";
    EXPECT_LT(p[i], 1.0f) << "Element " << i << " >= 1";
    if (p[i] != 0.0f) any_nonzero = true;
  }
  EXPECT_TRUE(any_nonzero) << "All values are zero — likely not random";
}

// ============================================================================
// 5. add — element-wise correctness
// ============================================================================
TEST(Ops, Add) {
  auto a = from_vec({1.0f, 2.0f, 3.0f});
  auto b = from_vec({10.0f, 20.0f, 30.0f});
  auto c = aten::add(a, b);
  float* p = c.data_ptr<float>();
  EXPECT_FLOAT_EQ(p[0], 11.0f);
  EXPECT_FLOAT_EQ(p[1], 22.0f);
  EXPECT_FLOAT_EQ(p[2], 33.0f);
}

// ============================================================================
// 6. sub — element-wise correctness
// ============================================================================
TEST(Ops, Sub) {
  auto a = from_vec({10.0f, 20.0f, 30.0f});
  auto b = from_vec({1.0f, 2.0f, 3.0f});
  auto c = aten::sub(a, b);
  float* p = c.data_ptr<float>();
  EXPECT_FLOAT_EQ(p[0], 9.0f);
  EXPECT_FLOAT_EQ(p[1], 18.0f);
  EXPECT_FLOAT_EQ(p[2], 27.0f);
}

// ============================================================================
// 7. mul — element-wise correctness
// ============================================================================
TEST(Ops, Mul) {
  auto a = from_vec({2.0f, 3.0f, 4.0f});
  auto b = from_vec({5.0f, 6.0f, 7.0f});
  auto c = aten::mul(a, b);
  float* p = c.data_ptr<float>();
  EXPECT_FLOAT_EQ(p[0], 10.0f);
  EXPECT_FLOAT_EQ(p[1], 18.0f);
  EXPECT_FLOAT_EQ(p[2], 28.0f);
}

// ============================================================================
// 8. neg — element-wise correctness
// ============================================================================
TEST(Ops, Neg) {
  auto a = from_vec({1.0f, -2.0f, 0.0f, 3.5f});
  auto c = aten::neg(a);
  float* p = c.data_ptr<float>();
  EXPECT_FLOAT_EQ(p[0], -1.0f);
  EXPECT_FLOAT_EQ(p[1], 2.0f);
  EXPECT_FLOAT_EQ(p[2], 0.0f);
  EXPECT_FLOAT_EQ(p[3], -3.5f);
}

// ============================================================================
// 9. relu — negatives become 0, positives unchanged
// ============================================================================
TEST(Ops, Relu) {
  auto a = from_vec({-3.0f, -1.0f, 0.0f, 1.0f, 5.0f});
  auto c = aten::relu(a);
  float* p = c.data_ptr<float>();
  EXPECT_FLOAT_EQ(p[0], 0.0f);
  EXPECT_FLOAT_EQ(p[1], 0.0f);
  EXPECT_FLOAT_EQ(p[2], 0.0f);
  EXPECT_FLOAT_EQ(p[3], 1.0f);
  EXPECT_FLOAT_EQ(p[4], 5.0f);
}

// ============================================================================
// 10. abs — negatives flipped, positives unchanged
// ============================================================================
TEST(Ops, Abs) {
  auto a = from_vec({-3.0f, -1.5f, 0.0f, 2.0f});
  auto c = aten::abs(a);
  float* p = c.data_ptr<float>();
  EXPECT_FLOAT_EQ(p[0], 3.0f);
  EXPECT_FLOAT_EQ(p[1], 1.5f);
  EXPECT_FLOAT_EQ(p[2], 0.0f);
  EXPECT_FLOAT_EQ(p[3], 2.0f);
}

// ============================================================================
// 11. sum — correct scalar result, ndim==0, numel==1
// ============================================================================
TEST(Ops, Sum) {
  auto a = from_vec({1.0f, 2.0f, 3.0f, 4.0f});
  auto s = aten::sum(a);
  EXPECT_EQ(s.ndim(), 0);
  EXPECT_EQ(s.numel(), 1);
  EXPECT_FLOAT_EQ(s.data_ptr<float>()[0], 10.0f);
}

// ============================================================================
// 12. matmul — 2x3 times 3x2, verify all four output elements
// ============================================================================
TEST(Ops, MatmulCorrectness) {
  // A = [[1, 2, 3],    B = [[7, 8],
  //      [4, 5, 6]]         [9, 10],
  //                          [11, 12]]
  //
  // C = A @ B = [[1*7+2*9+3*11, 1*8+2*10+3*12],   = [[58,  64],
  //              [4*7+5*9+6*11, 4*8+5*10+6*12]]      [139, 154]]
  auto a = Tensor::empty({2, 3}, ScalarType::Float32);
  float* ap = a.data_ptr<float>();
  ap[0]=1; ap[1]=2; ap[2]=3; ap[3]=4; ap[4]=5; ap[5]=6;

  auto b = Tensor::empty({3, 2}, ScalarType::Float32);
  float* bp = b.data_ptr<float>();
  bp[0]=7; bp[1]=8; bp[2]=9; bp[3]=10; bp[4]=11; bp[5]=12;

  auto c = aten::matmul(a, b);
  float* cp = c.data_ptr<float>();
  EXPECT_FLOAT_EQ(cp[0], 58.0f);
  EXPECT_FLOAT_EQ(cp[1], 64.0f);
  EXPECT_FLOAT_EQ(cp[2], 139.0f);
  EXPECT_FLOAT_EQ(cp[3], 154.0f);
}

// ============================================================================
// 13. matmul — output shape is {a.size(0), b.size(1)}
// ============================================================================
TEST(Ops, MatmulShape) {
  auto a = Tensor::empty({4, 3}, ScalarType::Float32);
  auto b = Tensor::empty({3, 5}, ScalarType::Float32);
  auto c = aten::matmul(a, b);
  EXPECT_EQ(c.ndim(), 2);
  EXPECT_EQ(c.size(0), 4);
  EXPECT_EQ(c.size(1), 5);
}

// ============================================================================
// 14. Shape mismatch on binary op — death
// ============================================================================
TEST(Ops, BinaryShapeMismatchDeath) {
  auto a = from_vec({1.0f, 2.0f, 3.0f});
  auto b = from_vec({1.0f, 2.0f});
  EXPECT_DEATH(aten::add(a, b), "shape mismatch");
}

// ============================================================================
// 15. Output is always a new tensor — identity check
// ============================================================================
TEST(Ops, OutputIsNew) {
  auto a = from_vec({1.0f, 2.0f, 3.0f});
  auto b = from_vec({4.0f, 5.0f, 6.0f});
  auto c = aten::add(a, b);
  EXPECT_NE(c, a);
  EXPECT_NE(c, b);

  auto d = aten::neg(a);
  EXPECT_NE(d, a);
}

// ============================================================================
// 16. Dtype preserved through ops
// ============================================================================
TEST(Ops, DtypePreserved) {
  auto a = Tensor::empty({4}, ScalarType::Float64);
  auto b = Tensor::empty({4}, ScalarType::Float64);
  auto c = aten::add(a, b);
  EXPECT_EQ(c.dtype(), ScalarType::Float64);

  auto d = aten::neg(a);
  EXPECT_EQ(d.dtype(), ScalarType::Float64);

  auto s = aten::sum(a);
  EXPECT_EQ(s.dtype(), ScalarType::Float64);
}

}  // namespace
