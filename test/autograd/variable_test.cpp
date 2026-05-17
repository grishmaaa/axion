// ============================================================================
// Tests for Phase 4 — Autograd backward functions + numerical grad check
// ============================================================================

#include <gtest/gtest.h>
#include <cmath>

#include "autograd/Variable.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/engine/Engine.h"
#include "aten/ops/Ops.h"

namespace {

using c10::Tensor;

/// Numerical gradient check for a scalar function f : R^n -> R.
/// Computes (f(x+h*ei) - f(x-h*ei)) / (2h) for each element i.
/// Returns a tensor of the same shape as x with numerical gradients.
template <typename Fn>
Tensor numerical_grad(Fn f, Tensor& x, float h = 1e-3f) {
  int64_t n = x.numel();
  auto ng = c10::Tensor::empty(
      std::vector<int64_t>(x.sizes().begin(), x.sizes().end()), x.dtype());
  float* xp = static_cast<float*>(x.data_ptr());
  float* gp = static_cast<float*>(ng.data_ptr());

  for (int64_t i = 0; i < n; ++i) {
    float orig = xp[i];
    xp[i] = orig + h;
    auto tp = f(x);
    float fplus = static_cast<float*>(tp.data_ptr())[0];
    xp[i] = orig - h;
    auto tm = f(x);
    float fminus = static_cast<float*>(tm.data_ptr())[0];
    xp[i] = orig;
    gp[i] = (fplus - fminus) / (2.0f * h);
  }
  return ng;
}

/// Helper: run backward and return gradient of x
Tensor get_grad(Tensor& x, const Tensor& loss) {
  autograd::Engine::backward(loss);
  return autograd::mutable_grad(x);
}

/// Helper: clear gradient
void zero_grad(Tensor& x) {
  auto* meta = autograd::get_autograd_meta(x);
  if (meta) meta->grad_ = Tensor();
}

// ============================================================================
// Add
// ============================================================================
TEST(AutogradOps, Add) {
  auto x = aten::full({3}, 2.0f);
  autograd::set_requires_grad(x, true);
  auto y = aten::full({3}, 3.0f);
  autograd::set_requires_grad(y, true);

  auto z = autograd::sum(autograd::add(x, y));
  autograd::Engine::backward(z);

  // d(sum(x+y))/dx = 1 for each element
  auto gx = autograd::mutable_grad(x);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_NEAR(gx.data_ptr<float>()[i], 1.0f, 1e-4f);
}

// ============================================================================
// Sub
// ============================================================================
TEST(AutogradOps, Sub) {
  auto x = aten::full({3}, 2.0f);
  autograd::set_requires_grad(x, true);
  auto y = aten::full({3}, 3.0f);
  autograd::set_requires_grad(y, true);

  auto z = autograd::sum(autograd::sub(x, y));
  autograd::Engine::backward(z);

  auto gx = autograd::mutable_grad(x);
  auto gy = autograd::mutable_grad(y);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(gx.data_ptr<float>()[i], 1.0f, 1e-4f);
    EXPECT_NEAR(gy.data_ptr<float>()[i], -1.0f, 1e-4f);
  }
}

// ============================================================================
// Mul with numerical gradient check
// ============================================================================
TEST(AutogradOps, MulNumericalCheck) {
  auto x = aten::full({4}, 3.0f);
  autograd::set_requires_grad(x, true);

  auto f = [](Tensor& t) {
    return aten::sum(aten::mul(t, t));  // f(x) = sum(x^2)
  };

  auto z = autograd::sum(autograd::mul(x, x));
  autograd::Engine::backward(z);
  auto ag = autograd::mutable_grad(x);

  auto ng = numerical_grad(f, x);

  for (int64_t i = 0; i < 4; ++i)
    EXPECT_NEAR(ag.data_ptr<float>()[i], ng.data_ptr<float>()[i], 1e-2f);
}

// ============================================================================
// Div
// ============================================================================
TEST(AutogradOps, Div) {
  auto x = aten::full({}, 6.0f);
  auto y = aten::full({}, 3.0f);
  autograd::set_requires_grad(x, true);
  autograd::set_requires_grad(y, true);

  auto z = autograd::div(x, y);  // z = 6/3 = 2
  autograd::Engine::backward(z);

  // dz/dx = 1/y = 1/3
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0], 1.0f/3.0f, 1e-4f);
  // dz/dy = -x/y^2 = -6/9
  EXPECT_NEAR(autograd::mutable_grad(y).data_ptr<float>()[0], -6.0f/9.0f, 1e-4f);
}

// ============================================================================
// Neg
// ============================================================================
TEST(AutogradOps, Neg) {
  auto x = aten::full({}, 5.0f);
  autograd::set_requires_grad(x, true);
  auto z = autograd::neg(x);
  autograd::Engine::backward(z);
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0], -1.0f, 1e-4f);
}

// ============================================================================
// Exp with numerical check
// ============================================================================
TEST(AutogradOps, Exp) {
  auto x = aten::full({}, 1.0f);
  autograd::set_requires_grad(x, true);

  auto z = autograd::exp(x);
  autograd::Engine::backward(z);

  // d(exp(x))/dx = exp(x) = e
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0],
              std::exp(1.0f), 1e-4f);
}

// ============================================================================
// Log
// ============================================================================
TEST(AutogradOps, Log) {
  auto x = aten::full({}, 2.0f);
  autograd::set_requires_grad(x, true);
  auto z = autograd::log(x);
  autograd::Engine::backward(z);
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0], 0.5f, 1e-4f);
}

// ============================================================================
// Sqrt
// ============================================================================
TEST(AutogradOps, Sqrt) {
  auto x = aten::full({}, 4.0f);
  autograd::set_requires_grad(x, true);
  auto z = autograd::sqrt(x);
  autograd::Engine::backward(z);
  // d(sqrt(x))/dx = 1/(2*sqrt(x)) = 1/4
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0], 0.25f, 1e-4f);
}

// ============================================================================
// Tanh
// ============================================================================
TEST(AutogradOps, Tanh) {
  auto x = aten::full({}, 0.5f);
  autograd::set_requires_grad(x, true);
  auto z = autograd::tanh(x);
  autograd::Engine::backward(z);
  float t = std::tanh(0.5f);
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0],
              1.0f - t * t, 1e-4f);
}

// ============================================================================
// Sigmoid
// ============================================================================
TEST(AutogradOps, Sigmoid) {
  auto x = aten::full({}, 1.0f);
  autograd::set_requires_grad(x, true);
  auto z = autograd::sigmoid(x);
  autograd::Engine::backward(z);
  float s = 1.0f / (1.0f + std::exp(-1.0f));
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0],
              s * (1.0f - s), 1e-4f);
}

// ============================================================================
// Gelu with numerical check
// ============================================================================
TEST(AutogradOps, GeluNumerical) {
  auto x = aten::full({}, 0.5f);
  autograd::set_requires_grad(x, true);

  auto f = [](Tensor& t) { return aten::gelu(t); };

  auto z = autograd::gelu(x);
  autograd::Engine::backward(z);

  auto ng = numerical_grad(f, x, 1e-3f);
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0],
              ng.data_ptr<float>()[0], 1e-2f);
}

// ============================================================================
// Scalar ops
// ============================================================================
TEST(AutogradOps, MulScalar) {
  auto x = aten::full({}, 3.0f);
  autograd::set_requires_grad(x, true);
  auto z = autograd::mul_scalar(x, 5.0);
  autograd::Engine::backward(z);
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0], 5.0f, 1e-4f);
}

TEST(AutogradOps, DivScalar) {
  auto x = aten::full({}, 6.0f);
  autograd::set_requires_grad(x, true);
  auto z = autograd::div_scalar(x, 3.0);
  autograd::Engine::backward(z);
  EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[0], 1.0f/3.0f, 1e-4f);
}

// ============================================================================
// Mean
// ============================================================================
TEST(AutogradOps, Mean) {
  auto x = aten::full({4}, 3.0f);
  autograd::set_requires_grad(x, true);
  auto z = autograd::mean(x);
  autograd::Engine::backward(z);
  for (int64_t i = 0; i < 4; ++i)
    EXPECT_NEAR(autograd::mutable_grad(x).data_ptr<float>()[i], 0.25f, 1e-4f);
}

// ============================================================================
// Matmul
// ============================================================================
TEST(AutogradOps, Matmul) {
  // A(2x3) @ B(3x2) = C(2x2)
  auto a = aten::ones({2, 3});
  auto b = aten::full({3, 2}, 2.0f);
  autograd::set_requires_grad(a, true);
  autograd::set_requires_grad(b, true);

  auto c = autograd::matmul(a, b);
  auto loss = autograd::sum(c);  // scalar
  autograd::Engine::backward(loss);

  // dL/dA = grad @ B^T  (grad is ones(2x2), B^T is (2x3) filled with 2)
  // Each element of dL/dA = sum over row of (1*2) = 2*2 = 4? No.
  // grad = ones(2,2), B^T = full(2,3, val=2)
  // dA[i][j] = sum_k grad[i][k] * B^T[k][j] = sum_k 1 * 2 = 2*2 = 4
  // Wait: B^T is (2,3). dA = grad(2,2) @ B^T(2,3) = (2,3).
  // dA[i][j] = sum_k grad[i][k] * B[j][k] = sum over k of 1*2 = 2+2 = 4? No.
  // grad is (2,2) ones. B^T is (2,3) all 2s. matmul (2,2)@(2,3) = (2,3)
  // Each element = 1*2 + 1*2 = 4.
  auto ga = autograd::mutable_grad(a);
  EXPECT_EQ(ga.size(0), 2);
  EXPECT_EQ(ga.size(1), 3);
  EXPECT_NEAR(ga.data_ptr<float>()[0], 4.0f, 1e-4f);

  // dL/dB = A^T @ grad. A^T(3,2), grad(2,2). Result (3,2).
  // Each element = 1*1 + 1*1 = 2.
  auto gb = autograd::mutable_grad(b);
  EXPECT_EQ(gb.size(0), 3);
  EXPECT_EQ(gb.size(1), 2);
  EXPECT_NEAR(gb.data_ptr<float>()[0], 2.0f, 1e-4f);
}

// ============================================================================
// Chained ops: f(x) = sum(relu(x * 2 + 1))
// ============================================================================
TEST(AutogradOps, ChainedOps) {
  auto x = c10::Tensor::empty({4}, c10::ScalarType::Float32);
  x.data_ptr<float>()[0] = -1.0f;
  x.data_ptr<float>()[1] = 0.5f;
  x.data_ptr<float>()[2] = 1.0f;
  x.data_ptr<float>()[3] = -0.5f;
  autograd::set_requires_grad(x, true);

  auto y = autograd::mul_scalar(x, 2.0);  // [-2, 1, 2, -1]
  auto z = autograd::add_scalar(y, 1.0);  // [-1, 2, 3, 0]
  auto r = autograd::relu(z);             // [0, 2, 3, 0]
  auto loss = autograd::sum(r);           // 5

  autograd::Engine::backward(loss);

  auto g = autograd::mutable_grad(x);
  // d(relu(2x+1))/dx = 2 if 2x+1 > 0, else 0
  // x=-1: 2(-1)+1=-1<0 => 0
  // x=0.5: 2(0.5)+1=2>0 => 2
  // x=1: 2(1)+1=3>0 => 2
  // x=-0.5: 2(-0.5)+1=0=0 => 0 (relu at 0 has grad 0)
  EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-4f);
  EXPECT_NEAR(g.data_ptr<float>()[1], 2.0f, 1e-4f);
  EXPECT_NEAR(g.data_ptr<float>()[2], 2.0f, 1e-4f);
  EXPECT_NEAR(g.data_ptr<float>()[3], 0.0f, 1e-4f);
}

// ============================================================================
// NoGrad: ops should NOT record when GradMode is off
// ============================================================================
TEST(AutogradOps, NoGradDoesNotRecord) {
  auto x = aten::full({}, 3.0f);
  autograd::set_requires_grad(x, true);

  c10::Tensor y;
  {
    autograd::NoGradGuard guard;
    y = autograd::mul_scalar(x, 2.0);
  }
  // y should NOT have a grad_fn
  EXPECT_EQ(autograd::grad_fn(y), nullptr);
}

}  // namespace
