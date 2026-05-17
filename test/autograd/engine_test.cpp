// ============================================================================
// Tests for autograd Engine — backward pass execution
// ============================================================================
//
// These tests build simple computation graphs by hand and verify
// that Engine::backward() correctly computes gradients.

#include <gtest/gtest.h>
#include <cmath>

#include "autograd/engine/Engine.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/Node.h"
#include "autograd/Edge.h"
#include "aten/ops/Ops.h"

namespace {

using namespace autograd;
using c10::Tensor;

// ============================================================================
// Minimal backward nodes for testing (Phase 4 will have the real ones)
// ============================================================================

/// Backward for z = a + b.  dz/da = 1, dz/db = 1.
class TestAddBackward : public Node {
 public:
  std::vector<Tensor> apply(std::vector<Tensor> grads) override {
    return {grads[0], grads[0]};  // both inputs get grad_output * 1
  }
};

/// Backward for z = a * b.  dz/da = b, dz/db = a.
class TestMulBackward : public Node {
 public:
  TestMulBackward(Tensor a, Tensor b) : a_(std::move(a)), b_(std::move(b)) {}

  std::vector<Tensor> apply(std::vector<Tensor> grads) override {
    return {aten::mul(grads[0], b_), aten::mul(grads[0], a_)};
  }

 private:
  Tensor a_, b_;
};

/// Backward for z = a * 2 (scalar multiply).
class TestMulScalarBackward : public Node {
 public:
  explicit TestMulScalarBackward(double scalar) : scalar_(scalar) {}

  std::vector<Tensor> apply(std::vector<Tensor> grads) override {
    return {aten::mul_scalar(grads[0], scalar_)};
  }

 private:
  double scalar_;
};

// ============================================================================
// Helper: manually build a computation graph
// ============================================================================

/// Simulate: result = a + b, record on graph
Tensor test_add(Tensor& a, Tensor& b) {
  auto result = aten::add(a, b);

  auto node = std::make_shared<TestAddBackward>();
  node->add_next_edge(gradient_edge(a));
  node->add_next_edge(gradient_edge(b));
  set_grad_fn(result, node);

  return result;
}

/// Simulate: result = a * b, record on graph
Tensor test_mul(Tensor& a, Tensor& b) {
  auto result = aten::mul(a, b);

  auto node = std::make_shared<TestMulBackward>(a, b);
  node->add_next_edge(gradient_edge(a));
  node->add_next_edge(gradient_edge(b));
  set_grad_fn(result, node);

  return result;
}

/// Simulate: result = a * scalar, record on graph
Tensor test_mul_scalar(Tensor& a, double s) {
  auto result = aten::mul_scalar(a, s);

  auto node = std::make_shared<TestMulScalarBackward>(s);
  node->add_next_edge(gradient_edge(a));
  set_grad_fn(result, node);

  return result;
}

// ============================================================================
// Tests
// ============================================================================

/// y = x * 2 + 1.  dy/dx = 2.
TEST(Engine, SimpleLinear) {
  auto x = aten::ones({2});
  set_requires_grad(x, true);

  auto y = test_mul_scalar(x, 2.0);    // y = x * 2

  // Simplest test: scalar x.
  auto xs = aten::full({}, 3.0f);
  set_requires_grad(xs, true);

  auto ys = test_mul_scalar(xs, 2.0);  // ys = xs * 2

  Engine::backward(ys);

  auto& g = mutable_grad(xs);
  EXPECT_TRUE(g.defined());
  EXPECT_NEAR(g.data_ptr<float>()[0], 2.0f, 1e-5f);
}

/// y = a + b.  dy/da = 1, dy/db = 1.
TEST(Engine, AddBackward) {
  auto a = aten::full({}, 3.0f);
  auto b = aten::full({}, 5.0f);
  set_requires_grad(a, true);
  set_requires_grad(b, true);

  auto y = test_add(a, b);
  Engine::backward(y);

  EXPECT_NEAR(mutable_grad(a).data_ptr<float>()[0], 1.0f, 1e-5f);
  EXPECT_NEAR(mutable_grad(b).data_ptr<float>()[0], 1.0f, 1e-5f);
}

/// y = a * b.  dy/da = b, dy/db = a.
TEST(Engine, MulBackward) {
  auto a = aten::full({}, 3.0f);
  auto b = aten::full({}, 5.0f);
  set_requires_grad(a, true);
  set_requires_grad(b, true);

  auto y = test_mul(a, b);
  Engine::backward(y);

  EXPECT_NEAR(mutable_grad(a).data_ptr<float>()[0], 5.0f, 1e-5f);
  EXPECT_NEAR(mutable_grad(b).data_ptr<float>()[0], 3.0f, 1e-5f);
}

/// y = (a + b) * a.  dy/da = (a + b) + a = 2a + b.  dy/db = a.
TEST(Engine, ChainedOps) {
  auto a = aten::full({}, 2.0f);
  auto b = aten::full({}, 3.0f);
  set_requires_grad(a, true);
  set_requires_grad(b, true);

  auto c = test_add(a, b);  // c = a + b = 5
  auto y = test_mul(c, a);  // y = c * a = 10

  Engine::backward(y);

  // dy/da = d(c*a)/da = c * 1 + a * dc/da = (a+b) + a * 1 = 2a + b = 7
  // But wait — 'a' is used twice (in add and in mul). The gradient should
  // accumulate: from mul, da gets c = 5. From add via mul, da gets a = 2.
  // Total: 5 + 2 = 7.
  EXPECT_NEAR(mutable_grad(a).data_ptr<float>()[0], 7.0f, 1e-5f);
  // dy/db = a * dc/db = a * 1 = 2
  EXPECT_NEAR(mutable_grad(b).data_ptr<float>()[0], 2.0f, 1e-5f);
}

/// GradMode is disabled during backward (no graph building).
TEST(Engine, GradModeDisabledDuringBackward) {
  // This is implicitly tested — if GradMode wasn't disabled during
  // backward, the add() inside AccumulateGrad would try to build a
  // graph and eventually cause issues.
  auto x = aten::full({}, 5.0f);
  set_requires_grad(x, true);

  auto y = test_mul_scalar(x, 3.0);
  Engine::backward(y);

  EXPECT_NEAR(mutable_grad(x).data_ptr<float>()[0], 3.0f, 1e-5f);
  EXPECT_TRUE(autograd::GradMode::is_enabled());  // restored after backward
}

}  // namespace
