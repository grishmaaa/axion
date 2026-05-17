// ============================================================================
// Tests for autograd metadata
// ============================================================================

#include <gtest/gtest.h>

#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "aten/ops/Ops.h"

namespace {

TEST(AutogradMeta, DefaultTensorHasNoMeta) {
  auto t = aten::ones({3});
  EXPECT_EQ(t.autograd_meta(), nullptr);
  EXPECT_FALSE(t.requires_grad());
}

TEST(AutogradMeta, SetRequiresGrad) {
  auto t = aten::ones({3});
  autograd::set_requires_grad(t, true);
  EXPECT_TRUE(t.requires_grad());
  EXPECT_NE(t.autograd_meta(), nullptr);
}

TEST(AutogradMeta, GradInitiallyUndefined) {
  auto t = aten::ones({3});
  autograd::set_requires_grad(t, true);
  auto& g = autograd::mutable_grad(t);
  EXPECT_FALSE(g.defined());
}

TEST(AutogradMeta, SetAndGetGrad) {
  auto t = aten::ones({3});
  autograd::set_requires_grad(t, true);
  autograd::mutable_grad(t) = aten::full({3}, 2.0f);

  auto& g = autograd::mutable_grad(t);
  EXPECT_TRUE(g.defined());
  EXPECT_EQ(g.numel(), 3);
  EXPECT_NEAR(g.data_ptr<float>()[0], 2.0f, 1e-5f);
}

TEST(AutogradMeta, GradFnInitiallyNull) {
  auto t = aten::ones({3});
  autograd::set_requires_grad(t, true);
  EXPECT_EQ(autograd::grad_fn(t), nullptr);
}

TEST(AutogradMeta, GradientEdgeForLeaf) {
  auto t = aten::ones({3});
  autograd::set_requires_grad(t, true);

  auto edge = autograd::gradient_edge(t);
  EXPECT_TRUE(static_cast<bool>(edge));
  EXPECT_EQ(edge.input_nr, 0u);
  // Should be an AccumulateGrad node
  EXPECT_NE(edge.function, nullptr);
}

TEST(AutogradMeta, GradientEdgeForNoGrad) {
  auto t = aten::ones({3});
  auto edge = autograd::gradient_edge(t);
  EXPECT_FALSE(static_cast<bool>(edge));
}

// ============================================================================
// GradMode tests
// ============================================================================

TEST(GradMode, DefaultEnabled) {
  EXPECT_TRUE(autograd::GradMode::is_enabled());
}

TEST(GradMode, NoGradGuard) {
  EXPECT_TRUE(autograd::GradMode::is_enabled());
  {
    autograd::NoGradGuard guard;
    EXPECT_FALSE(autograd::GradMode::is_enabled());
  }
  EXPECT_TRUE(autograd::GradMode::is_enabled());
}

TEST(GradMode, NestedNoGradGuard) {
  EXPECT_TRUE(autograd::GradMode::is_enabled());
  {
    autograd::NoGradGuard outer;
    EXPECT_FALSE(autograd::GradMode::is_enabled());
    {
      autograd::NoGradGuard inner;
      EXPECT_FALSE(autograd::GradMode::is_enabled());
    }
    EXPECT_FALSE(autograd::GradMode::is_enabled());
  }
  EXPECT_TRUE(autograd::GradMode::is_enabled());
}

}  // namespace
