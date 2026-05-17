#pragma once

// ============================================================================
// Axion / autograd / functions — Basic arithmetic backward nodes
// ============================================================================

#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace autograd {

struct AddBackward : Node {
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    return {g[0], g[0]};
  }
};

struct SubBackward : Node {
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    return {g[0], aten::neg(g[0])};
  }
};

struct MulBackward : Node {
  c10::Tensor a_, b_;
  MulBackward(c10::Tensor a, c10::Tensor b)
      : a_(std::move(a)), b_(std::move(b)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    return {aten::mul(g[0], b_), aten::mul(g[0], a_)};
  }
};

struct DivBackward : Node {
  c10::Tensor a_, b_;
  DivBackward(c10::Tensor a, c10::Tensor b)
      : a_(std::move(a)), b_(std::move(b)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // da = grad / b
    auto da = aten::div(g[0], b_);
    // db = -grad * a / b^2
    auto db = aten::neg(aten::div(aten::mul(g[0], a_), aten::mul(b_, b_)));
    return {da, db};
  }
};

struct NegBackward : Node {
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    return {aten::neg(g[0])};
  }
};

struct PowBackward : Node {
  c10::Tensor a_, b_;
  PowBackward(c10::Tensor a, c10::Tensor b)
      : a_(std::move(a)), b_(std::move(b)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // da = grad * b * a^(b-1)
    auto da = aten::mul(g[0], aten::mul(b_, aten::pow(a_,
        aten::sub_scalar(b_, 1.0))));
    // db = grad * a^b * ln(a)
    auto db = aten::mul(g[0], aten::mul(aten::pow(a_, b_), aten::log(a_)));
    return {da, db};
  }
};

// Scalar ops backward
struct MulScalarBackward : Node {
  double scalar_;
  explicit MulScalarBackward(double s) : scalar_(s) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    return {aten::mul_scalar(g[0], scalar_)};
  }
};

struct DivScalarBackward : Node {
  double scalar_;
  explicit DivScalarBackward(double s) : scalar_(s) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    return {aten::div_scalar(g[0], scalar_)};
  }
};

// add_scalar and sub_scalar backward is identity (da = grad)
struct PassthroughBackward : Node {
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    return {g[0]};
  }
};

}  // namespace autograd
