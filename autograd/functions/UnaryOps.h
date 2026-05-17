#pragma once

// ============================================================================
// Axion / autograd / functions — Unary ops backward nodes
// ============================================================================

#include <cmath>
#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace autograd {

struct ReluBackward : Node {
  c10::Tensor input_;
  explicit ReluBackward(c10::Tensor in) : input_(std::move(in)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // da = grad * (input > 0)
    auto mask = aten::relu(input_);  // reuse relu: 0 where <=0
    // Need sign-like: 1 where input>0, 0 elsewhere
    // relu(x)/x doesn't work for x=0. Use: relu(input) > 0 ? 1 : 0
    auto out = c10::Tensor::empty(
        std::vector<int64_t>(input_.sizes().begin(), input_.sizes().end()),
        input_.dtype());
    int64_t n = input_.numel();
    const float* in = input_.data_ptr<float>();
    const float* gr = g[0].data_ptr<float>();
    float* o = out.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) o[i] = in[i] > 0.0f ? gr[i] : 0.0f;
    return {out};
  }
};

struct AbsBackward : Node {
  c10::Tensor input_;
  explicit AbsBackward(c10::Tensor in) : input_(std::move(in)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // da = grad * sign(input)
    auto out = c10::Tensor::empty(
        std::vector<int64_t>(input_.sizes().begin(), input_.sizes().end()),
        input_.dtype());
    int64_t n = input_.numel();
    const float* in = input_.data_ptr<float>();
    const float* gr = g[0].data_ptr<float>();
    float* o = out.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i)
      o[i] = in[i] > 0.0f ? gr[i] : (in[i] < 0.0f ? -gr[i] : 0.0f);
    return {out};
  }
};

struct ExpBackward : Node {
  c10::Tensor result_;
  explicit ExpBackward(c10::Tensor res) : result_(std::move(res)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    return {aten::mul(g[0], result_)};
  }
};

struct LogBackward : Node {
  c10::Tensor input_;
  explicit LogBackward(c10::Tensor in) : input_(std::move(in)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    return {aten::div(g[0], input_)};
  }
};

struct SqrtBackward : Node {
  c10::Tensor result_;
  explicit SqrtBackward(c10::Tensor res) : result_(std::move(res)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // da = grad / (2 * sqrt(a)) = grad / (2 * result)
    return {aten::div(g[0], aten::mul_scalar(result_, 2.0))};
  }
};

struct TanhBackward : Node {
  c10::Tensor result_;
  explicit TanhBackward(c10::Tensor res) : result_(std::move(res)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // da = grad * (1 - tanh(a)^2)
    auto one_minus_sq = aten::sub_scalar(
        aten::mul(result_, result_), -1.0);  // 1 - r^2 via -(r^2 - 1)
    // Actually: 1 - r^2. Let me use full(1) - r*r.
    auto sq = aten::mul(result_, result_);
    auto sizes = std::vector<int64_t>(
        result_.sizes().begin(), result_.sizes().end());
    auto one = aten::full(sizes, 1.0, result_.dtype());
    auto factor = aten::sub(one, sq);
    return {aten::mul(g[0], factor)};
  }
};

struct SigmoidBackward : Node {
  c10::Tensor result_;
  explicit SigmoidBackward(c10::Tensor res) : result_(std::move(res)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // da = grad * sigmoid(a) * (1 - sigmoid(a))
    auto sizes = std::vector<int64_t>(
        result_.sizes().begin(), result_.sizes().end());
    auto one = aten::full(sizes, 1.0, result_.dtype());
    auto factor = aten::mul(result_, aten::sub(one, result_));
    return {aten::mul(g[0], factor)};
  }
};

struct GeluBackward : Node {
  c10::Tensor input_;
  explicit GeluBackward(c10::Tensor in) : input_(std::move(in)) {}
  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // d/dx gelu(x) = 0.5*(1+erf(x/sqrt(2))) + x*exp(-x^2/2)/sqrt(2*pi)
    int64_t n = input_.numel();
    auto out = c10::Tensor::empty(
        std::vector<int64_t>(input_.sizes().begin(), input_.sizes().end()),
        input_.dtype());
    const float* x = input_.data_ptr<float>();
    const float* gr = g[0].data_ptr<float>();
    float* o = out.data_ptr<float>();
    constexpr float sqrt2 = 1.4142135623730951f;
    constexpr float sqrt2pi = 2.5066282746310002f;
    for (int64_t i = 0; i < n; ++i) {
      float cdf = 0.5f * (1.0f + std::erf(x[i] / sqrt2));
      float pdf = std::exp(-0.5f * x[i] * x[i]) / sqrt2pi;
      o[i] = gr[i] * (cdf + x[i] * pdf);
    }
    return {out};
  }
};

}  // namespace autograd
