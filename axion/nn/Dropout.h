#pragma once

// ============================================================================
// Axion / nn — Dropout
// ============================================================================
//
// During training: randomly zeros elements with probability p.
// During eval: identity (no-op).
//
// Remaining elements are scaled by 1/(1-p) to maintain expected values
// (inverted dropout).

#include <random>
#include "axion/nn/Module.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace nn {

namespace detail {

/// Backward for Dropout: just multiply by the same mask.
struct DropoutBackward : autograd::Node {
  c10::Tensor mask_;  // (same shape as input) — 0 or 1/(1-p)

  explicit DropoutBackward(c10::Tensor mask) : mask_(std::move(mask)) {}

  std::vector<c10::Tensor> apply(
      std::vector<c10::Tensor> grads) override {
    return {aten::mul(grads[0], mask_)};
  }
};

}  // namespace detail

class Dropout : public Module {
 public:
  explicit Dropout(float p = 0.5f) : p_(p) {
    assert(p >= 0.0f && p < 1.0f && "Dropout probability must be in [0, 1)");
  }

  c10::Tensor forward(const c10::Tensor& input) override {
    if (!is_training() || p_ == 0.0f) {
      return input;  // eval mode or p=0: identity
    }

    // Generate mask
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0 - static_cast<double>(p_));

    auto mask = c10::Tensor::empty(input.sizes(), input.dtype());
    float scale = 1.0f / (1.0f - p_);
    float* mp = mask.data_ptr<float>();
    int64_t n = input.numel();
    for (int64_t i = 0; i < n; ++i) {
      mp[i] = dist(gen) ? scale : 0.0f;
    }

    auto output = aten::mul(input, mask);

    // Record on autograd graph
    if (autograd::GradMode::is_enabled() && input.requires_grad()) {
      auto node = std::make_shared<detail::DropoutBackward>(mask);
      node->add_next_edge(autograd::gradient_edge(input));
      autograd::set_grad_fn(output, node);
    }

    return output;
  }

 private:
  float p_;
};

}  // namespace nn
