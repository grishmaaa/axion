#pragma once

// ============================================================================
// Axion / nn — LayerNorm
// ============================================================================
//
// Layer normalization (Ba et al., 2016):
//   y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
//
// Normalizes over the last dimension (features).
// Learnable parameters: gamma (scale), beta (shift).
//
// Input:  (batch, features)  or  (batch, seq, features)
// Output: same shape as input

#include <cmath>
#include <cassert>
#include <cstring>

#include "axion/nn/Module.h"
#include "axion/nn/init.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace nn {

namespace detail {

/// Backward for LayerNorm.
/// Uses the saved normalized output x_hat and computed stats.
struct LayerNormBackward : autograd::Node {
  c10::Tensor input_;
  c10::Tensor gamma_;
  int64_t normalized_size_;
  float eps_;

  LayerNormBackward(c10::Tensor input, c10::Tensor gamma,
                    int64_t norm_size, float eps)
      : input_(std::move(input)),
        gamma_(std::move(gamma)),
        normalized_size_(norm_size),
        eps_(eps) {}

  std::vector<c10::Tensor> apply(
      std::vector<c10::Tensor> grads) override {
    auto& grad = grads[0];

    int64_t total = input_.numel();
    int64_t N = normalized_size_;
    int64_t outer = total / N;

    auto d_input = c10::Tensor::empty(input_.sizes(), input_.dtype());
    auto d_gamma = aten::zeros({N}, input_.dtype());
    auto d_beta = aten::zeros({N}, input_.dtype());

    const float* xp = input_.data_ptr<float>();
    const float* gp = grad.data_ptr<float>();
    const float* gamma_p = gamma_.data_ptr<float>();
    float* dxp = d_input.data_ptr<float>();
    float* dgp = d_gamma.data_ptr<float>();
    float* dbp = d_beta.data_ptr<float>();

    for (int64_t i = 0; i < outer; ++i) {
      const float* xi = xp + i * N;
      const float* gi = gp + i * N;
      float* dxi = dxp + i * N;

      // Recompute mean and variance for this row
      float mean = 0.0f;
      for (int64_t j = 0; j < N; ++j) mean += xi[j];
      mean /= static_cast<float>(N);

      float var = 0.0f;
      for (int64_t j = 0; j < N; ++j) {
        float d = xi[j] - mean;
        var += d * d;
      }
      var /= static_cast<float>(N);

      float inv_std = 1.0f / std::sqrt(var + eps_);
      float inv_N = 1.0f / static_cast<float>(N);

      // x_hat = (x - mean) * inv_std
      // Gradient flow:
      //   d_x = inv_std * (gamma * grad - mean(gamma*grad) -
      //          x_hat * mean(x_hat * gamma * grad))
      float sum_gg = 0.0f;  // sum of gamma * grad
      float sum_xhat_gg = 0.0f;
      for (int64_t j = 0; j < N; ++j) {
        float x_hat = (xi[j] - mean) * inv_std;
        float gg = gamma_p[j] * gi[j];
        sum_gg += gg;
        sum_xhat_gg += x_hat * gg;
      }

      for (int64_t j = 0; j < N; ++j) {
        float x_hat = (xi[j] - mean) * inv_std;
        float gg = gamma_p[j] * gi[j];
        dxi[j] = inv_std * (gg - inv_N * sum_gg -
                             inv_N * x_hat * sum_xhat_gg);
      }

      // d_gamma and d_beta accumulate across rows
      for (int64_t j = 0; j < N; ++j) {
        float x_hat = (xi[j] - mean) * inv_std;
        dgp[j] += gi[j] * x_hat;
        dbp[j] += gi[j];
      }
    }

    return {d_input, d_gamma, d_beta};
  }
};

}  // namespace detail

class LayerNorm : public Module {
 public:
  explicit LayerNorm(int64_t normalized_shape, float eps = 1e-5f)
      : normalized_shape_(normalized_shape), eps_(eps) {
    // gamma (scale) initialized to 1
    auto g = aten::ones({normalized_shape}, c10::ScalarType::Float32);
    register_parameter("weight", Parameter(std::move(g)));

    // beta (bias) initialized to 0
    auto b = aten::zeros({normalized_shape}, c10::ScalarType::Float32);
    register_parameter("bias", Parameter(std::move(b)));
  }

  c10::Tensor forward(const c10::Tensor& input) override {
    assert(input.defined() && "LayerNorm: input undefined");

    // Normalize over the last dimension
    int64_t total = input.numel();
    int64_t N = normalized_shape_;
    int64_t outer = total / N;
    assert(total % N == 0 && "LayerNorm: input size not divisible");

    auto output = c10::Tensor::empty(input.sizes(), input.dtype());

    const float* xp = input.data_ptr<float>();
    const float* gp = gamma().data().data_ptr<float>();
    const float* bp = beta().data().data_ptr<float>();
    float* op = output.data_ptr<float>();

    for (int64_t i = 0; i < outer; ++i) {
      const float* xi = xp + i * N;
      float* oi = op + i * N;

      // Compute mean
      float mean = 0.0f;
      for (int64_t j = 0; j < N; ++j) mean += xi[j];
      mean /= static_cast<float>(N);

      // Compute variance
      float var = 0.0f;
      for (int64_t j = 0; j < N; ++j) {
        float d = xi[j] - mean;
        var += d * d;
      }
      var /= static_cast<float>(N);

      // Normalize and apply affine
      float inv_std = 1.0f / std::sqrt(var + eps_);
      for (int64_t j = 0; j < N; ++j) {
        oi[j] = (xi[j] - mean) * inv_std * gp[j] + bp[j];
      }
    }

    // Record on autograd graph
    if (autograd::GradMode::is_enabled() &&
        (input.requires_grad() ||
         gamma().data().requires_grad())) {
      auto node = std::make_shared<detail::LayerNormBackward>(
          input, gamma().data(), N, eps_);
      node->add_next_edge(autograd::gradient_edge(input));
      node->add_next_edge(autograd::gradient_edge(gamma().data()));
      node->add_next_edge(autograd::gradient_edge(beta().data()));
      autograd::set_grad_fn(output, node);
    }

    return output;
  }

  int64_t normalized_shape() const { return normalized_shape_; }

 private:
  int64_t normalized_shape_;
  float eps_;

  Parameter& gamma() { return params_[0].second; }
  Parameter& beta()  { return params_[1].second; }
};

}  // namespace nn
