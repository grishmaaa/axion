#pragma once

// ============================================================================
// Axion / nn — Softmax (functional, with autograd)
// ============================================================================
//
// Row-wise softmax: softmax(x)[i,j] = exp(x[i,j]) / sum_k exp(x[i,k])
//
// Uses the log-sum-exp trick for numerical stability.
//
// Backward:
//   d_x[i,j] = y[i,j] * (grad[i,j] - sum_k(grad[i,k] * y[i,k]))

#include <cmath>
#include <algorithm>
#include <cassert>

#include "c10/core/Tensor.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace nn {
namespace functional {

namespace detail {

struct SoftmaxBackward : autograd::Node {
  c10::Tensor output_;  // save softmax output for efficient backward

  explicit SoftmaxBackward(c10::Tensor output)
      : output_(std::move(output)) {}

  std::vector<c10::Tensor> apply(
      std::vector<c10::Tensor> grads) override {
    auto& grad = grads[0];

    int64_t rows = output_.size(0);
    int64_t cols = output_.size(1);

    auto d_input = c10::Tensor::empty({rows, cols}, output_.dtype());

    const float* yp = output_.data_ptr<float>();
    const float* gp = grad.data_ptr<float>();
    float* dp = d_input.data_ptr<float>();

    for (int64_t i = 0; i < rows; ++i) {
      // dot = sum_k(grad[i,k] * y[i,k])
      float dot = 0.0f;
      for (int64_t j = 0; j < cols; ++j) {
        dot += gp[i * cols + j] * yp[i * cols + j];
      }
      // d_x[i,j] = y[i,j] * (grad[i,j] - dot)
      for (int64_t j = 0; j < cols; ++j) {
        dp[i * cols + j] = yp[i * cols + j] * (gp[i * cols + j] - dot);
      }
    }

    return {d_input};
  }
};

}  // namespace detail

/// Row-wise softmax with autograd support.
/// Input: (rows, cols), Output: (rows, cols)
inline c10::Tensor softmax(const c10::Tensor& input) {
  assert(input.defined() && "softmax: input undefined");
  assert(input.ndim() == 2 && "softmax: input must be 2D");

  int64_t rows = input.size(0);
  int64_t cols = input.size(1);

  auto output = c10::Tensor::empty({rows, cols}, input.dtype());

  const float* ip = input.data_ptr<float>();
  float* op = output.data_ptr<float>();

  for (int64_t i = 0; i < rows; ++i) {
    // Find max for numerical stability
    float max_val = ip[i * cols];
    for (int64_t j = 1; j < cols; ++j) {
      max_val = std::max(max_val, ip[i * cols + j]);
    }

    // exp and sum
    float sum = 0.0f;
    for (int64_t j = 0; j < cols; ++j) {
      op[i * cols + j] = std::exp(ip[i * cols + j] - max_val);
      sum += op[i * cols + j];
    }

    // normalize
    for (int64_t j = 0; j < cols; ++j) {
      op[i * cols + j] /= sum;
    }
  }

  // Record on autograd graph
  if (autograd::GradMode::is_enabled() && input.requires_grad()) {
    auto node = std::make_shared<detail::SoftmaxBackward>(output);
    node->add_next_edge(autograd::gradient_edge(input));
    autograd::set_grad_fn(output, node);
  }

  return output;
}

}  // namespace functional
}  // namespace nn
