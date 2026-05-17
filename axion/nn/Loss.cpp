// ============================================================================
// Axion / nn — Loss function implementations
// ============================================================================

#include "axion/nn/Loss.h"

#include <cmath>
#include <cassert>
#include <algorithm>

#include "autograd/Variable.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace nn {

// ============================================================================
// MSE Loss
// ============================================================================

c10::Tensor mse_loss(const c10::Tensor& pred, const c10::Tensor& target) {
  auto diff = autograd::sub(pred, target);
  auto sq = autograd::mul(diff, diff);
  return autograd::mean(sq);
}

// ============================================================================
// Cross-Entropy Loss
// ============================================================================
//
// Numerically stable cross-entropy for classification.
//
// Input:
//   logits: (batch, num_classes) — raw scores (unnormalized)
//   targets: (batch,) — integer class indices stored as float
//
// Formula:
//   loss = -mean_i [ logits[i, t_i] - log(sum_j exp(logits[i, j])) ]
//
// Gradient w.r.t. logits:
//   d_logits[i, j] = (softmax(logits)[i, j] - 1{j == t_i}) / batch_size
//

namespace {

struct CrossEntropyBackward : autograd::Node {
  c10::Tensor logits_;
  c10::Tensor targets_;

  CrossEntropyBackward(c10::Tensor logits, c10::Tensor targets)
      : logits_(std::move(logits)), targets_(std::move(targets)) {}

  std::vector<c10::Tensor> apply(
      std::vector<c10::Tensor> grads) override {
    auto& grad = grads[0];
    float grad_scale = grad.data_ptr<float>()[0];

    int64_t batch = logits_.size(0);
    int64_t classes = logits_.size(1);

    auto d_logits = c10::Tensor::empty(
        {batch, classes}, logits_.dtype());

    const float* lp = logits_.data_ptr<float>();
    const float* tp = targets_.data_ptr<float>();
    float* dp = d_logits.data_ptr<float>();

    for (int64_t i = 0; i < batch; ++i) {
      // Compute softmax for row i (numerically stable)
      float max_val = lp[i * classes];
      for (int64_t j = 1; j < classes; ++j) {
        max_val = std::max(max_val, lp[i * classes + j]);
      }

      float sum_exp = 0.0f;
      for (int64_t j = 0; j < classes; ++j) {
        sum_exp += std::exp(lp[i * classes + j] - max_val);
      }

      int64_t target_idx = static_cast<int64_t>(tp[i]);

      for (int64_t j = 0; j < classes; ++j) {
        float softmax_j = std::exp(lp[i * classes + j] - max_val) / sum_exp;
        float indicator = (j == target_idx) ? 1.0f : 0.0f;
        dp[i * classes + j] = grad_scale * (softmax_j - indicator)
                               / static_cast<float>(batch);
      }
    }

    return {d_logits};
  }
};

}  // namespace

c10::Tensor cross_entropy_loss(
    const c10::Tensor& logits,
    const c10::Tensor& targets) {
  assert(logits.defined() && "cross_entropy_loss: logits undefined");
  assert(targets.defined() && "cross_entropy_loss: targets undefined");
  assert(logits.ndim() == 2 && "cross_entropy_loss: logits must be 2D");
  assert(targets.ndim() == 1 && "cross_entropy_loss: targets must be 1D");
  assert(logits.size(0) == targets.size(0) &&
         "cross_entropy_loss: batch size mismatch");

  int64_t batch = logits.size(0);
  int64_t classes = logits.size(1);

  const float* lp = logits.data_ptr<float>();
  const float* tp = targets.data_ptr<float>();

  // Compute loss = -mean_i [ logit_{i,t_i} - log(sum_j exp(logit_{i,j})) ]
  float total_loss = 0.0f;
  for (int64_t i = 0; i < batch; ++i) {
    int64_t target_idx = static_cast<int64_t>(tp[i]);
    assert(target_idx >= 0 && target_idx < classes &&
           "cross_entropy_loss: target index out of range");

    // log-sum-exp (numerically stable)
    float max_val = lp[i * classes];
    for (int64_t j = 1; j < classes; ++j) {
      max_val = std::max(max_val, lp[i * classes + j]);
    }

    float sum_exp = 0.0f;
    for (int64_t j = 0; j < classes; ++j) {
      sum_exp += std::exp(lp[i * classes + j] - max_val);
    }
    float log_sum_exp = max_val + std::log(sum_exp);

    total_loss += -(lp[i * classes + target_idx] - log_sum_exp);
  }
  total_loss /= static_cast<float>(batch);

  // Create scalar output
  auto loss = aten::full({}, total_loss, logits.dtype());

  // Record on autograd graph
  if (autograd::GradMode::is_enabled() && logits.requires_grad()) {
    auto node = std::make_shared<CrossEntropyBackward>(logits, targets);
    node->add_next_edge(autograd::gradient_edge(logits));
    autograd::set_grad_fn(loss, node);
  }

  return loss;
}

}  // namespace nn
