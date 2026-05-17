#pragma once

// ============================================================================
// Axion / optim — Adam optimizer
// ============================================================================
//
// Adam: Adaptive Moment Estimation (Kingma & Ba, 2015)
//
//   m_t = β1 * m_{t-1} + (1 - β1) * g
//   v_t = β2 * v_{t-1} + (1 - β2) * g²
//   m̂_t = m_t / (1 - β1^t)         (bias correction)
//   v̂_t = v_t / (1 - β2^t)
//   θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
//

#include <cmath>
#include <vector>
#include "axion/nn/Parameter.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "aten/ops/Ops.h"

namespace optim {

class Adam {
 public:
  Adam(std::vector<nn::Parameter*> params,
       double lr = 1e-3,
       double beta1 = 0.9,
       double beta2 = 0.999,
       double eps = 1e-8,
       double weight_decay = 0.0)
      : params_(std::move(params)),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        weight_decay_(weight_decay),
        step_count_(0) {
    // Initialize moment buffers
    m_.resize(params_.size());   // first moment  (mean)
    v_.resize(params_.size());   // second moment (variance)
  }

  /// Zero all parameter gradients.
  void zero_grad() {
    for (auto* p : params_) {
      p->zero_grad();
    }
  }

  /// Apply one step of Adam.
  void step() {
    autograd::NoGradGuard no_grad;

    step_count_++;

    for (size_t i = 0; i < params_.size(); ++i) {
      auto& param = params_[i];
      if (!param->has_grad()) continue;

      auto& data = param->data();
      auto grad = param->grad();

      // Weight decay (decoupled, AdamW style)
      if (weight_decay_ > 0.0) {
        data = aten::sub(data, aten::mul_scalar(data, lr_ * weight_decay_));
      }

      // Initialize moments lazily on first use
      if (!m_[i].defined()) {
        m_[i] = aten::zeros(data.sizes(), data.dtype());
        v_[i] = aten::zeros(data.sizes(), data.dtype());
      }

      // m_t = β1 * m_{t-1} + (1 - β1) * g
      m_[i] = aten::add(
          aten::mul_scalar(m_[i], beta1_),
          aten::mul_scalar(grad, 1.0 - beta1_));

      // v_t = β2 * v_{t-1} + (1 - β2) * g²
      v_[i] = aten::add(
          aten::mul_scalar(v_[i], beta2_),
          aten::mul_scalar(aten::mul(grad, grad), 1.0 - beta2_));

      // Bias correction
      double bc1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
      double bc2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

      auto m_hat = aten::mul_scalar(m_[i], 1.0 / bc1);
      auto v_hat = aten::mul_scalar(v_[i], 1.0 / bc2);

      // θ = θ - lr * m̂ / (√v̂ + ε)
      auto v_sqrt = aten::sqrt(v_hat);
      auto denom = aten::add_scalar(v_sqrt, eps_);
      auto update = aten::div(m_hat, denom);

      data = aten::sub(data, aten::mul_scalar(update, lr_));

      // Restore requires_grad on the updated tensor
      autograd::set_requires_grad(data, true);
    }
  }

 private:
  std::vector<nn::Parameter*> params_;
  double lr_;
  double beta1_;
  double beta2_;
  double eps_;
  double weight_decay_;
  int64_t step_count_;

  std::vector<c10::Tensor> m_;   // first moment estimates
  std::vector<c10::Tensor> v_;   // second moment estimates
};

}  // namespace optim
