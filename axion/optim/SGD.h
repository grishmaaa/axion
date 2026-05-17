#pragma once

// ============================================================================
// Axion / optim — Optimizer base + SGD
// ============================================================================

#include <vector>
#include "axion/nn/Parameter.h"
#include "autograd/GradMode.h"
#include "aten/ops/Ops.h"

namespace optim {

/// Base class for all optimizers.
class Optimizer {
 public:
  explicit Optimizer(std::vector<nn::Parameter*> params, double lr)
      : params_(std::move(params)), lr_(lr) {}

  virtual ~Optimizer() = default;

  /// Apply parameter updates. Subclasses implement this.
  virtual void step() = 0;

  /// Zero all parameter gradients.
  void zero_grad() {
    for (auto* p : params_) {
      p->zero_grad();
    }
  }

 protected:
  std::vector<nn::Parameter*> params_;
  double lr_;
};

/// Stochastic Gradient Descent with optional momentum and weight decay.
class SGD : public Optimizer {
 public:
  SGD(std::vector<nn::Parameter*> params,
      double lr,
      double momentum = 0.0,
      double weight_decay = 0.0)
      : Optimizer(std::move(params), lr),
        momentum_(momentum),
        weight_decay_(weight_decay) {
    if (momentum_ > 0.0) {
      velocity_.resize(params_.size());
    }
  }

  void step() override {
    autograd::NoGradGuard no_grad;

    for (size_t i = 0; i < params_.size(); ++i) {
      auto& param = params_[i];
      if (!param->has_grad()) continue;

      auto grad = param->grad();

      // Weight decay: grad += weight_decay * param
      if (weight_decay_ > 0.0) {
        grad = aten::add(grad, aten::mul_scalar(param->data(), weight_decay_));
      }

      if (momentum_ > 0.0) {
        if (velocity_[i].defined()) {
          // v = momentum * v + grad
          velocity_[i] = aten::add(
              aten::mul_scalar(velocity_[i], momentum_), grad);
        } else {
          velocity_[i] = grad;
        }
        // param -= lr * v
        param->data() = aten::sub(
            param->data(), aten::mul_scalar(velocity_[i], lr_));
      } else {
        // param -= lr * grad
        param->data() = aten::sub(
            param->data(), aten::mul_scalar(grad, lr_));
      }

      // CRITICAL: re-set requires_grad on the new tensor.
      // The sub() above created a fresh tensor without autograd metadata
      // (NoGradGuard is active). We must restore requires_grad so the
      // next forward pass records onto the graph.
      autograd::set_requires_grad(param->data(), true);
    }
  }

 private:
  double momentum_;
  double weight_decay_;
  std::vector<c10::Tensor> velocity_;
};

}  // namespace optim
