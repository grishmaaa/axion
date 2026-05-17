#pragma once

// ============================================================================
// Axion / nn — Parameter
// ============================================================================
//
// A Parameter wraps a Tensor with requires_grad=true by default.
// Separated from Tensor so optimizers can traverse only learnable
// weights, and Modules can register them by name.
//
// Design: NOT a Tensor subclass (avoids diamond inheritance).
// Instead, provides implicit conversion for use in autograd ops.

#include "c10/core/Tensor.h"
#include "autograd/AutogradMeta.h"

namespace nn {

class Parameter {
 public:
  /// Construct from existing tensor data. Sets requires_grad=true.
  explicit Parameter(c10::Tensor data) : data_(std::move(data)) {
    autograd::set_requires_grad(data_, true);
  }

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------

  c10::Tensor& data() { return data_; }
  const c10::Tensor& data() const { return data_; }

  /// Get the gradient (may be undefined if backward hasn't been called).
  c10::Tensor& grad() { return autograd::mutable_grad(data_); }

  /// Check if gradient is defined.
  bool has_grad() const {
    auto* meta = autograd::get_autograd_meta(data_);
    return meta && meta->grad_.defined();
  }

  /// Zero the gradient.
  void zero_grad() {
    auto* meta = autograd::get_autograd_meta(data_);
    if (meta) meta->grad_ = c10::Tensor();
  }

  // ------------------------------------------------------------------
  // Implicit conversion to const Tensor& for use in autograd ops
  // ------------------------------------------------------------------

  operator const c10::Tensor&() const { return data_; }

  // Convenience accessors
  int64_t numel() const { return data_.numel(); }
  c10::ScalarType dtype() const { return data_.dtype(); }
  auto sizes() const { return data_.sizes(); }

 private:
  c10::Tensor data_;
};

}  // namespace nn
