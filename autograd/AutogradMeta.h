#pragma once

// ============================================================================
// Axion / autograd — AutogradMeta (concrete)
// ============================================================================
//
// The concrete metadata stored on tensors that participate in autograd.
// Contains the gradient tensor, the grad_fn (Node that created this
// tensor), and the gradient accumulator for leaf tensors.

#include <memory>

#include "c10/core/AutogradMetaInterface.h"
#include "c10/core/Tensor.h"
#include "autograd/Node.h"
#include "autograd/Edge.h"

namespace autograd {

struct AutogradMeta : c10::AutogradMetaInterface {
  /// The accumulated gradient. Undefined until backward() is called.
  c10::Tensor grad_;

  /// The Node that produced this tensor (null for leaf tensors).
  std::shared_ptr<Node> grad_fn_;

  /// Which output of grad_fn_ this tensor corresponds to.
  uint32_t output_nr_ = 0;

  /// Weak reference to the AccumulateGrad node for leaf tensors.
  /// Created lazily on first backward pass.
  std::weak_ptr<Node> grad_accumulator_;
};

// ============================================================================
// Helper functions (work on any Tensor, downcast internally)
// ============================================================================

/// Get or create AutogradMeta on a tensor.
inline AutogradMeta* get_autograd_meta(const c10::Tensor& t) {
  return static_cast<AutogradMeta*>(t.autograd_meta());
}

/// Ensure a tensor has AutogradMeta. Creates it if missing.
inline AutogradMeta& materialize_autograd_meta(c10::Tensor& t) {
  if (!t.autograd_meta()) {
    t.unsafeGetTensorImplRaw()->set_autograd_meta(
        std::make_unique<AutogradMeta>());
  }
  return *static_cast<AutogradMeta*>(t.autograd_meta());
}

/// Set requires_grad on a tensor.
inline void set_requires_grad(c10::Tensor& t, bool requires_grad) {
  auto& meta = materialize_autograd_meta(t);
  meta.requires_grad_ = requires_grad;
}

/// Get the gradient of a tensor (mutable reference).
inline c10::Tensor& mutable_grad(const c10::Tensor& t) {
  auto* meta = get_autograd_meta(t);
  assert(meta && "mutable_grad: tensor has no autograd metadata");
  return meta->grad_;
}

/// Get the gradient of a tensor (const access).
inline const c10::Tensor& get_grad(const c10::Tensor& t) {
  auto* meta = get_autograd_meta(t);
  assert(meta && "get_grad: tensor has no autograd metadata");
  return meta->grad_;
}

/// Get the grad_fn of a tensor (the Node that created it).
inline std::shared_ptr<Node> grad_fn(const c10::Tensor& t) {
  auto* meta = get_autograd_meta(t);
  if (!meta) return nullptr;
  return meta->grad_fn_;
}

/// Set the grad_fn on a tensor.
inline void set_grad_fn(
    c10::Tensor& t, std::shared_ptr<Node> fn, uint32_t output_nr = 0) {
  auto& meta = materialize_autograd_meta(t);
  meta.grad_fn_ = std::move(fn);
  meta.output_nr_ = output_nr;
  meta.requires_grad_ = true;
}

/// Get the Edge for a tensor's gradient flow.
/// For non-leaf tensors: Edge(grad_fn, output_nr)
/// For leaf tensors with requires_grad: Edge(accumulate_grad_node, 0)
/// For tensors without requires_grad: empty Edge
Edge gradient_edge(const c10::Tensor& t);

}  // namespace autograd
