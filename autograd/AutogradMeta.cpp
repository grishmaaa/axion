// ============================================================================
// Axion / autograd — AutogradMeta implementation
// ============================================================================
//
// gradient_edge() creates AccumulateGrad nodes for leaf tensors lazily.

#include "autograd/AutogradMeta.h"
#include "autograd/functions/Accumulate.h"

namespace autograd {

Edge gradient_edge(const c10::Tensor& t) {
  auto* meta = get_autograd_meta(t);

  // No autograd meta or doesn't require grad — empty edge.
  if (!meta || !meta->requires_grad_) {
    return Edge();
  }

  // Non-leaf tensor — use its grad_fn.
  if (meta->grad_fn_) {
    return Edge(meta->grad_fn_, meta->output_nr_);
  }

  // Leaf tensor with requires_grad — get or create AccumulateGrad.
  auto accumulator = meta->grad_accumulator_.lock();
  if (!accumulator) {
    accumulator = std::make_shared<AccumulateGrad>(t);
    meta->grad_accumulator_ = accumulator;
  }
  return Edge(std::move(accumulator), 0);
}

}  // namespace autograd
