#pragma once

// ============================================================================
// Axion / autograd / functions — AccumulateGrad
// ============================================================================
//
// Leaf node in the computation graph. Instead of propagating gradients
// further, it stores them on the tensor's AutogradMeta.grad_.

#include "autograd/Node.h"
#include "autograd/AutogradMeta.h"
#include "aten/ops/Ops.h"

namespace autograd {

class AccumulateGrad : public Node {
 public:
  explicit AccumulateGrad(c10::Tensor variable)
      : variable_(std::move(variable)) {
    // Leaf node — no next_edges.
  }

  std::vector<c10::Tensor> apply(
      std::vector<c10::Tensor> grads) override {
    assert(!grads.empty() && "AccumulateGrad: no input gradients");

    auto* meta = get_autograd_meta(variable_);
    assert(meta && "AccumulateGrad: variable has no autograd meta");

    if (meta->grad_.defined()) {
      meta->grad_ = aten::add(meta->grad_, grads[0]);
    } else {
      meta->grad_ = grads[0];
    }

    return {};  // No further propagation — this is a leaf.
  }

 private:
  c10::Tensor variable_;
};

}  // namespace autograd
