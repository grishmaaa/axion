#pragma once

// ============================================================================
// Axion / autograd / engine — Backward Engine
// ============================================================================
//
// Executes the backward pass via topological traversal of the
// computation graph.  Uses Kahn's algorithm (BFS with dependency
// counting) to ensure correct ordering.

#include "c10/core/Tensor.h"
#include "autograd/Edge.h"

namespace autograd {

class Engine {
 public:
  /// Execute backward from a root tensor.
  /// root must be a scalar tensor (numel == 1).
  /// grad_output is the initial gradient (defaults to ones).
  static void backward(
      const c10::Tensor& root,
      const c10::Tensor& grad_output = c10::Tensor());
};

}  // namespace autograd
