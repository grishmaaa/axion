#pragma once

// ============================================================================
// Axion / c10 — AutogradMetaInterface
// ============================================================================
//
// Abstract base stored on TensorImpl via unique_ptr. Lightweight — no
// Tensor or Node dependencies. The concrete AutogradMeta (in autograd/)
// adds grad_, grad_fn_, etc.
//
// Lives in c10/ because TensorImpl needs it. Autograd code downcasts.

#include "c10/macros/Macros.h"
#include <memory>

namespace c10 {

struct C10_API AutogradMetaInterface {
  virtual ~AutogradMetaInterface() = default;

  /// Whether this tensor participates in gradient computation.
  bool requires_grad_ = false;
};

}  // namespace c10
