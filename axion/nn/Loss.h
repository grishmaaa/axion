#pragma once

// ============================================================================
// Axion / nn — Loss functions
// ============================================================================

#include "c10/core/Tensor.h"

namespace nn {

/// MSE Loss: mean((pred - target)^2)
c10::Tensor mse_loss(const c10::Tensor& pred, const c10::Tensor& target);

}  // namespace nn
