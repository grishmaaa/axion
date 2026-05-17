#pragma once

// ============================================================================
// Axion / nn — Loss functions
// ============================================================================

#include "c10/core/Tensor.h"

namespace nn {

/// MSE Loss: mean((pred - target)^2)
c10::Tensor mse_loss(const c10::Tensor& pred, const c10::Tensor& target);

/// Cross-Entropy Loss (with built-in log-softmax):
///   -mean_i [ logits[i, t_i] - log(sum_j exp(logits[i, j])) ]
///
/// @param logits  (batch, num_classes) raw scores
/// @param targets (batch,) integer class indices stored as float
c10::Tensor cross_entropy_loss(const c10::Tensor& logits,
                                const c10::Tensor& targets);

}  // namespace nn
