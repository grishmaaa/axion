// ============================================================================
// Axion / nn — Loss function implementations
// ============================================================================

#include "axion/nn/Loss.h"

#include "autograd/Variable.h"
#include "aten/ops/Ops.h"

namespace nn {

c10::Tensor mse_loss(const c10::Tensor& pred, const c10::Tensor& target) {
  auto diff = autograd::sub(pred, target);
  auto sq = autograd::mul(diff, diff);
  return autograd::mean(sq);
}

}  // namespace nn
