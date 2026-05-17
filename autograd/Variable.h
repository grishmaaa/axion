#pragma once

// ============================================================================
// Axion / autograd — Variable (autograd-aware operations)
// ============================================================================
//
// These functions mirror aten:: ops but also record onto the
// computation graph when GradMode is enabled and at least one input
// requires grad.
//
// User code should call autograd::add() instead of aten::add() when
// gradients are needed.

#include "c10/core/Tensor.h"
#include "c10/core/ScalarType.h"
#include <vector>

namespace autograd {

// ------------------------------------------------------------------
// Binary ops
// ------------------------------------------------------------------
c10::Tensor add(const c10::Tensor& a, const c10::Tensor& b);
c10::Tensor sub(const c10::Tensor& a, const c10::Tensor& b);
c10::Tensor mul(const c10::Tensor& a, const c10::Tensor& b);
c10::Tensor div(const c10::Tensor& a, const c10::Tensor& b);
c10::Tensor pow(const c10::Tensor& a, const c10::Tensor& b);

// ------------------------------------------------------------------
// Unary ops
// ------------------------------------------------------------------
c10::Tensor neg(const c10::Tensor& a);
c10::Tensor relu(const c10::Tensor& a);
c10::Tensor abs(const c10::Tensor& a);
c10::Tensor exp(const c10::Tensor& a);
c10::Tensor log(const c10::Tensor& a);
c10::Tensor sqrt(const c10::Tensor& a);
c10::Tensor tanh(const c10::Tensor& a);
c10::Tensor sigmoid(const c10::Tensor& a);
c10::Tensor gelu(const c10::Tensor& a);

// ------------------------------------------------------------------
// Scalar ops
// ------------------------------------------------------------------
c10::Tensor add_scalar(const c10::Tensor& a, double s);
c10::Tensor sub_scalar(const c10::Tensor& a, double s);
c10::Tensor mul_scalar(const c10::Tensor& a, double s);
c10::Tensor div_scalar(const c10::Tensor& a, double s);

// ------------------------------------------------------------------
// Reduce ops
// ------------------------------------------------------------------
c10::Tensor sum(const c10::Tensor& a);
c10::Tensor mean(const c10::Tensor& a);

// ------------------------------------------------------------------
// Matmul
// ------------------------------------------------------------------
c10::Tensor matmul(const c10::Tensor& a, const c10::Tensor& b);

}  // namespace autograd
