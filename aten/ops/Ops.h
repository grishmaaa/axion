#pragma once

// ============================================================================
// Axion / ATen — Ops (public API)
// ============================================================================
//
// The named operations that constitute ATen's public surface.
// Everything above this layer — autograd, axion, user code — calls
// these free functions.
//
// Callers include this header.  They never include TensorIterator.h.
//
// CATEGORIES
// ----------
//   Creation:  zeros, ones, full, rand
//   Unary:     neg, relu, abs
//   Binary:    add, sub, mul
//   Reduce:    sum
//   Linalg:    matmul

#include <cstdint>
#include <vector>

#include "c10/core/ScalarType.h"
#include "c10/core/Tensor.h"

namespace aten {

// ------------------------------------------------------------------
// Creation ops — no input tensor, produce a filled output
// ------------------------------------------------------------------

/// Tensor of zeros with the given shape and dtype.
c10::Tensor zeros(
    std::vector<int64_t> sizes,
    c10::ScalarType dtype = c10::ScalarType::Float32);

/// Tensor of ones with the given shape and dtype.
c10::Tensor ones(
    std::vector<int64_t> sizes,
    c10::ScalarType dtype = c10::ScalarType::Float32);

/// Tensor filled with a scalar value.
c10::Tensor full(
    std::vector<int64_t> sizes,
    double value,
    c10::ScalarType dtype = c10::ScalarType::Float32);

/// Tensor filled with uniform random values in [0, 1).
c10::Tensor rand(
    std::vector<int64_t> sizes,
    c10::ScalarType dtype = c10::ScalarType::Float32);

// ------------------------------------------------------------------
// Unary ops — one tensor in, same shape out
// ------------------------------------------------------------------

/// Element-wise negation.
c10::Tensor neg(const c10::Tensor& a);

/// Element-wise ReLU: max(0, x).
c10::Tensor relu(const c10::Tensor& a);

/// Element-wise absolute value.
c10::Tensor abs(const c10::Tensor& a);

// ------------------------------------------------------------------
// Binary ops — two tensors in, same shape out
// ------------------------------------------------------------------

/// Element-wise addition.
c10::Tensor add(const c10::Tensor& a, const c10::Tensor& b);

/// Element-wise subtraction.
c10::Tensor sub(const c10::Tensor& a, const c10::Tensor& b);

/// Element-wise multiplication.
c10::Tensor mul(const c10::Tensor& a, const c10::Tensor& b);

// ------------------------------------------------------------------
// Reduce ops — tensor in, scalar tensor out
// ------------------------------------------------------------------

/// Sum all elements, returns a scalar tensor (ndim==0, numel==1).
c10::Tensor sum(const c10::Tensor& a);

// ------------------------------------------------------------------
// Linear algebra
// ------------------------------------------------------------------

/// 2D matrix multiplication.  a must be (M,K), b must be (K,N).
/// Returns (M,N).
c10::Tensor matmul(const c10::Tensor& a, const c10::Tensor& b);

}  // namespace aten
