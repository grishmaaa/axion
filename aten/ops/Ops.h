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
//   Creation:  zeros, ones, full, rand, arange, eye
//   Unary:     neg, relu, abs, exp, log, sqrt, tanh, sigmoid, gelu
//   Binary:    add, sub, mul, div, pow
//   Scalar:    add_scalar, sub_scalar, mul_scalar, div_scalar
//   Reduce:    sum, mean
//   Shape:     transpose, reshape, view, contiguous
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

/// 1D tensor with values from start to end (exclusive), step 1.
c10::Tensor arange(
    double start, double end, double step = 1.0,
    c10::ScalarType dtype = c10::ScalarType::Float32);

/// Identity matrix of size n x n.
c10::Tensor eye(
    int64_t n,
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

/// Element-wise exponential.
c10::Tensor exp(const c10::Tensor& a);

/// Element-wise natural logarithm.
c10::Tensor log(const c10::Tensor& a);

/// Element-wise square root.
c10::Tensor sqrt(const c10::Tensor& a);

/// Element-wise hyperbolic tangent.
c10::Tensor tanh(const c10::Tensor& a);

/// Element-wise sigmoid: 1 / (1 + exp(-x)).
c10::Tensor sigmoid(const c10::Tensor& a);

/// Element-wise GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
c10::Tensor gelu(const c10::Tensor& a);

// ------------------------------------------------------------------
// Binary ops — two tensors in, same shape out
// ------------------------------------------------------------------

/// Element-wise addition.
c10::Tensor add(const c10::Tensor& a, const c10::Tensor& b);

/// Element-wise subtraction.
c10::Tensor sub(const c10::Tensor& a, const c10::Tensor& b);

/// Element-wise multiplication.
c10::Tensor mul(const c10::Tensor& a, const c10::Tensor& b);

/// Element-wise division.
c10::Tensor div(const c10::Tensor& a, const c10::Tensor& b);

/// Element-wise power.
c10::Tensor pow(const c10::Tensor& a, const c10::Tensor& b);

// ------------------------------------------------------------------
// Scalar-tensor ops — tensor and scalar in, same shape out
// ------------------------------------------------------------------

/// Add scalar to every element.
c10::Tensor add_scalar(const c10::Tensor& a, double scalar);

/// Subtract scalar from every element.
c10::Tensor sub_scalar(const c10::Tensor& a, double scalar);

/// Multiply every element by scalar.
c10::Tensor mul_scalar(const c10::Tensor& a, double scalar);

/// Divide every element by scalar.
c10::Tensor div_scalar(const c10::Tensor& a, double scalar);

// ------------------------------------------------------------------
// Reduce ops — tensor in, scalar tensor out
// ------------------------------------------------------------------

/// Sum all elements, returns a scalar tensor (ndim==0, numel==1).
c10::Tensor sum(const c10::Tensor& a);

/// Mean of all elements, returns a scalar tensor.
c10::Tensor mean(const c10::Tensor& a);

// ------------------------------------------------------------------
// Shape ops — change tensor layout without modifying data
// ------------------------------------------------------------------

/// Swap two dimensions. Returns a view (shares storage, not contiguous).
c10::Tensor transpose(const c10::Tensor& a, int64_t dim0, int64_t dim1);

/// Reshape to new sizes. Returns a view if contiguous, else copies.
c10::Tensor reshape(const c10::Tensor& a, std::vector<int64_t> shape);

/// Same as reshape but asserts input is contiguous.
c10::Tensor view(const c10::Tensor& a, std::vector<int64_t> shape);

/// If already contiguous, returns same tensor. Otherwise copies to
/// a fresh contiguous tensor.
c10::Tensor contiguous(const c10::Tensor& a);

// ------------------------------------------------------------------
// Linear algebra
// ------------------------------------------------------------------

/// 2D matrix multiplication.  a must be (M,K), b must be (K,N).
/// Returns (M,N).
c10::Tensor matmul(const c10::Tensor& a, const c10::Tensor& b);

}  // namespace aten
