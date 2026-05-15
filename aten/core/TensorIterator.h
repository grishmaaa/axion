#pragma once

// ============================================================================
// Axion / ATen — TensorIterator
// ============================================================================
//
// The loop abstraction that every element-wise op in ATen delegates to.
//
// Without it, every op — add, sub, relu, neg, abs, mul — writes the same
// five lines: validate inputs, allocate output, cast pointer, loop, return.
// TensorIterator extracts that skeleton so each op only provides the one
// line that differs: the math.
//
// It is not a dispatch table.  It is not an allocator.  It is not an op
// itself.  It is a header-only utility that takes tensors and a lambda,
// and handles everything except the arithmetic.
//
// DESIGN DECISIONS
// ----------------
//  1. Header-only — both functions are templated on the lambda type F.
//     The compiler cannot instantiate a template without seeing its full
//     definition at the call site.  Everything lives here, no paired .cpp.
//
//  2. Dtype dispatch is a switch inside both functions.  Only Float32 and
//     Float64 are supported for v1.  The switch is duplicated — acceptable
//     at this scale, and avoids abstraction overhead.
//
//  3. The output tensor is always NEW.  It never aliases an input.  It is
//     always contiguous (allocated via Tensor::empty with default strides).
//     It always has the same dtype as the input(s).
//
//  4. No broadcasting.  No type promotion.  Same-shape, same-dtype only
//     for v1.  These are explicit non-goals to keep the foundation clean.

#include <cassert>
#include <cstdint>

#include "c10/core/ScalarType.h"
#include "c10/core/Tensor.h"

namespace aten {

// ============================================================================
// cpu_kernel_unary — one input, one output
// ============================================================================
//
// For ops that take one tensor and produce one tensor of the same shape
// and dtype.  Neg, relu, abs.
//
// The lambda receives one element and returns one element:
//   [](float x) { return -x; }
//
// Usage:
//   auto out = aten::cpu_kernel_unary(input, [](float x){ return -x; });
//
template <typename F>
c10::Tensor cpu_kernel_unary(const c10::Tensor& input, F op) {
  assert(input.defined() && "cpu_kernel_unary: input tensor is undefined");

  auto dtype = input.dtype();
  assert((dtype == c10::ScalarType::Float32 ||
          dtype == c10::ScalarType::Float64) &&
         "cpu_kernel_unary: unsupported dtype (only Float32/Float64)");

  auto out = c10::Tensor::empty(input.sizes(), dtype);
  int64_t n = input.numel();

  switch (dtype) {
    case c10::ScalarType::Float32: {
      const float* in_ptr = input.data_ptr<float>();
      float* out_ptr = out.data_ptr<float>();
      for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = op(in_ptr[i]);
      }
      break;
    }
    case c10::ScalarType::Float64: {
      const double* in_ptr = input.data_ptr<double>();
      double* out_ptr = out.data_ptr<double>();
      for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = op(in_ptr[i]);
      }
      break;
    }
    default:
      assert(false && "cpu_kernel_unary: unreachable");
  }

  return out;
}

// ============================================================================
// cpu_kernel_binary — two inputs, one output
// ============================================================================
//
// For ops that take two tensors of the same shape and dtype and produce
// one tensor of the same shape and dtype.  Add, sub, mul.
//
// The lambda receives two elements and returns one element:
//   [](float x, float y) { return x + y; }
//
// Usage:
//   auto out = aten::cpu_kernel_binary(a, b, [](float x, float y){ return x + y; });
//
template <typename F>
c10::Tensor cpu_kernel_binary(const c10::Tensor& a, const c10::Tensor& b, F op) {
  assert(a.defined() && "cpu_kernel_binary: first input tensor is undefined");
  assert(b.defined() && "cpu_kernel_binary: second input tensor is undefined");
  assert(a.sizes() == b.sizes() &&
         "cpu_kernel_binary: shape mismatch between inputs");
  assert(a.dtype() == b.dtype() &&
         "cpu_kernel_binary: dtype mismatch between inputs");

  auto dtype = a.dtype();
  assert((dtype == c10::ScalarType::Float32 ||
          dtype == c10::ScalarType::Float64) &&
         "cpu_kernel_binary: unsupported dtype (only Float32/Float64)");

  auto out = c10::Tensor::empty(a.sizes(), dtype);
  int64_t n = a.numel();

  switch (dtype) {
    case c10::ScalarType::Float32: {
      const float* a_ptr = a.data_ptr<float>();
      const float* b_ptr = b.data_ptr<float>();
      float* out_ptr = out.data_ptr<float>();
      for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = op(a_ptr[i], b_ptr[i]);
      }
      break;
    }
    case c10::ScalarType::Float64: {
      const double* a_ptr = a.data_ptr<double>();
      const double* b_ptr = b.data_ptr<double>();
      double* out_ptr = out.data_ptr<double>();
      for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = op(a_ptr[i], b_ptr[i]);
      }
      break;
    }
    default:
      assert(false && "cpu_kernel_binary: unreachable");
  }

  return out;
}

}  // namespace aten
