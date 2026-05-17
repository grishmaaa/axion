// ============================================================================
// Axion / ATen / native / cpu — BLAS Ops (matmul)
// ============================================================================
//
// CPU kernel for matrix multiplication.
// Uses a naive triple loop with accumulator for now.
// Future: cache-friendly tiling, SIMD, or link against a real BLAS.

#include <cassert>

#include "c10/core/Dispatcher.h"
#include "c10/core/Tensor.h"

namespace aten {
namespace native {
namespace cpu {

namespace {

using MatmulFn = c10::Tensor (*)(const c10::Tensor&, const c10::Tensor&);

c10::Tensor cpu_matmul(const c10::Tensor& a, const c10::Tensor& b) {
  assert(a.defined() && "matmul: first input tensor is undefined");
  assert(b.defined() && "matmul: second input tensor is undefined");
  assert(a.ndim() == 2 && "matmul: first input must be 2D");
  assert(b.ndim() == 2 && "matmul: second input must be 2D");
  assert(a.size(1) == b.size(0) && "matmul: inner dimensions must match");
  assert(a.dtype() == b.dtype() && "matmul: dtype mismatch");

  auto dtype = a.dtype();
  assert((dtype == c10::ScalarType::Float32 ||
          dtype == c10::ScalarType::Float64) &&
         "matmul: unsupported dtype (only Float32/Float64)");

  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b.size(1);

  auto out = c10::Tensor::empty({M, N}, dtype);

  switch (dtype) {
    case c10::ScalarType::Float32: {
      const float* ap = a.data_ptr<float>();
      const float* bp = b.data_ptr<float>();
      float* op = out.data_ptr<float>();
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          float acc = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            acc += ap[i * K + k] * bp[k * N + j];
          }
          op[i * N + j] = acc;
        }
      }
      break;
    }
    case c10::ScalarType::Float64: {
      const double* ap = a.data_ptr<double>();
      const double* bp = b.data_ptr<double>();
      double* op = out.data_ptr<double>();
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          double acc = 0.0;
          for (int64_t k = 0; k < K; ++k) {
            acc += ap[i * K + k] * bp[k * N + j];
          }
          op[i * N + j] = acc;
        }
      }
      break;
    }
    default:
      assert(false && "matmul: unreachable");
  }
  return out;
}

// ============================================================================
// Static registration
// ============================================================================

static c10::RegisterKernel reg_matmul(
    "aten::matmul", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<MatmulFn>(&cpu_matmul)));

}  // namespace
}  // namespace cpu
}  // namespace native
}  // namespace aten
