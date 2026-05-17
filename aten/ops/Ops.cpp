// ============================================================================
// Axion / ATen — Ops implementation
// ============================================================================
//
// This is the only file in ATen that includes TensorIterator.h.
// It is also where matmul and sum live in full, since they don't go
// through TensorIterator (different output shape than input).

#include "aten/ops/Ops.h"

#include <cassert>
#include <random>

#include "aten/core/TensorIterator.h"
#include "c10/core/CPUAllocator.h"

namespace aten {

// ============================================================================
// Creation ops
// ============================================================================

c10::Tensor zeros(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  assert((dtype == c10::ScalarType::Float32 ||
          dtype == c10::ScalarType::Float64) &&
         "zeros: unsupported dtype (only Float32/Float64)");

  auto out = c10::Tensor::empty(std::move(sizes), dtype);
  // CPUAllocator already zero-initializes via std::memset in allocate(),
  // so out is already filled with zeros.  No extra work needed.
  return out;
}

c10::Tensor ones(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  return full(std::move(sizes), 1.0, dtype);
}

c10::Tensor full(std::vector<int64_t> sizes, double value, c10::ScalarType dtype) {
  assert((dtype == c10::ScalarType::Float32 ||
          dtype == c10::ScalarType::Float64) &&
         "full: unsupported dtype (only Float32/Float64)");

  auto out = c10::Tensor::empty(std::move(sizes), dtype);
  int64_t n = out.numel();

  switch (dtype) {
    case c10::ScalarType::Float32: {
      float val = static_cast<float>(value);
      float* p = out.data_ptr<float>();
      for (int64_t i = 0; i < n; ++i) p[i] = val;
      break;
    }
    case c10::ScalarType::Float64: {
      double* p = out.data_ptr<double>();
      for (int64_t i = 0; i < n; ++i) p[i] = value;
      break;
    }
    default:
      assert(false && "full: unreachable");
  }

  return out;
}

c10::Tensor rand(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  assert((dtype == c10::ScalarType::Float32 ||
          dtype == c10::ScalarType::Float64) &&
         "rand: unsupported dtype (only Float32/Float64)");

  auto out = c10::Tensor::empty(std::move(sizes), dtype);
  int64_t n = out.numel();

  // Local engine and distribution — no global state.
  std::random_device rd;
  std::mt19937 gen(rd());

  switch (dtype) {
    case c10::ScalarType::Float32: {
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      float* p = out.data_ptr<float>();
      for (int64_t i = 0; i < n; ++i) p[i] = dist(gen);
      break;
    }
    case c10::ScalarType::Float64: {
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      double* p = out.data_ptr<double>();
      for (int64_t i = 0; i < n; ++i) p[i] = dist(gen);
      break;
    }
    default:
      assert(false && "rand: unreachable");
  }

  return out;
}

// ============================================================================
// Unary ops — delegate to TensorIterator
// ============================================================================

c10::Tensor neg(const c10::Tensor& a) {
  assert(a.defined() && "neg: input tensor is undefined");
  return cpu_kernel_unary(a, [](auto x) { return -x; });
}

c10::Tensor relu(const c10::Tensor& a) {
  assert(a.defined() && "relu: input tensor is undefined");
  return cpu_kernel_unary(a, [](auto x) {
    using T = decltype(x);
    return x > T(0) ? x : T(0);
  });
}

c10::Tensor abs(const c10::Tensor& a) {
  assert(a.defined() && "abs: input tensor is undefined");
  return cpu_kernel_unary(a, [](auto x) {
    using T = decltype(x);
    return x < T(0) ? -x : x;
  });
}

// ============================================================================
// Binary ops — delegate to TensorIterator
// ============================================================================

c10::Tensor add(const c10::Tensor& a, const c10::Tensor& b) {
  assert(a.defined() && "add: first input tensor is undefined");
  assert(b.defined() && "add: second input tensor is undefined");
  return cpu_kernel_binary(a, b, [](auto x, auto y) { return x + y; });
}

c10::Tensor sub(const c10::Tensor& a, const c10::Tensor& b) {
  assert(a.defined() && "sub: first input tensor is undefined");
  assert(b.defined() && "sub: second input tensor is undefined");
  return cpu_kernel_binary(a, b, [](auto x, auto y) { return x - y; });
}

c10::Tensor mul(const c10::Tensor& a, const c10::Tensor& b) {
  assert(a.defined() && "mul: first input tensor is undefined");
  assert(b.defined() && "mul: second input tensor is undefined");
  return cpu_kernel_binary(a, b, [](auto x, auto y) { return x * y; });
}

// ============================================================================
// Reduce — sum (does NOT use TensorIterator)
// ============================================================================

c10::Tensor sum(const c10::Tensor& a) {
  assert(a.defined() && "sum: input tensor is undefined");

  auto dtype = a.dtype();
  assert((dtype == c10::ScalarType::Float32 ||
          dtype == c10::ScalarType::Float64) &&
         "sum: unsupported dtype (only Float32/Float64)");

  // Scalar tensor: sizes={}, ndim==0, numel==1.
  auto out = c10::Tensor::empty({}, dtype);
  int64_t n = a.numel();

  switch (dtype) {
    case c10::ScalarType::Float32: {
      const float* p = a.data_ptr<float>();
      float acc = 0.0f;
      for (int64_t i = 0; i < n; ++i) acc += p[i];
      out.data_ptr<float>()[0] = acc;
      break;
    }
    case c10::ScalarType::Float64: {
      const double* p = a.data_ptr<double>();
      double acc = 0.0;
      for (int64_t i = 0; i < n; ++i) acc += p[i];
      out.data_ptr<double>()[0] = acc;
      break;
    }
    default:
      assert(false && "sum: unreachable");
  }

  return out;
}

// ============================================================================
// Linear algebra — matmul (does NOT use TensorIterator)
// ============================================================================

c10::Tensor matmul(const c10::Tensor& a, const c10::Tensor& b) {
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

  // Output is (M, N).  Allocated via Tensor::empty → CPUAllocator which
  // zero-initializes via std::memset, so the accumulation loop below
  // starts from zero without explicit initialization.
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

}  // namespace aten
