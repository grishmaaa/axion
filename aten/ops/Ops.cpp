// ============================================================================
// Axion / ATen — Ops implementation (dispatch layer)
// ============================================================================
//
// PUBLIC API → Dispatcher bridge.  Each function validates inputs,
// computes dispatch key, and calls through the type-erased kernel.
//
// Shape ops (transpose, reshape, view) are metadata-only — they
// create new TensorImpl instances with different sizes/strides but
// share the same Storage.  These don't go through dispatch.

#include "aten/ops/Ops.h"

#include <cassert>
#include <numeric>

#include "c10/core/Dispatcher.h"

namespace aten {

// ============================================================================
// Dispatch key helpers
// ============================================================================

namespace {

inline c10::DispatchKey dk(const c10::Tensor& a) {
  return a.dispatch_key_set().highestPriorityKey();
}

inline c10::DispatchKey dk(const c10::Tensor& a, const c10::Tensor& b) {
  return (a.dispatch_key_set() | b.dispatch_key_set()).highestPriorityKey();
}

template <typename FnPtr>
FnPtr lookup(const char* name, c10::DispatchKey key) {
  auto fn = c10::Dispatcher::singleton().lookup(name, key);
  assert(fn && "No kernel registered");
  return fn.as<FnPtr>();
}

}  // namespace

// ============================================================================
// Type aliases
// ============================================================================

using UnaryFn   = c10::Tensor (*)(const c10::Tensor&);
using BinaryFn  = c10::Tensor (*)(const c10::Tensor&, const c10::Tensor&);
using ScalarFn  = c10::Tensor (*)(const c10::Tensor&, double);
using ReduceFn  = c10::Tensor (*)(const c10::Tensor&);
using CreateFn  = c10::Tensor (*)(std::vector<int64_t>, c10::ScalarType);
using FullFn    = c10::Tensor (*)(std::vector<int64_t>, double, c10::ScalarType);
using ArangeFn  = c10::Tensor (*)(double, double, double, c10::ScalarType);
using EyeFn     = c10::Tensor (*)(int64_t, c10::ScalarType);

// ============================================================================
// Creation ops
// ============================================================================

c10::Tensor zeros(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  return lookup<CreateFn>("aten::zeros", c10::DispatchKey::CPU)(
      std::move(sizes), dtype);
}

c10::Tensor ones(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  return lookup<CreateFn>("aten::ones", c10::DispatchKey::CPU)(
      std::move(sizes), dtype);
}

c10::Tensor full(
    std::vector<int64_t> sizes, double value, c10::ScalarType dtype) {
  return lookup<FullFn>("aten::full", c10::DispatchKey::CPU)(
      std::move(sizes), value, dtype);
}

c10::Tensor rand(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  return lookup<CreateFn>("aten::rand", c10::DispatchKey::CPU)(
      std::move(sizes), dtype);
}

c10::Tensor arange(
    double start, double end, double step, c10::ScalarType dtype) {
  return lookup<ArangeFn>("aten::arange", c10::DispatchKey::CPU)(
      start, end, step, dtype);
}

c10::Tensor eye(int64_t n, c10::ScalarType dtype) {
  return lookup<EyeFn>("aten::eye", c10::DispatchKey::CPU)(n, dtype);
}

// ============================================================================
// Unary ops
// ============================================================================

#define UNARY_OP(NAME)                                           \
  c10::Tensor NAME(const c10::Tensor& a) {                      \
    assert(a.defined() && #NAME ": input undefined");            \
    return lookup<UnaryFn>("aten::" #NAME, dk(a))(a);            \
  }

UNARY_OP(neg)
UNARY_OP(relu)
UNARY_OP(abs)
UNARY_OP(exp)
UNARY_OP(log)
UNARY_OP(sqrt)
UNARY_OP(tanh)
UNARY_OP(sigmoid)
UNARY_OP(gelu)

#undef UNARY_OP

// ============================================================================
// Binary ops
// ============================================================================

#define BINARY_OP(NAME)                                              \
  c10::Tensor NAME(const c10::Tensor& a, const c10::Tensor& b) {    \
    assert(a.defined() && #NAME ": first input undefined");          \
    assert(b.defined() && #NAME ": second input undefined");         \
    return lookup<BinaryFn>("aten::" #NAME, dk(a, b))(a, b);        \
  }

BINARY_OP(add)
BINARY_OP(sub)
BINARY_OP(mul)
BINARY_OP(div)
BINARY_OP(pow)

#undef BINARY_OP

// ============================================================================
// Scalar ops
// ============================================================================

#define SCALAR_OP(NAME)                                              \
  c10::Tensor NAME(const c10::Tensor& a, double scalar) {           \
    assert(a.defined() && #NAME ": input undefined");                \
    return lookup<ScalarFn>("aten::" #NAME, dk(a))(a, scalar);       \
  }

SCALAR_OP(add_scalar)
SCALAR_OP(sub_scalar)
SCALAR_OP(mul_scalar)
SCALAR_OP(div_scalar)

#undef SCALAR_OP

// ============================================================================
// Reduce ops
// ============================================================================

c10::Tensor sum(const c10::Tensor& a) {
  assert(a.defined() && "sum: input undefined");
  return lookup<ReduceFn>("aten::sum", dk(a))(a);
}

c10::Tensor mean(const c10::Tensor& a) {
  assert(a.defined() && "mean: input undefined");
  return lookup<ReduceFn>("aten::mean", dk(a))(a);
}

// ============================================================================
// Shape ops — metadata only, no dispatch needed
// ============================================================================

c10::Tensor transpose(const c10::Tensor& a, int64_t dim0, int64_t dim1) {
  assert(a.defined() && "transpose: input undefined");
  int64_t ndim = a.ndim();
  assert(dim0 >= 0 && dim0 < ndim && "transpose: dim0 out of range");
  assert(dim1 >= 0 && dim1 < ndim && "transpose: dim1 out of range");

  if (dim0 == dim1) return a;

  // Copy sizes and strides, swap the two dimensions
  auto sizes_span = a.sizes();
  auto strides_span = a.strides();
  std::vector<int64_t> new_sizes(sizes_span.begin(), sizes_span.end());
  std::vector<int64_t> new_strides(strides_span.begin(), strides_span.end());

  std::swap(new_sizes[dim0], new_sizes[dim1]);
  std::swap(new_strides[dim0], new_strides[dim1]);

  // Create a new TensorImpl sharing the same Storage
  auto impl = c10::make_intrusive<c10::TensorImpl>(
      a.storage(), a.dtype(), std::move(new_sizes), std::move(new_strides),
      a.storage_offset());

  return c10::Tensor(std::move(impl));
}

c10::Tensor reshape(const c10::Tensor& a, std::vector<int64_t> shape) {
  assert(a.defined() && "reshape: input undefined");

  // Handle -1: infer one dimension
  int64_t total = a.numel();
  int64_t infer_idx = -1;
  int64_t known_product = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      assert(infer_idx == -1 && "reshape: at most one -1 allowed");
      infer_idx = static_cast<int64_t>(i);
    } else {
      assert(shape[i] > 0 && "reshape: dimensions must be positive (or -1)");
      known_product *= shape[i];
    }
  }
  if (infer_idx >= 0) {
    assert(known_product > 0 && total % known_product == 0 &&
           "reshape: cannot infer dimension");
    shape[infer_idx] = total / known_product;
  }

  // Verify total elements match
  int64_t new_total = 1;
  for (auto s : shape) new_total *= s;
  assert(new_total == total && "reshape: total elements must match");

  // If not contiguous, make contiguous first
  c10::Tensor src = a.is_contiguous() ? a : contiguous(a);

  // Compute default strides for the new shape
  std::vector<int64_t> new_strides(shape.size());
  if (!shape.empty()) {
    new_strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
      new_strides[i] = new_strides[i + 1] * shape[i + 1];
    }
  }

  auto impl = c10::make_intrusive<c10::TensorImpl>(
      src.storage(), src.dtype(), std::move(shape), std::move(new_strides),
      src.storage_offset());

  return c10::Tensor(std::move(impl));
}

c10::Tensor view(const c10::Tensor& a, std::vector<int64_t> shape) {
  assert(a.is_contiguous() && "view: input must be contiguous");
  return reshape(a, std::move(shape));
}

c10::Tensor contiguous(const c10::Tensor& a) {
  assert(a.defined() && "contiguous: input undefined");
  if (a.is_contiguous()) return a;
  return lookup<UnaryFn>("aten::contiguous", dk(a))(a);
}

// ============================================================================
// Linear algebra
// ============================================================================

c10::Tensor matmul(const c10::Tensor& a, const c10::Tensor& b) {
  assert(a.defined() && "matmul: first input undefined");
  assert(b.defined() && "matmul: second input undefined");
  return lookup<BinaryFn>("aten::matmul", dk(a, b))(a, b);
}

}  // namespace aten
