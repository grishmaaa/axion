// ============================================================================
// Axion / ATen / native / cpu — Scalar-Tensor Ops
// ============================================================================
//
// Ops that combine a tensor with a scalar value.
// More efficient than creating a filled tensor + binary op.

#include <cassert>

#include "c10/core/Dispatcher.h"
#include "c10/core/Tensor.h"

namespace aten {
namespace native {
namespace cpu {

namespace {

using ScalarFn = c10::Tensor (*)(const c10::Tensor&, double);

template <typename Op>
c10::Tensor scalar_op(const c10::Tensor& a, double scalar, Op op) {
  assert(a.defined() && "scalar op: input undefined");
  auto dtype = a.dtype();
  auto out = c10::Tensor::empty(
      std::vector<int64_t>(a.sizes().begin(), a.sizes().end()), dtype);
  int64_t n = a.numel();

  switch (dtype) {
    case c10::ScalarType::Float32: {
      const float* in = a.data_ptr<float>();
      float* o = out.data_ptr<float>();
      float s = static_cast<float>(scalar);
      for (int64_t i = 0; i < n; ++i) o[i] = op(in[i], s);
      break;
    }
    case c10::ScalarType::Float64: {
      const double* in = a.data_ptr<double>();
      double* o = out.data_ptr<double>();
      for (int64_t i = 0; i < n; ++i) o[i] = op(in[i], scalar);
      break;
    }
    default:
      assert(false && "scalar op: unsupported dtype");
  }
  return out;
}

c10::Tensor cpu_add_scalar(const c10::Tensor& a, double s) {
  return scalar_op(a, s, [](auto x, auto y) { return x + y; });
}
c10::Tensor cpu_sub_scalar(const c10::Tensor& a, double s) {
  return scalar_op(a, s, [](auto x, auto y) { return x - y; });
}
c10::Tensor cpu_mul_scalar(const c10::Tensor& a, double s) {
  return scalar_op(a, s, [](auto x, auto y) { return x * y; });
}
c10::Tensor cpu_div_scalar(const c10::Tensor& a, double s) {
  return scalar_op(a, s, [](auto x, auto y) { return x / y; });
}

// ============================================================================
// Static registration
// ============================================================================

static c10::RegisterKernel r0("aten::add_scalar", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<ScalarFn>(&cpu_add_scalar)));
static c10::RegisterKernel r1("aten::sub_scalar", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<ScalarFn>(&cpu_sub_scalar)));
static c10::RegisterKernel r2("aten::mul_scalar", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<ScalarFn>(&cpu_mul_scalar)));
static c10::RegisterKernel r3("aten::div_scalar", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<ScalarFn>(&cpu_div_scalar)));

}  // namespace
}  // namespace cpu
}  // namespace native
}  // namespace aten
