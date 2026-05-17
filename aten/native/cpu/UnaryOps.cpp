// ============================================================================
// Axion / ATen / native / cpu — Unary Ops
// ============================================================================

#include <cmath>

#include "aten/core/TensorIterator.h"
#include "c10/core/Dispatcher.h"

namespace aten {
namespace native {
namespace cpu {

namespace {

using UnaryFn = c10::Tensor (*)(const c10::Tensor&);

c10::Tensor cpu_neg(const c10::Tensor& a) {
  return cpu_kernel_unary(a, [](auto x) { return -x; });
}

c10::Tensor cpu_relu(const c10::Tensor& a) {
  return cpu_kernel_unary(a, [](auto x) {
    using T = decltype(x);
    return x > T(0) ? x : T(0);
  });
}

c10::Tensor cpu_abs(const c10::Tensor& a) {
  return cpu_kernel_unary(a, [](auto x) {
    using T = decltype(x);
    return x < T(0) ? -x : x;
  });
}

c10::Tensor cpu_exp(const c10::Tensor& a) {
  return cpu_kernel_unary(a, [](auto x) { return std::exp(x); });
}

c10::Tensor cpu_log(const c10::Tensor& a) {
  return cpu_kernel_unary(a, [](auto x) { return std::log(x); });
}

c10::Tensor cpu_sqrt(const c10::Tensor& a) {
  return cpu_kernel_unary(a, [](auto x) { return std::sqrt(x); });
}

c10::Tensor cpu_tanh(const c10::Tensor& a) {
  return cpu_kernel_unary(a, [](auto x) { return std::tanh(x); });
}

c10::Tensor cpu_sigmoid(const c10::Tensor& a) {
  return cpu_kernel_unary(a, [](auto x) {
    using T = decltype(x);
    return T(1) / (T(1) + std::exp(-x));
  });
}

c10::Tensor cpu_gelu(const c10::Tensor& a) {
  return cpu_kernel_unary(a, [](auto x) {
    using T = decltype(x);
    // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * T(0.5) * (T(1) + std::erf(x / std::sqrt(T(2))));
  });
}

// ============================================================================
// Static registration
// ============================================================================

static c10::RegisterKernel r0("aten::neg", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<UnaryFn>(&cpu_neg)));
static c10::RegisterKernel r1("aten::relu", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<UnaryFn>(&cpu_relu)));
static c10::RegisterKernel r2("aten::abs", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<UnaryFn>(&cpu_abs)));
static c10::RegisterKernel r3("aten::exp", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<UnaryFn>(&cpu_exp)));
static c10::RegisterKernel r4("aten::log", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<UnaryFn>(&cpu_log)));
static c10::RegisterKernel r5("aten::sqrt", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<UnaryFn>(&cpu_sqrt)));
static c10::RegisterKernel r6("aten::tanh", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<UnaryFn>(&cpu_tanh)));
static c10::RegisterKernel r7("aten::sigmoid", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<UnaryFn>(&cpu_sigmoid)));
static c10::RegisterKernel r8("aten::gelu", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<UnaryFn>(&cpu_gelu)));

}  // namespace
}  // namespace cpu
}  // namespace native
}  // namespace aten
