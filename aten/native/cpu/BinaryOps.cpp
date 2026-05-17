// ============================================================================
// Axion / ATen / native / cpu — Binary Ops
// ============================================================================

#include <cmath>

#include "aten/core/TensorIterator.h"
#include "c10/core/Dispatcher.h"

namespace aten {
namespace native {
namespace cpu {

namespace {

using BinaryFn = c10::Tensor (*)(const c10::Tensor&, const c10::Tensor&);

c10::Tensor cpu_add(const c10::Tensor& a, const c10::Tensor& b) {
  return cpu_kernel_binary(a, b, [](auto x, auto y) { return x + y; });
}

c10::Tensor cpu_sub(const c10::Tensor& a, const c10::Tensor& b) {
  return cpu_kernel_binary(a, b, [](auto x, auto y) { return x - y; });
}

c10::Tensor cpu_mul(const c10::Tensor& a, const c10::Tensor& b) {
  return cpu_kernel_binary(a, b, [](auto x, auto y) { return x * y; });
}

c10::Tensor cpu_div(const c10::Tensor& a, const c10::Tensor& b) {
  return cpu_kernel_binary(a, b, [](auto x, auto y) { return x / y; });
}

c10::Tensor cpu_pow(const c10::Tensor& a, const c10::Tensor& b) {
  return cpu_kernel_binary(a, b, [](auto x, auto y) { return std::pow(x, y); });
}

// ============================================================================
// Static registration
// ============================================================================

static c10::RegisterKernel r0("aten::add", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<BinaryFn>(&cpu_add)));
static c10::RegisterKernel r1("aten::sub", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<BinaryFn>(&cpu_sub)));
static c10::RegisterKernel r2("aten::mul", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<BinaryFn>(&cpu_mul)));
static c10::RegisterKernel r3("aten::div", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<BinaryFn>(&cpu_div)));
static c10::RegisterKernel r4("aten::pow", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<BinaryFn>(&cpu_pow)));

}  // namespace
}  // namespace cpu
}  // namespace native
}  // namespace aten
