// ============================================================================
// Axion / ATen / native / cpu — Reduce Ops
// ============================================================================

#include <cassert>

#include "c10/core/Dispatcher.h"
#include "c10/core/Tensor.h"

namespace aten {
namespace native {
namespace cpu {

namespace {

using ReduceFn = c10::Tensor (*)(const c10::Tensor&);

c10::Tensor cpu_sum(const c10::Tensor& a) {
  assert(a.defined() && "sum: input undefined");
  auto dtype = a.dtype();
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
      assert(false && "sum: unsupported dtype");
  }
  return out;
}

c10::Tensor cpu_mean(const c10::Tensor& a) {
  assert(a.defined() && "mean: input undefined");
  auto dtype = a.dtype();
  auto out = c10::Tensor::empty({}, dtype);
  int64_t n = a.numel();
  assert(n > 0 && "mean: cannot take mean of empty tensor");

  switch (dtype) {
    case c10::ScalarType::Float32: {
      const float* p = a.data_ptr<float>();
      float acc = 0.0f;
      for (int64_t i = 0; i < n; ++i) acc += p[i];
      out.data_ptr<float>()[0] = acc / static_cast<float>(n);
      break;
    }
    case c10::ScalarType::Float64: {
      const double* p = a.data_ptr<double>();
      double acc = 0.0;
      for (int64_t i = 0; i < n; ++i) acc += p[i];
      out.data_ptr<double>()[0] = acc / static_cast<double>(n);
      break;
    }
    default:
      assert(false && "mean: unsupported dtype");
  }
  return out;
}

// ============================================================================
// Static registration
// ============================================================================

static c10::RegisterKernel r0("aten::sum", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<ReduceFn>(&cpu_sum)));
static c10::RegisterKernel r1("aten::mean", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<ReduceFn>(&cpu_mean)));

}  // namespace
}  // namespace cpu
}  // namespace native
}  // namespace aten
