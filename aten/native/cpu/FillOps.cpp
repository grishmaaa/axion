// ============================================================================
// Axion / ATen / native / cpu — Fill & Creation Ops
// ============================================================================

#include <cassert>
#include <cmath>
#include <random>

#include "c10/core/Dispatcher.h"
#include "c10/core/Tensor.h"
#include "c10/core/CPUAllocator.h"

namespace aten {
namespace native {
namespace cpu {

namespace {

// ---- zeros / ones / full / rand (existing) ----

c10::Tensor cpu_zeros(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  return c10::Tensor::empty(std::move(sizes), dtype);
}

c10::Tensor cpu_ones(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  auto out = c10::Tensor::empty(std::move(sizes), dtype);
  int64_t n = out.numel();
  switch (dtype) {
    case c10::ScalarType::Float32: {
      float* p = out.data_ptr<float>();
      for (int64_t i = 0; i < n; ++i) p[i] = 1.0f;
      break;
    }
    case c10::ScalarType::Float64: {
      double* p = out.data_ptr<double>();
      for (int64_t i = 0; i < n; ++i) p[i] = 1.0;
      break;
    }
    default: assert(false && "ones: unsupported dtype");
  }
  return out;
}

c10::Tensor cpu_full(
    std::vector<int64_t> sizes, double value, c10::ScalarType dtype) {
  auto out = c10::Tensor::empty(std::move(sizes), dtype);
  int64_t n = out.numel();
  switch (dtype) {
    case c10::ScalarType::Float32: {
      float v = static_cast<float>(value);
      float* p = out.data_ptr<float>();
      for (int64_t i = 0; i < n; ++i) p[i] = v;
      break;
    }
    case c10::ScalarType::Float64: {
      double* p = out.data_ptr<double>();
      for (int64_t i = 0; i < n; ++i) p[i] = value;
      break;
    }
    default: assert(false && "full: unsupported dtype");
  }
  return out;
}

c10::Tensor cpu_rand(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  auto out = c10::Tensor::empty(std::move(sizes), dtype);
  int64_t n = out.numel();
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
    default: assert(false && "rand: unsupported dtype");
  }
  return out;
}

// ---- arange ----

c10::Tensor cpu_arange(
    double start, double end, double step, c10::ScalarType dtype) {
  assert(step != 0.0 && "arange: step cannot be zero");
  int64_t n = static_cast<int64_t>(std::ceil((end - start) / step));
  if (n < 0) n = 0;

  auto out = c10::Tensor::empty({n}, dtype);
  switch (dtype) {
    case c10::ScalarType::Float32: {
      float* p = out.data_ptr<float>();
      for (int64_t i = 0; i < n; ++i)
        p[i] = static_cast<float>(start + i * step);
      break;
    }
    case c10::ScalarType::Float64: {
      double* p = out.data_ptr<double>();
      for (int64_t i = 0; i < n; ++i)
        p[i] = start + i * step;
      break;
    }
    default: assert(false && "arange: unsupported dtype");
  }
  return out;
}

// ---- eye ----

c10::Tensor cpu_eye(int64_t n, c10::ScalarType dtype) {
  assert(n > 0 && "eye: n must be positive");
  auto out = c10::Tensor::empty({n, n}, dtype);  // zero-initialized by CPUAllocator
  switch (dtype) {
    case c10::ScalarType::Float32: {
      float* p = out.data_ptr<float>();
      for (int64_t i = 0; i < n; ++i) p[i * n + i] = 1.0f;
      break;
    }
    case c10::ScalarType::Float64: {
      double* p = out.data_ptr<double>();
      for (int64_t i = 0; i < n; ++i) p[i * n + i] = 1.0;
      break;
    }
    default: assert(false && "eye: unsupported dtype");
  }
  return out;
}

// ============================================================================
// Static registration
// ============================================================================

using CreationFn = c10::Tensor (*)(std::vector<int64_t>, c10::ScalarType);
using FullFn = c10::Tensor (*)(std::vector<int64_t>, double, c10::ScalarType);
using ArangeFn = c10::Tensor (*)(double, double, double, c10::ScalarType);
using EyeFn = c10::Tensor (*)(int64_t, c10::ScalarType);

static c10::RegisterKernel r0("aten::zeros", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<CreationFn>(&cpu_zeros)));
static c10::RegisterKernel r1("aten::ones", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<CreationFn>(&cpu_ones)));
static c10::RegisterKernel r2("aten::full", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<FullFn>(&cpu_full)));
static c10::RegisterKernel r3("aten::rand", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<CreationFn>(&cpu_rand)));
static c10::RegisterKernel r4("aten::arange", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<ArangeFn>(&cpu_arange)));
static c10::RegisterKernel r5("aten::eye", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<EyeFn>(&cpu_eye)));

}  // namespace
}  // namespace cpu
}  // namespace native
}  // namespace aten
