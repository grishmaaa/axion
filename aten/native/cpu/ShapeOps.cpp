// ============================================================================
// Axion / ATen / native / cpu — Shape Ops
// ============================================================================
//
// contiguous() is the only shape op that needs a kernel (data copy).
// transpose/reshape/view are metadata-only and live in Ops.cpp directly.

#include <cassert>
#include <cstring>

#include "c10/core/Dispatcher.h"
#include "c10/core/Tensor.h"

namespace aten {
namespace native {
namespace cpu {

namespace {

using ContiguousFn = c10::Tensor (*)(const c10::Tensor&);

/// Copy a strided tensor to a contiguous tensor.
/// Handles arbitrary strides by computing the source index from
/// multi-dimensional coordinates.
c10::Tensor cpu_contiguous(const c10::Tensor& a) {
  assert(a.defined() && "contiguous: input undefined");

  if (a.is_contiguous()) return a;  // already contiguous, return same

  auto dtype = a.dtype();
  auto sizes_span = a.sizes();
  std::vector<int64_t> sizes(sizes_span.begin(), sizes_span.end());
  auto strides_span = a.strides();
  std::vector<int64_t> strides(strides_span.begin(), strides_span.end());
  int64_t ndim = a.ndim();
  int64_t n = a.numel();

  auto out = c10::Tensor::empty(sizes, dtype);

  auto elem_size = c10::elementSize(dtype);

  // Generic strided copy using coordinate iteration
  const char* src = static_cast<const char*>(a.data_ptr());
  char* dst = static_cast<char*>(out.data_ptr());

  std::vector<int64_t> coord(ndim, 0);
  for (int64_t i = 0; i < n; ++i) {
    // Compute source offset from strides
    int64_t src_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      src_offset += coord[d] * strides[d];
    }
    std::memcpy(dst + i * elem_size, src + src_offset * elem_size, elem_size);

    // Increment coordinate (row-major)
    for (int64_t d = ndim - 1; d >= 0; --d) {
      if (++coord[d] < sizes[d]) break;
      coord[d] = 0;
    }
  }
  return out;
}

// ============================================================================
// Static registration
// ============================================================================

static c10::RegisterKernel r0("aten::contiguous", c10::DispatchKey::CPU,
    c10::KernelFunction::from(static_cast<ContiguousFn>(&cpu_contiguous)));

}  // namespace
}  // namespace cpu
}  // namespace native
}  // namespace aten
