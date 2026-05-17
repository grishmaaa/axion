#include "c10/core/TensorImpl.h"

#include <numeric>
#include <algorithm>

namespace c10 {

TensorImpl::TensorImpl(
    Storage storage,
    ScalarType dtype,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides,
    int64_t storage_offset)
    : storage_(std::move(storage)),
      dtype_(dtype),
      sizes_(std::move(sizes)),
      strides_(std::move(strides)),
      storage_offset_(storage_offset),
      key_set_(storage_ ? dispatchKeySetForDevice(storage_.device())
                        : DispatchKeySet()) {
  refresh_numel();
  refresh_contiguous();
}

TensorImpl::TensorImpl(
    Storage storage,
    ScalarType dtype,
    std::vector<int64_t> sizes,
    int64_t storage_offset)
    : storage_(std::move(storage)),
      dtype_(dtype),
      sizes_(std::move(sizes)),
      strides_(default_strides(sizes_)),
      storage_offset_(storage_offset),
      key_set_(storage_ ? dispatchKeySetForDevice(storage_.device())
                        : DispatchKeySet()) {
  refresh_numel();
  refresh_contiguous();
}

void TensorImpl::refresh_numel() noexcept {
  if (sizes_.empty()) {
    numel_ = 1;
  } else {
    numel_ = 1;
    for (auto s : sizes_) {
      numel_ *= s;
    }
  }
}

void TensorImpl::refresh_contiguous() noexcept {
  if (sizes_.empty()) {
    is_contiguous_ = true;
    return;
  }

  int64_t expected_stride = 1;
  for (int64_t i = static_cast<int64_t>(sizes_.size()) - 1; i >= 0; --i) {
    if (strides_[i] != expected_stride) {
      is_contiguous_ = false;
      return;
    }
    expected_stride *= sizes_[i];
  }
  is_contiguous_ = true;
}

std::vector<int64_t> TensorImpl::default_strides(const std::vector<int64_t>& sizes) {
  int64_t d = static_cast<int64_t>(sizes.size());
  std::vector<int64_t> strides(d);
  if (d == 0) {
    return strides;
  }
  strides[d - 1] = 1;
  for (int64_t i = d - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
  return strides;
}

} // namespace c10
