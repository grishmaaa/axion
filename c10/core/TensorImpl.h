#pragma once

#include <vector>
#include <numeric>
#include <cstdint>

#include "c10/core/AutogradMetaInterface.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/Storage.h"
#include "c10/core/ScalarType.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/macros/Macros.h"

namespace c10 {

/**
 * TensorImpl is the core object that represents a Tensor.
 * It ties together Storage (raw memory) with metadata (shape, strides, dtype).
 */
class C10_API TensorImpl : public intrusive_ptr_target {
 public:
  /**
   * Construct a TensorImpl with explicit storage, sizes, strides, and offset.
   */
  TensorImpl(
      Storage storage,
      ScalarType dtype,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides,
      int64_t storage_offset = 0);

  /**
   * Construct a TensorImpl with explicit storage, sizes, and offset.
   * Strides are automatically computed for a contiguous layout.
   */
  TensorImpl(
      Storage storage,
      ScalarType dtype,
      std::vector<int64_t> sizes,
      int64_t storage_offset = 0);

  C10_DISABLE_COPY(TensorImpl);

  virtual ~TensorImpl() = default;

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------

  const std::vector<int64_t>& sizes() const noexcept { return sizes_; }
  const std::vector<int64_t>& strides() const noexcept { return strides_; }
  ScalarType dtype() const noexcept { return dtype_; }
  int64_t storage_offset() const noexcept { return storage_offset_; }
  int64_t numel() const noexcept { return numel_; }
  int64_t ndim() const noexcept { return static_cast<int64_t>(sizes_.size()); }

  const Storage& storage() const noexcept { return storage_; }
  Device device() const noexcept { return storage_.device(); }

  /// The dispatch key set for this tensor — used by the Dispatcher
  /// to route ops to the correct backend kernel.
  DispatchKeySet dispatch_key_set() const noexcept { return key_set_; }

  /**
   * Returns the raw pointer to the data, adjusted for storage_offset.
   */
  void* data() const noexcept {
    if (!storage_) return nullptr;
    return static_cast<char*>(storage_.data()) + (storage_offset_ * elementSize(dtype_));
  }

  /**
   * Returns the raw pointer to the data, adjusted for storage_offset.
   * Alias for data() to match PyTorch naming.
   */
  void* data_ptr() const noexcept {
    return data();
  }

  /**
   * Checks if the tensor is contiguous in memory.
   * Caches the result in is_contiguous_ for O(1) access.
   */
  bool is_contiguous() const noexcept { return is_contiguous_; }

  /**
   * Autograd metadata slot — nullable, zero-cost when unused.
   * Owned via unique_ptr. Created lazily when requires_grad is set.
   */
  AutogradMetaInterface* autograd_meta() const noexcept {
    return autograd_meta_.get();
  }
  void set_autograd_meta(
      std::unique_ptr<AutogradMetaInterface> meta) noexcept {
    autograd_meta_ = std::move(meta);
  }

  /// Whether this tensor requires gradient computation.
  bool requires_grad() const noexcept {
    return autograd_meta_ && autograd_meta_->requires_grad_;
  }

 private:
  void refresh_numel() noexcept;
  void refresh_contiguous() noexcept;
  static std::vector<int64_t> default_strides(const std::vector<int64_t>& sizes);

  Storage storage_;
  ScalarType dtype_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t storage_offset_;
  int64_t numel_;
  bool is_contiguous_;
  DispatchKeySet key_set_;

  /// Autograd metadata. Null for inference-only tensors (zero overhead).
  std::unique_ptr<AutogradMetaInterface> autograd_meta_;
};

} // namespace c10
