#pragma once

// ============================================================================
// Axion / c10 — Tensor
// ============================================================================
//
// The user-facing handle to a tensor.
//
// Tensor wraps intrusive_ptr<TensorImpl> exactly the way Storage wraps
// intrusive_ptr<StorageImpl>.  It adds no fields of its own — it is a
// thin, zero-overhead handle whose only job is to:
//
//   1. Own a strong reference to the underlying TensorImpl.
//   2. Forward every accessor/mutator through to that TensorImpl.
//   3. Provide convenient factories so users never touch TensorImpl directly.
//
// Copying a Tensor is cheap — it increments the refcount and both handles
// now point at the same TensorImpl (same metadata, same Storage, same
// memory).  This is how views, slicing, and aliasing work: the Tensor
// handle is lightweight, the TensorImpl carries the real state.
//
// Move is even cheaper — just pointer pilfering, no atomic ops.
//
// RELATIONSHIP TO PYTORCH:
//   PyTorch's at::Tensor is the moral equivalent.  In their stack:
//     Tensor → TensorImpl → StorageImpl → DataPtr
//   We mirror this exactly.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "c10/core/AllocatorRegistry.h"
#include "c10/core/CPUAllocator.h"
#include "c10/core/ScalarType.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/macros/Macros.h"
#include "c10/util/intrusive_ptr.h"

namespace c10 {

class C10_API Tensor {
 public:
  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------

  /// Default — undefined tensor.  No TensorImpl, no memory.
  Tensor() = default;

  /// Wrap an existing intrusive_ptr<TensorImpl>.
  explicit Tensor(intrusive_ptr<TensorImpl> impl) noexcept
      : impl_(std::move(impl)) {}

  /// Wrap a raw TensorImpl pointer (takes a new strong reference).
  explicit Tensor(TensorImpl* impl) noexcept : impl_(impl) {}

  // ------------------------------------------------------------------
  // Factories
  // ------------------------------------------------------------------

  /// Allocate an uninitialized tensor with the given shape and dtype on CPU.
  ///
  ///   auto t = Tensor::empty({3, 4}, ScalarType::Float32);
  ///
  static Tensor empty(
      std::vector<int64_t> sizes,
      ScalarType dtype = ScalarType::Float32,
      Allocator* allocator = nullptr) {
    if (!allocator) {
      allocator = GetAllocator(DeviceType::CPU);
    }
    // Fallback: if the registry hasn't been populated (static archive
    // linker stripping), use the direct singleton accessor.
    if (!allocator) {
      allocator = GetCPUAllocator();
    }
    // Compute total bytes: product(sizes) * elementSize(dtype).
    int64_t numel = 1;
    for (auto s : sizes) {
      numel *= s;
    }
    size_t nbytes = static_cast<size_t>(numel) * elementSize(dtype);
    auto storage = Storage::create(nbytes, allocator);
    return Tensor(
        make_intrusive<TensorImpl>(std::move(storage), dtype, std::move(sizes)));
  }

  /// Wrap externally-owned memory as a tensor (non-owning — uses
  /// deleteNothing so the caller retains ownership of the buffer).
  ///
  ///   float buf[12];
  ///   auto t = Tensor::from_blob(buf, {3, 4}, ScalarType::Float32);
  ///
  static Tensor from_blob(
      void* data,
      std::vector<int64_t> sizes,
      ScalarType dtype = ScalarType::Float32) {
    int64_t numel = 1;
    for (auto s : sizes) {
      numel *= s;
    }
    size_t nbytes = static_cast<size_t>(numel) * elementSize(dtype);
    // deleteNothing — caller owns the memory, we just observe it.
    auto storage =
        Storage::wrap(nbytes, DataPtr(data, deleteNothing, Device::cpu()));
    return Tensor(
        make_intrusive<TensorImpl>(std::move(storage), dtype, std::move(sizes)));
  }

  // ------------------------------------------------------------------
  // Forwarding accessors — all delegate to TensorImpl
  // ------------------------------------------------------------------

  /// Shape — number of elements along each dimension.
  const std::vector<int64_t>& sizes() const noexcept {
    return impl_->sizes();
  }

  /// Strides — element-skip per dimension.
  const std::vector<int64_t>& strides() const noexcept {
    return impl_->strides();
  }

  /// Element type.
  ScalarType dtype() const noexcept { return impl_->dtype(); }

  /// Total number of elements.
  int64_t numel() const noexcept { return impl_->numel(); }

  /// Number of dimensions.
  int64_t ndim() const noexcept { return impl_->ndim(); }

  /// Element offset from the start of storage.
  int64_t storage_offset() const noexcept { return impl_->storage_offset(); }

  /// Device the data lives on.
  Device device() const noexcept { return impl_->device(); }

  /// The set of dispatch keys for routing this tensor through the dispatcher.
  DispatchKeySet dispatch_key_set() const noexcept {
    return impl_->dispatch_key_set();
  }

  /// True if the tensor is contiguous in memory (row-major, no gaps).
  bool is_contiguous() const noexcept { return impl_->is_contiguous(); }

  /// Whether this tensor participates in gradient computation.
  bool requires_grad() const noexcept { return impl_->requires_grad(); }

  /// Raw access to autograd metadata (may be null).
  AutogradMetaInterface* autograd_meta() const noexcept {
    return impl_->autograd_meta();
  }

  /// Access the underlying TensorImpl (non-const, for autograd code).
  TensorImpl* unsafeGetTensorImplRaw() const noexcept {
    return impl_.get();
  }

  /// Size along a specific dimension.
  int64_t size(int64_t dim) const noexcept {
    assert(dim >= 0 && dim < ndim() && "dimension out of range");
    return impl_->sizes()[static_cast<size_t>(dim)];
  }

  /// Stride along a specific dimension.
  int64_t stride(int64_t dim) const noexcept {
    assert(dim >= 0 && dim < ndim() && "dimension out of range");
    return impl_->strides()[static_cast<size_t>(dim)];
  }

  // ------------------------------------------------------------------
  // Data access
  // ------------------------------------------------------------------

  /// Untyped raw pointer, adjusted for storage_offset.
  void* data_ptr() const noexcept { return impl_->data_ptr(); }

  /// Typed data pointer — casts the raw pointer to T*.
  ///
  ///   float* p = t.data_ptr<float>();
  ///
  template <typename T>
  T* data_ptr() const noexcept {
    return static_cast<T*>(impl_->data_ptr());
  }

  // ------------------------------------------------------------------
  // Storage access
  // ------------------------------------------------------------------

  /// The underlying Storage handle.
  const Storage& storage() const noexcept { return impl_->storage(); }

  // ------------------------------------------------------------------
  // Handle management
  // ------------------------------------------------------------------

  /// True if this handle points to a valid TensorImpl.
  bool defined() const noexcept { return static_cast<bool>(impl_); }

  /// Explicit bool conversion — same as defined().
  explicit operator bool() const noexcept { return defined(); }

  /// Access the underlying intrusive_ptr.
  const intrusive_ptr<TensorImpl>& unsafeGetTensorImpl() const noexcept {
    return impl_;
  }

  /// Access the raw TensorImpl pointer.
  TensorImpl* unsafe_get_tensor_impl() const noexcept {
    return impl_.get();
  }

  /// Current strong reference count on the TensorImpl.
  size_t use_count() const noexcept { return impl_.use_count(); }

  /// True if this is the sole owner.
  bool unique() const noexcept { return impl_.unique(); }

  // ------------------------------------------------------------------
  // Equality — identity-based, not value-based.
  // Two Tensor handles are equal iff they point to the same TensorImpl.
  // ------------------------------------------------------------------

  bool operator==(const Tensor& rhs) const noexcept {
    return impl_ == rhs.impl_;
  }
  bool operator!=(const Tensor& rhs) const noexcept {
    return impl_ != rhs.impl_;
  }

 private:
  intrusive_ptr<TensorImpl> impl_;
};

}  // namespace c10
