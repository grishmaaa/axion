#pragma once

// ============================================================================
// Axion / c10 — Storage
// ============================================================================
//
// A thin intrusive_ptr wrapper around StorageImpl.
//
// StorageImpl is the actual object — owns memory, tracks size, holds the
// allocator.  Storage is just the handle that users and TensorImpl hold
// to reach StorageImpl safely with automatic reference counting.
//
// Two Storage objects pointing at the same StorageImpl share ownership.
// When the last one dies the StorageImpl is destroyed and memory freed.
//
// This pattern repeats at every level: Tensor wraps TensorImpl the same way.

#include <utility>

#include "c10/core/Allocator.h"
#include "c10/core/DataPtr.h"
#include "c10/core/Device.h"
#include "c10/core/StorageImpl.h"
#include "c10/macros/Macros.h"
#include "c10/util/intrusive_ptr.h"

namespace c10 {

class C10_API Storage {
 public:
  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------

  /// Default — invalid/null storage.
  Storage() = default;

  /// Wrap an existing intrusive_ptr<StorageImpl>.
  explicit Storage(intrusive_ptr<StorageImpl> impl) noexcept
      : impl_(std::move(impl)) {}

  /// Wrap a raw StorageImpl pointer (takes a new strong reference).
  explicit Storage(StorageImpl* impl) noexcept : impl_(impl) {}

  // ------------------------------------------------------------------
  // Factories
  // ------------------------------------------------------------------

  /// Allocate fresh storage: `nbytes` of memory from `allocator`.
  ///
  ///   auto s = Storage::create(1024, GetCPUAllocator());
  ///
  static Storage create(size_t nbytes, Allocator* allocator) {
    return Storage(make_intrusive<StorageImpl>(nbytes, allocator));
  }

  /// Wrap externally-provided memory via a DataPtr.
  ///
  ///   auto s = Storage::wrap(size, std::move(dp), nullptr);
  ///
  static Storage wrap(size_t nbytes, DataPtr data,
                      Allocator* allocator = nullptr) {
    return Storage(
        make_intrusive<StorageImpl>(nbytes, std::move(data), allocator));
  }

  // ------------------------------------------------------------------
  // Forwarding accessors — all delegate to StorageImpl
  // ------------------------------------------------------------------

  /// Raw pointer to the start of the buffer.
  void* data() const noexcept { return impl_->data(); }

  /// Number of bytes allocated.
  size_t nbytes() const noexcept { return impl_->nbytes(); }

  /// Where this memory lives.
  Device device() const noexcept { return impl_->device(); }

  /// The allocator that produced this memory (may be null).
  Allocator* allocator() const noexcept { return impl_->allocator(); }

  /// Whether this storage can be resized.
  bool resizable() const noexcept { return impl_->resizable(); }

  // ------------------------------------------------------------------
  // Modifiers — forwarded to StorageImpl
  // ------------------------------------------------------------------

  /// Resize the underlying storage.
  void resize(size_t new_nbytes) { impl_->resize(new_nbytes); }

  /// Set the resizable flag.
  void set_resizable(bool resizable) noexcept {
    impl_->set_resizable(resizable);
  }

  // ------------------------------------------------------------------
  // Handle management
  // ------------------------------------------------------------------

  /// True if this handle points to a valid StorageImpl.
  bool valid() const noexcept { return static_cast<bool>(impl_); }

  /// Explicit bool conversion — same as valid().
  explicit operator bool() const noexcept { return valid(); }

  /// Access the underlying intrusive_ptr.
  const intrusive_ptr<StorageImpl>& storage_impl() const noexcept {
    return impl_;
  }

  /// Access the raw StorageImpl pointer.
  StorageImpl* unsafe_get_storage_impl() const noexcept {
    return impl_.get();
  }

  /// Current strong reference count on the StorageImpl.
  size_t use_count() const noexcept { return impl_.use_count(); }

  /// True if this is the sole owner.
  bool unique() const noexcept { return impl_.unique(); }

  // ------------------------------------------------------------------
  // Equality — two Storage handles are equal iff they point to the
  // same StorageImpl (identity, not value equality).
  // ------------------------------------------------------------------

  bool operator==(const Storage& rhs) const noexcept {
    return impl_ == rhs.impl_;
  }
  bool operator!=(const Storage& rhs) const noexcept {
    return impl_ != rhs.impl_;
  }

 private:
  intrusive_ptr<StorageImpl> impl_;
};

}  // namespace c10
