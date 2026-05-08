#pragma once

// ============================================================================
// Axion / c10 — StorageImpl
// ============================================================================
//
// StorageImpl is the first object in the stack that ties memory to metadata.
// It wraps a DataPtr (the raw owned memory) and adds:
//
//   1. nbytes_     — how many bytes are allocated
//   2. allocator_  — which allocator produced this memory (non-owning)
//   3. resizable_  — whether resize operations are allowed
//
// StorageImpl knows NOTHING about shapes, strides, dtypes, or dimensions.
// It just knows: here is a contiguous block of bytes, here is how big it is,
// here is how to free it, here is where it came from.
//
// WHY INTRUSIVE_PTR_TARGET:
//   Multiple TensorImpls can share the same StorageImpl — that's how views
//   work (transpose, slice, reshape don't copy data).  intrusive_ptr gives
//   automatic lifetime management: the memory lives as long as at least one
//   tensor references it.
//
// OWNERSHIP:
//   StorageImpl exclusively owns its DataPtr (DataPtr is move-only).
//   StorageImpl itself is managed via intrusive_ptr<StorageImpl>.
//   The allocator_ pointer is non-owning — allocators are singletons.

#include <cstddef>
#include <utility>

#include "c10/core/Allocator.h"
#include "c10/core/DataPtr.h"
#include "c10/core/Device.h"
#include "c10/macros/Macros.h"
#include "c10/util/intrusive_ptr.h"

namespace c10 {

class C10_API StorageImpl : public intrusive_ptr_target {
 public:
  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------

  /// Allocate-on-construction: asks the given allocator for `nbytes`
  /// of memory.  This is the normal construction path.
  ///
  ///   auto storage = c10::make_intrusive<StorageImpl>(
  ///       1024, GetAllocator(DeviceType::CPU));
  ///
  StorageImpl(size_t nbytes, Allocator* allocator);

  /// Wrap externally-provided memory: takes ownership of an existing
  /// DataPtr.  The allocator may be null if the memory was not produced
  /// by our allocator system (e.g., memory-mapped files, foreign buffers).
  ///
  ///   DataPtr dp(external_ptr, custom_deleter, Device::cpu());
  ///   auto storage = c10::make_intrusive<StorageImpl>(
  ///       size, std::move(dp), /*allocator=*/nullptr);
  ///
  StorageImpl(size_t nbytes, DataPtr data, Allocator* allocator);

  // StorageImpl is not copyable — DataPtr is move-only.
  C10_DISABLE_COPY(StorageImpl);

  // Movable — ownership of the DataPtr transfers.
  StorageImpl(StorageImpl&& other) noexcept;
  StorageImpl& operator=(StorageImpl&& other) noexcept;

  ~StorageImpl() override = default;

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------

  /// Raw pointer to the start of the buffer.
  void* data() const noexcept { return data_.get(); }

  /// Number of bytes allocated.
  size_t nbytes() const noexcept { return nbytes_; }

  /// Where this memory lives (CPU, CUDA, etc.).
  Device device() const noexcept { return data_.device(); }

  /// The allocator that produced this memory (may be null for external).
  Allocator* allocator() const noexcept { return allocator_; }

  /// Whether this storage can be resized.
  bool resizable() const noexcept { return resizable_; }

  /// The underlying DataPtr — const ref for inspection.
  const DataPtr& data_ptr() const noexcept { return data_; }

  // ------------------------------------------------------------------
  // Modifiers
  // ------------------------------------------------------------------

  /// Allow or disallow resizing.
  void set_resizable(bool resizable) noexcept { resizable_ = resizable; }

  /// Resize the storage to `new_nbytes`.
  ///
  /// - If not resizable, asserts / aborts.
  /// - If new_nbytes == nbytes_, no-op.
  /// - Otherwise allocates new memory via allocator_, copies the
  ///   min(old, new) bytes, and replaces the DataPtr.
  void resize(size_t new_nbytes);

 private:
  DataPtr data_;
  size_t nbytes_;
  Allocator* allocator_;  // non-owning — allocators are singletons
  bool resizable_;
};

}  // namespace c10
