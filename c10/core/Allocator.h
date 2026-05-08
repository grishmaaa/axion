#pragma once

// ============================================================================
// Axion / c10 — Allocator
// ============================================================================
//
// Pure abstract interface — the contract every allocator must fulfill.
//
// The framework talks ONLY to this interface, never to concrete allocators.
// StorageImpl holds an Allocator*, not a CPUAllocator* or CUDAAllocator*.
// That means adding a new allocator (pool, arena, CUDA, Metal) never
// requires modifying StorageImpl.
//
// Three capabilities:
//
//   allocate(nbytes)   — core method: returns a DataPtr owning the memory
//   raw_deleter()      — the deleter function this allocator uses
//   raw_allocate()     — convenience: allocate + unwrap to raw pointer
//   raw_deallocate()   — convenience: free a raw pointer via raw_deleter()
//
// Allocators are not copyable or movable — they are singletons or
// externally managed objects, not value types.

#include <cstddef>

#include "c10/core/DataPtr.h"
#include "c10/core/Device.h"
#include "c10/macros/Macros.h"

namespace c10 {

class C10_API Allocator {
 public:
  virtual ~Allocator() = default;

  // ------------------------------------------------------------------
  // Pure virtual — every concrete allocator must implement these
  // ------------------------------------------------------------------

  /// Allocate `nbytes` of memory.
  /// Returns a DataPtr fully owning the allocation with the correct
  /// deleter already attached.
  ///
  /// nbytes == 0 is valid — returns a DataPtr with a null data pointer
  /// and a valid deleter.  Must not crash.
  virtual DataPtr allocate(size_t nbytes) const = 0;

  /// Return the deleter function this allocator uses for raw pointers.
  ///
  /// Needed for cases where memory arrived from outside — not through
  /// allocate() — but still needs to be freed correctly.  StorageImpl
  /// uses this when wrapping externally provided memory.
  virtual Deleter raw_deleter() const = 0;

  // ------------------------------------------------------------------
  // Convenience helpers — built on top of the pure virtuals
  // ------------------------------------------------------------------

  /// Allocate and return the raw pointer.
  /// The caller takes ownership — must call raw_deallocate() to free.
  void* raw_allocate(size_t nbytes) const {
    auto dptr = allocate(nbytes);
    return dptr.release_context();
  }

  /// Free a raw pointer using this allocator's deleter.
  void raw_deallocate(void* ptr) const {
    auto del = raw_deleter();
    if (del && ptr) {
      del(ptr);
    }
  }

  // ------------------------------------------------------------------
  // Non-copyable, non-movable — allocators are singletons
  // ------------------------------------------------------------------
  Allocator(const Allocator&) = delete;
  Allocator& operator=(const Allocator&) = delete;
  Allocator(Allocator&&) = delete;
  Allocator& operator=(Allocator&&) = delete;

 protected:
  // Only subclasses can construct.
  Allocator() = default;
};

}  // namespace c10
