#pragma once

// ============================================================================
// Axion / c10 — DataPtr
// ============================================================================
//
// DataPtr is the lowest-level memory ownership primitive in the framework.
// It owns exactly one allocation and is responsible for freeing it when
// it goes out of scope or is explicitly cleared.
//
// It is NOT a tensor, NOT storage — just a non-copyable, movable wrapper
// around a raw pointer and its deleter.
//
// KEY DESIGN DECISIONS
// --------------------
//
//  1. Four fields:
//       data_   — the pointer the user reads/writes through
//       ctx_    — the context pointer passed to the deleter
//       del_    — void(*)(void*) function pointer, receives ctx_
//       device_ — Device struct describing where this memory lives
//
//  2. data_ and ctx_ are separate to support pool allocators:
//       For simple allocators (malloc), data_ == ctx_.
//       For pool allocators, data_ is an offset into a block,
//       but ctx_ carries whatever the deleter needs to free correctly.
//       The deleter always receives ctx_, never data_.
//
//  3. Ownership rules:
//       Not copyable — two owners = double free.
//       Movable — ownership transfers cleanly, source becomes null.
//       Destructor calls del_(ctx_) iff both are non-null.
//
//  4. DataPtr does NOT allocate memory.  The allocator layer above it
//     calls malloc/cudaMalloc and constructs a DataPtr with the result.

#include <cstdlib>
#include <utility>

#include "c10/core/Device.h"
#include "c10/macros/Macros.h"

namespace c10 {

// ============================================================================
// Default deleters
// ============================================================================

/// No-op deleter for borrowed (non-owning) pointers.
/// Use this when the DataPtr should observe but not free memory.
inline void deleteNothing(void* /*ctx*/) noexcept {}

/// Standard CPU deleter — calls free().
/// Pairs with memory obtained from malloc/calloc/realloc.
inline void deleteCPU(void* ctx) noexcept {
  std::free(ctx);
}

// ============================================================================
// DataPtr
// ============================================================================

/// Deleter function type: receives the context pointer, frees it.
using Deleter = void (*)(void*);

class C10_API DataPtr {
 public:
  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------

  /// Default — null pointer, no deleter, CPU device.
  DataPtr() noexcept
      : data_(nullptr),
        ctx_(nullptr),
        del_(nullptr),
        device_(DeviceType::CPU, 0) {}

  /// Convenience constructor — data and ctx are the same pointer.
  /// Use this for simple malloc/free allocations.
  ///
  ///   void* raw = std::malloc(1024);
  ///   DataPtr ptr(raw, deleteCPU, Device::cpu());
  ///
  DataPtr(void* data, Deleter del, Device device) noexcept
      : data_(data), ctx_(data), del_(del), device_(device) {}

  /// Full constructor — data and ctx are separate.
  /// Use this for pool allocators where the user-visible pointer
  /// differs from what the deleter needs.
  ///
  ///   void* block = pool.allocate(4096);
  ///   void* user  = static_cast<char*>(block) + offset;
  ///   DataPtr ptr(user, block, poolDeleter, Device::cpu());
  ///
  DataPtr(void* data, void* ctx, Deleter del, Device device) noexcept
      : data_(data), ctx_(ctx), del_(del), device_(device) {}

  // ------------------------------------------------------------------
  // Non-copyable
  // ------------------------------------------------------------------
  C10_DISABLE_COPY(DataPtr);

  // ------------------------------------------------------------------
  // Movable — ownership transfers, source becomes null
  // ------------------------------------------------------------------

  DataPtr(DataPtr&& other) noexcept
      : data_(other.data_),
        ctx_(other.ctx_),
        del_(other.del_),
        device_(other.device_) {
    other.data_ = nullptr;
    other.ctx_ = nullptr;
    other.del_ = nullptr;
  }

  DataPtr& operator=(DataPtr&& other) noexcept {
    if (this != &other) {
      // Free our current allocation first.
      free_();
      data_ = other.data_;
      ctx_ = other.ctx_;
      del_ = other.del_;
      device_ = other.device_;
      other.data_ = nullptr;
      other.ctx_ = nullptr;
      other.del_ = nullptr;
    }
    return *this;
  }

  // ------------------------------------------------------------------
  // Destructor — calls del_(ctx_) if both are non-null
  // ------------------------------------------------------------------

  ~DataPtr() noexcept { free_(); }

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------

  /// The pointer the user reads/writes through.
  void* get() const noexcept { return data_; }

  /// The context pointer that will be passed to the deleter.
  void* get_context() const noexcept { return ctx_; }

  /// The deleter function.
  Deleter get_deleter() const noexcept { return del_; }

  /// Where this memory lives.
  Device device() const noexcept { return device_; }

  /// Boolean conversion — true if data_ is non-null.
  explicit operator bool() const noexcept { return data_ != nullptr; }

  // ------------------------------------------------------------------
  // Modifiers
  // ------------------------------------------------------------------

  /// Free immediately and null all fields.
  /// Safe to call multiple times — second call is a no-op.
  void clear() noexcept {
    free_();
    data_ = nullptr;
    ctx_ = nullptr;
    del_ = nullptr;
  }

  /// Release the context pointer WITHOUT calling the deleter.
  /// The caller takes responsibility for freeing the memory.
  /// Returns the context pointer; all internal fields are nulled.
  void* release_context() noexcept {
    void* ctx = ctx_;
    data_ = nullptr;
    ctx_ = nullptr;
    del_ = nullptr;
    return ctx;
  }

 private:
  /// Call the deleter if both del_ and ctx_ are non-null,
  /// then null out ctx_ and del_ to prevent double-free.
  void free_() noexcept {
    if (del_ && ctx_) {
      del_(ctx_);
    }
    // Null out to prevent double-free if free_() is called again
    // (e.g., clear() followed by destructor).
    ctx_ = nullptr;
    del_ = nullptr;
  }

  void* data_;
  void* ctx_;
  Deleter del_;
  Device device_;
};

}  // namespace c10
