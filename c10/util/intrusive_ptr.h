#pragma once

// ============================================================================
// Axion / c10 — intrusive_ptr<T>
// ============================================================================
//
// An intrusive reference-counting smart pointer, modeled after PyTorch's c10.
//
// KEY DESIGN DECISIONS
// --------------------
//  1. The reference count lives *inside* the managed object (intrusive_ptr_target),
//     not in a separate control block.  This gives:
//       • One fewer heap allocation per object (vs. std::shared_ptr).
//       • Cache-friendly access — the count is on the same cache line as the data.
//       • Ability to obtain an intrusive_ptr from a raw `this` pointer.
//
//  2. Two counts are maintained:
//       • refcount_  — strong references (owners).
//       • weakcount_ — weak references (observers that don't prevent destruction).
//     weakcount_ includes a +1 bias while any strong ref exists, mirroring the
//     weak_ptr / shared_ptr protocol.  The raw storage is freed when weakcount_
//     drops to zero.
//
//  3. Atomic operations use:
//       • memory_order_acq_rel  for decrements (release on the store, acquire on
//         the load that sees zero — ensures all preceding writes are visible to
//         the thread that destroys the object).
//       • memory_order_relaxed  for increments (we already hold a live reference,
//         so no synchronization is needed).
//
//  4. make_intrusive<T>(args...) is the canonical factory.

#include <atomic>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "c10/macros/Macros.h"

namespace c10 {

namespace detail {

// Internal tag for constructing an intrusive_ptr that adopts an already-
// incremented reference (used by the weak_intrusive_ptr::lock() path).
struct DontIncreaseRefcount {};

}  // namespace detail

// ============================================================================
// intrusive_ptr_target — the ref-counted base class
// ============================================================================
//
// Any class managed by intrusive_ptr<T> must inherit from this.
//
// Example:
//   class MyObject : public c10::intrusive_ptr_target {
//   public:
//     int value;
//   };
//
//   auto p = c10::make_intrusive<MyObject>();

class C10_API intrusive_ptr_target {
 public:
  // Allow polymorphic deletion through base pointer.
  virtual ~intrusive_ptr_target() = default;

  // The ref count starts at 0.  The first intrusive_ptr to adopt this object
  // will bump it to 1.
  intrusive_ptr_target() noexcept : refcount_(0), weakcount_(0) {}

  // Move and copy are allowed — they leave the reference counts untouched
  // because the counts belong to the *identity* of the object, not its value.
  intrusive_ptr_target(intrusive_ptr_target&& /*other*/) noexcept
      : refcount_(0), weakcount_(0) {}

  intrusive_ptr_target& operator=(intrusive_ptr_target&& /*other*/) noexcept {
    // Intentionally do NOT touch refcount_ / weakcount_.
    return *this;
  }

  intrusive_ptr_target(const intrusive_ptr_target& /*other*/) noexcept
      : refcount_(0), weakcount_(0) {}

  intrusive_ptr_target& operator=(
      const intrusive_ptr_target& /*other*/) noexcept {
    // Intentionally do NOT touch refcount_ / weakcount_.
    return *this;
  }

 private:
  // Allow the smart-pointer classes to reach the counts directly.
  template <typename T>
  friend class intrusive_ptr;
  template <typename T>
  friend class weak_intrusive_ptr;

  mutable std::atomic<size_t> refcount_;
  mutable std::atomic<size_t> weakcount_;

  // True once the destructor has been called (strong refcount hit 0).
  // Storage stays alive while weakcount_ > 0 even after destruction.
  mutable bool destroyed_ = false;
};

// Forward declaration — needed so intrusive_ptr can friend it.
template <typename T>
class weak_intrusive_ptr;

// ============================================================================
// intrusive_ptr<T> — strong-owning smart pointer
// ============================================================================

template <typename TTarget>
class intrusive_ptr final {
 public:
  // ------------------------------------------------------------------
  // Type requirements
  // ------------------------------------------------------------------
  static_assert(
      std::is_base_of<intrusive_ptr_target, TTarget>::value,
      "intrusive_ptr<T> requires T to derive from intrusive_ptr_target");

  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------

  // Default — null pointer.
  intrusive_ptr() noexcept : target_(nullptr) {}

  // Adopt a raw pointer.  The caller is transferring ownership of one
  // strong reference to us, so we increment the refcount.
  explicit intrusive_ptr(TTarget* target) noexcept : target_(target) {
    if (target_) {
      retain_();
    }
  }

  // Copy — adds a strong reference.
  intrusive_ptr(const intrusive_ptr& rhs) noexcept : target_(rhs.target_) {
    if (target_) {
      retain_();
    }
  }

  // Generalized copy for derived types.
  template <
      typename From,
      typename = std::enable_if_t<std::is_convertible<From*, TTarget*>::value>>
  intrusive_ptr(const intrusive_ptr<From>& rhs) noexcept
      : target_(rhs.target_) {
    if (target_) {
      retain_();
    }
  }

  // Move — pilfer the pointer; no refcount change.
  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = nullptr;
  }

  // Generalized move for derived types.
  template <
      typename From,
      typename = std::enable_if_t<std::is_convertible<From*, TTarget*>::value>>
  intrusive_ptr(intrusive_ptr<From>&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = nullptr;
  }

  // ------------------------------------------------------------------
  // Destructor
  // ------------------------------------------------------------------
  ~intrusive_ptr() noexcept { release_(); }

  // ------------------------------------------------------------------
  // Assignment
  // ------------------------------------------------------------------

  intrusive_ptr& operator=(const intrusive_ptr& rhs) noexcept {
    if (this != &rhs) {
      release_();
      target_ = rhs.target_;
      if (target_) {
        retain_();
      }
    }
    return *this;
  }

  intrusive_ptr& operator=(intrusive_ptr&& rhs) noexcept {
    if (this != &rhs) {
      release_();
      target_ = rhs.target_;
      rhs.target_ = nullptr;
    }
    return *this;
  }

  // ------------------------------------------------------------------
  // Observers
  // ------------------------------------------------------------------

  TTarget* get() const noexcept { return target_; }

  TTarget& operator*() const noexcept {
    assert(target_ && "dereferencing a null intrusive_ptr");
    return *target_;
  }

  TTarget* operator->() const noexcept {
    assert(target_ && "dereferencing a null intrusive_ptr");
    return target_;
  }

  explicit operator bool() const noexcept { return target_ != nullptr; }

  /// Returns the current strong reference count.  Useful for debugging
  /// and tests; do NOT base correctness decisions on this in production.
  size_t use_count() const noexcept {
    if (!target_) {
      return 0;
    }
    return target_->refcount_.load(std::memory_order_acquire);
  }

  /// True when this is the sole owner of the pointee.
  bool unique() const noexcept { return use_count() == 1; }

  // ------------------------------------------------------------------
  // Modifiers
  // ------------------------------------------------------------------

  /// Release ownership and return the raw pointer.
  /// The caller is responsible for eventually calling release or
  /// wrapping it back in an intrusive_ptr.
  TTarget* release() noexcept {
    TTarget* tmp = target_;
    target_ = nullptr;
    return tmp;
  }

  /// Replace the managed object.
  void reset() noexcept {
    release_();
    target_ = nullptr;
  }

  void reset(TTarget* target) noexcept {
    release_();
    target_ = target;
    if (target_) {
      retain_();
    }
  }

  void swap(intrusive_ptr& rhs) noexcept {
    TTarget* tmp = target_;
    target_ = rhs.target_;
    rhs.target_ = tmp;
  }

  // ------------------------------------------------------------------
  // Comparisons
  // ------------------------------------------------------------------

  bool operator==(const intrusive_ptr& rhs) const noexcept {
    return target_ == rhs.target_;
  }
  bool operator!=(const intrusive_ptr& rhs) const noexcept {
    return target_ != rhs.target_;
  }
  bool operator==(std::nullptr_t) const noexcept {
    return target_ == nullptr;
  }
  bool operator!=(std::nullptr_t) const noexcept {
    return target_ != nullptr;
  }

 private:
  // Allow other intrusive_ptr instantiations (for generalized copy/move).
  template <typename U>
  friend class intrusive_ptr;

  // Allow weak_intrusive_ptr::lock() to construct without incrementing.
  template <typename U>
  friend class weak_intrusive_ptr;

  // Private constructor used by weak_intrusive_ptr::lock().  The refcount
  // has already been incremented by the caller via compare_exchange.
  explicit intrusive_ptr(TTarget* target, detail::DontIncreaseRefcount) noexcept
      : target_(target) {}

  void retain_() const noexcept {
    // We already hold a live reference, so no ordering is needed.
    target_->refcount_.fetch_add(1, std::memory_order_relaxed);
  }

  void release_() noexcept {
    if (!target_) {
      return;
    }
    // acq_rel: the release ensures all our writes are visible to the
    // thread that sees the count drop to zero; the acquire on that
    // thread ensures it sees them before running the destructor.
    if (target_->refcount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // We were the last strong owner.  Call the destructor.
      auto* ptr = target_;
      ptr->destroyed_ = true;
      ptr->~TTarget();
      // Free storage only if no weak references are keeping it alive.
      if (ptr->weakcount_.load(std::memory_order_acquire) == 0) {
        ::operator delete(static_cast<void*>(ptr));
      }
    }
    target_ = nullptr;
  }

  TTarget* target_;
};

// ============================================================================
// weak_intrusive_ptr<T> — non-owning observer
// ============================================================================

template <typename TTarget>
class weak_intrusive_ptr final {
 public:
  static_assert(
      std::is_base_of<intrusive_ptr_target, TTarget>::value,
      "weak_intrusive_ptr<T> requires T to derive from intrusive_ptr_target");

  // Default — null.
  weak_intrusive_ptr() noexcept : target_(nullptr) {}

  // Construct from a strong pointer — adds a weak reference.
  explicit weak_intrusive_ptr(const intrusive_ptr<TTarget>& ptr) noexcept
      : target_(ptr.get()) {
    if (target_) {
      target_->weakcount_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Copy.
  weak_intrusive_ptr(const weak_intrusive_ptr& rhs) noexcept
      : target_(rhs.target_) {
    if (target_) {
      target_->weakcount_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Move.
  weak_intrusive_ptr(weak_intrusive_ptr&& rhs) noexcept
      : target_(rhs.target_) {
    rhs.target_ = nullptr;
  }

  ~weak_intrusive_ptr() noexcept { release_weak_(); }

  // Assignment.
  weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) noexcept {
    if (this != &rhs) {
      release_weak_();
      target_ = rhs.target_;
      if (target_) {
        target_->weakcount_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return *this;
  }

  weak_intrusive_ptr& operator=(weak_intrusive_ptr&& rhs) noexcept {
    if (this != &rhs) {
      release_weak_();
      target_ = rhs.target_;
      rhs.target_ = nullptr;
    }
    return *this;
  }

  /// Attempt to promote to a strong reference.
  /// Returns a null intrusive_ptr if the object has already been destroyed.
  intrusive_ptr<TTarget> lock() const noexcept {
    if (!target_) {
      return intrusive_ptr<TTarget>();
    }
    // CAS loop: try to increment refcount only if it is > 0.
    size_t cur = target_->refcount_.load(std::memory_order_relaxed);
    while (cur != 0) {
      if (target_->refcount_.compare_exchange_weak(
              cur, cur + 1, std::memory_order_acq_rel,
              std::memory_order_relaxed)) {
        // Successfully promoted.
        return intrusive_ptr<TTarget>(
            target_, detail::DontIncreaseRefcount{});
      }
      // cur was updated by compare_exchange_weak on failure; retry.
    }
    return intrusive_ptr<TTarget>();
  }

  /// Number of strong references.  0 means the object is dead.
  size_t use_count() const noexcept {
    if (!target_) {
      return 0;
    }
    return target_->refcount_.load(std::memory_order_acquire);
  }

  bool expired() const noexcept { return use_count() == 0; }

 private:
  void release_weak_() noexcept {
    if (!target_) {
      return;
    }
    auto* ptr = target_;
    target_ = nullptr;
    // If this was the last weak ref AND the object was already destroyed
    // by the strong side, we must free the storage.
    if (ptr->weakcount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (ptr->destroyed_) {
        ::operator delete(static_cast<void*>(ptr));
      }
    }
  }

  TTarget* target_;
};

// ============================================================================
// make_intrusive<T>(args...) — canonical factory
// ============================================================================

/// Constructs an object of type T and wraps it in an intrusive_ptr.
/// This is the preferred way to create intrusive-counted objects.
///
///   auto obj = c10::make_intrusive<MyObject>(arg1, arg2);
///
template <typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args) {
  return intrusive_ptr<T>(new T(std::forward<Args>(args)...));
}

// ============================================================================
// Free-standing swap
// ============================================================================

template <typename T>
void swap(intrusive_ptr<T>& lhs, intrusive_ptr<T>& rhs) noexcept {
  lhs.swap(rhs);
}

}  // namespace c10
