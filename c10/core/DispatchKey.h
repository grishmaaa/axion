#pragma once

// ============================================================================
// Axion / c10 — DispatchKey
// ============================================================================
//
// A DispatchKey identifies which kernel implementation should handle an
// operation.  Every tensor carries a DispatchKeySet (bitset of keys).
// When an op is called, the dispatcher computes the union of all input
// tensors' key sets, picks the highest-priority key, and jumps to the
// kernel registered for that key.
//
// KEY PRIORITY ORDER (highest → lowest numeric value wins):
//   AutogradCUDA > AutogradCPU > CUDA > CPU
//
// This means autograd intercepts BEFORE the backend kernel runs, which
// is exactly what we need: autograd records the op on the graph, then
// redispatches to the actual backend.
//
// DESIGN DECISIONS:
//   1. uint8_t underlying type — max 64 keys via DispatchKeySet (uint64_t).
//   2. DispatchKeySet is a value type (8 bytes, fits in a register).
//   3. highestPriorityKey() is O(log N) via bit scan — fast enough for
//      dispatch on every op call.

#include <cstdint>
#include "c10/core/Device.h"
#include "c10/core/DeviceType.h"
#include "c10/macros/Macros.h"

namespace c10 {

// ============================================================================
// DispatchKey enum
// ============================================================================

enum class DispatchKey : uint8_t {
  // Backend keys — actual compute implementations.
  CPU = 0,
  CUDA = 1,

  // Autograd keys — intercept ops to build the computation graph.
  // These have higher priority than backend keys so they run first.
  AutogradCPU = 2,
  AutogradCUDA = 3,

  // Sentinel — not a real key.
  NumKeys = 4,
  Undefined = 255,
};

constexpr size_t kNumDispatchKeys = static_cast<size_t>(DispatchKey::NumKeys);

/// Human-readable name for debugging and error messages.
inline const char* dispatchKeyToString(DispatchKey key) {
  switch (key) {
    case DispatchKey::CPU: return "CPU";
    case DispatchKey::CUDA: return "CUDA";
    case DispatchKey::AutogradCPU: return "AutogradCPU";
    case DispatchKey::AutogradCUDA: return "AutogradCUDA";
    default: return "Undefined";
  }
}

/// Map a DeviceType to its backend DispatchKey.
inline DispatchKey backendDispatchKey(DeviceType d) {
  switch (d) {
    case DeviceType::CPU: return DispatchKey::CPU;
    case DeviceType::CUDA: return DispatchKey::CUDA;
    default: return DispatchKey::Undefined;
  }
}

// ============================================================================
// DispatchKeySet — compact bitset of active dispatch keys
// ============================================================================
//
// Stored as a uint64_t where bit N corresponds to DispatchKey(N).
// Supports set union (|), intersection (&), difference (-), and
// extraction of the highest-priority (highest-numbered) key.

class C10_API DispatchKeySet {
 public:
  DispatchKeySet() = default;

  /// Construct a set containing a single key.
  explicit DispatchKeySet(DispatchKey k)
      : repr_(k == DispatchKey::Undefined
                  ? 0
                  : (1ull << static_cast<uint8_t>(k))) {}

  /// Check if a specific key is in the set.
  bool has(DispatchKey k) const {
    return (repr_ & (1ull << static_cast<uint8_t>(k))) != 0;
  }

  /// Return a new set with the given key added.
  DispatchKeySet add(DispatchKey k) const {
    return DispatchKeySet(repr_ | (1ull << static_cast<uint8_t>(k)));
  }

  /// Return a new set with the given key removed.
  DispatchKeySet remove(DispatchKey k) const {
    return DispatchKeySet(repr_ & ~(1ull << static_cast<uint8_t>(k)));
  }

  /// Set union.
  DispatchKeySet operator|(DispatchKeySet other) const {
    return DispatchKeySet(repr_ | other.repr_);
  }

  /// Set intersection.
  DispatchKeySet operator&(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & other.repr_);
  }

  /// Set difference.
  DispatchKeySet operator-(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & ~other.repr_);
  }

  bool empty() const { return repr_ == 0; }

  /// Return the highest-priority key in the set.
  /// Higher enum values = higher priority.
  /// Returns Undefined if the set is empty.
  DispatchKey highestPriorityKey() const {
    if (repr_ == 0) return DispatchKey::Undefined;
    // Find position of the highest set bit (portable).
    uint8_t pos = 0;
    uint64_t v = repr_;
    while (v > 1) {
      v >>= 1;
      ++pos;
    }
    return static_cast<DispatchKey>(pos);
  }

  uint64_t raw() const { return repr_; }

 private:
  explicit DispatchKeySet(uint64_t repr) : repr_(repr) {}
  uint64_t repr_ = 0;
};

// ============================================================================
// Helpers for computing dispatch keys from tensors
// ============================================================================

/// Build a DispatchKeySet for a device (backend key only).
inline DispatchKeySet dispatchKeySetForDevice(Device d) {
  return DispatchKeySet(backendDispatchKey(d.type));
}

}  // namespace c10
