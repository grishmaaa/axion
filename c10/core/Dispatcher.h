#pragma once

// ============================================================================
// Axion / c10 — Dispatcher
// ============================================================================
//
// The central routing mechanism for all tensor operations.
//
// ARCHITECTURE:
//   User calls aten::add(a, b)
//     → Ops.cpp computes dispatch key from input tensors
//     → Dispatcher looks up "aten::add" + key in its table
//     → Returns a type-erased function pointer (KernelFunction)
//     → Ops.cpp casts to the correct signature and calls
//
// COMPONENTS:
//   KernelFunction — type-erased wrapper around a function pointer
//   OperatorEntry  — per-op dispatch table (array indexed by DispatchKey)
//   Dispatcher     — singleton holding all OperatorEntries
//   RegisterKernel — RAII registrar for static-init-time registration
//
// TYPE ERASURE:
//   We cast function pointers to GenericFnPtr (void(*)()) for storage.
//   The call site casts back to the correct signature.  This is safe
//   as long as the cast-back type matches the original — which is
//   enforced by the op's dispatch wrapper in Ops.cpp.
//
//   Casting between function pointer types is well-defined in C++
//   (reinterpret_cast between function pointer types is legal; calling
//   through the wrong type is UB, but we always call through the original
//   type).  This is zero-overhead — no std::function, no heap allocation.
//
// REGISTRATION PATTERN:
//   // In native/cpu/BinaryOps.cpp:
//   static Tensor cpu_add(const Tensor& a, const Tensor& b) { ... }
//   static RegisterKernel reg("aten::add", DispatchKey::CPU, &cpu_add);
//
// DISPATCH PATTERN:
//   // In aten/ops/Ops.cpp:
//   Tensor add(const Tensor& a, const Tensor& b) {
//     auto key = computeDispatchKey(a, b);
//     auto fn = Dispatcher::singleton()
//                 .lookup("aten::add", key)
//                 .as<Tensor(*)(const Tensor&, const Tensor&)>();
//     return fn(a, b);
//   }

#include <array>
#include <cassert>
#include <string>
#include <unordered_map>

#include "c10/core/DispatchKey.h"
#include "c10/macros/Macros.h"

namespace c10 {

// Forward declare — Tensor.h is heavy, Dispatcher shouldn't pull it in.
class Tensor;

// ============================================================================
// KernelFunction — type-erased function pointer
// ============================================================================

/// A void(*)() used as the storage type for type-erased function pointers.
using GenericFnPtr = void (*)();

class C10_API KernelFunction {
 public:
  KernelFunction() = default;

  /// Wrap any function pointer into a KernelFunction.
  template <typename FnPtr>
  static KernelFunction from(FnPtr fn) {
    KernelFunction kf;
    kf.fn_ = reinterpret_cast<GenericFnPtr>(fn);
    return kf;
  }

  /// Cast back to the original function pointer type.
  /// The caller MUST use the exact same type that was passed to from().
  template <typename FnPtr>
  FnPtr as() const {
    return reinterpret_cast<FnPtr>(fn_);
  }

  /// True if a kernel has been registered.
  explicit operator bool() const { return fn_ != nullptr; }

 private:
  GenericFnPtr fn_ = nullptr;
};

// ============================================================================
// OperatorEntry — per-op dispatch table
// ============================================================================
//
// One entry per registered operator.  Stores one KernelFunction per
// DispatchKey.  Lookup is O(1) — array index by key value.

class C10_API OperatorEntry {
 public:
  OperatorEntry() = default;

  /// Register a kernel for a specific dispatch key.
  /// Overwrites any previously registered kernel for that key.
  void registerKernel(DispatchKey key, KernelFunction fn) {
    auto idx = static_cast<size_t>(key);
    assert(idx < kNumDispatchKeys && "DispatchKey out of range");
    kernels_[idx] = fn;
  }

  /// Look up the kernel for a dispatch key.
  /// Returns a null KernelFunction if no kernel is registered.
  KernelFunction lookup(DispatchKey key) const {
    auto idx = static_cast<size_t>(key);
    if (idx >= kNumDispatchKeys) return KernelFunction();
    return kernels_[idx];
  }

  /// Check if a kernel is registered for a key.
  bool hasKernel(DispatchKey key) const {
    return static_cast<bool>(lookup(key));
  }

 private:
  std::array<KernelFunction, kNumDispatchKeys> kernels_;
};

// ============================================================================
// Dispatcher — singleton holding all operator entries
// ============================================================================

class C10_API Dispatcher {
 public:
  /// Access the global singleton.  Meyer's singleton — created on first
  /// call, safe for static-init-time registration.
  static Dispatcher& singleton();

  /// Get or create the OperatorEntry for an op.
  /// Calling this with a new name implicitly registers the op.
  OperatorEntry& op(const std::string& name) {
    return ops_[name];
  }

  /// Look up an existing op.  Asserts if not found.
  const OperatorEntry& getOp(const std::string& name) const {
    auto it = ops_.find(name);
    assert(it != ops_.end() && "Op not registered in Dispatcher");
    return it->second;
  }

  /// Register a kernel for an op + dispatch key in one call.
  void registerKernel(
      const std::string& name, DispatchKey key, KernelFunction fn) {
    op(name).registerKernel(key, fn);
  }

  /// Convenience: look up a kernel for an op + key.
  KernelFunction lookup(const std::string& name, DispatchKey key) const {
    return getOp(name).lookup(key);
  }

  /// Check if an op is registered at all.
  bool hasOp(const std::string& name) const {
    return ops_.find(name) != ops_.end();
  }

 private:
  Dispatcher() = default;

  // Non-copyable, non-movable singleton.
  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;

  std::unordered_map<std::string, OperatorEntry> ops_;
};

// ============================================================================
// RegisterKernel — RAII static registrar
// ============================================================================
//
// Use at file scope to register a kernel at static init time:
//
//   static RegisterKernel reg("aten::add", DispatchKey::CPU,
//                             KernelFunction::from(&cpu_add));
//
// The constructor runs before main(), populating the Dispatcher.

struct C10_API RegisterKernel {
  RegisterKernel(const char* op_name, DispatchKey key, KernelFunction fn) {
    Dispatcher::singleton().registerKernel(op_name, key, fn);
  }
};

}  // namespace c10
