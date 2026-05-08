#pragma once

// ============================================================================
// Axion / c10 — CPUAllocator
// ============================================================================
//
// First concrete Allocator implementation.  Allocates 64-byte-aligned
// memory on the CPU heap via aligned_alloc and frees it via free().
//
// 64-byte alignment covers:
//   - Cache line size on x86 (avoids false sharing)
//   - SSE (16B), AVX (32B), AVX-512 (64B) SIMD requirements
//   - Good default for DMA / CUDA host-pinned transfers
//
// Self-registers into AllocatorRegistry at static initialization time
// for DeviceType::CPU — the rest of the framework calls
// GetAllocator(DeviceType::CPU) and gets this back automatically.
//
// DESIGN NOTE — why aligned_alloc over posix_memalign:
//   aligned_alloc is C11/C++17 standard, portable.
//   posix_memalign is POSIX-only.  Both free() correctly on Linux.

#include "c10/core/Allocator.h"
#include "c10/macros/Macros.h"

namespace c10 {

/// Cache-line alignment for all CPU tensor data.
constexpr size_t kCPUAlignment = 64;

class C10_API CPUAllocator final : public Allocator {
 public:
  /// Allocate `nbytes` of 64-byte-aligned CPU memory.
  ///
  /// Zero size:  returns an empty DataPtr (null data, valid device).
  ///             Does NOT call malloc(0) — behavior is implementation-defined.
  ///
  /// Non-zero:   rounds up to a multiple of kCPUAlignment (required by
  ///             aligned_alloc), allocates, returns DataPtr with deleteCPU.
  DataPtr allocate(size_t nbytes) const override;

  /// Returns deleteCPU — calls free().
  Deleter raw_deleter() const override;
};

/// Returns the global singleton CPUAllocator instance.
C10_API CPUAllocator* GetCPUAllocator();

}  // namespace c10
