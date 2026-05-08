// ============================================================================
// Axion / c10 — CPUAllocator implementation
// ============================================================================

#include "c10/core/CPUAllocator.h"

#include <cstdlib>
#include <cstring>

#include "c10/core/AllocatorRegistry.h"

namespace c10 {

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

namespace {

/// Round `n` up to the next multiple of `alignment`.
/// alignment must be a power of two.
inline size_t roundUpToAlignment(size_t n, size_t alignment) {
  return (n + alignment - 1) & ~(alignment - 1);
}

}  // namespace

// ----------------------------------------------------------------------------
// CPUAllocator implementation
// ----------------------------------------------------------------------------

DataPtr CPUAllocator::allocate(size_t nbytes) const {
  if (nbytes == 0) {
    // Zero-size: return a valid but empty DataPtr.
    // Do NOT call malloc(0) — its return value is implementation-defined.
    return DataPtr(nullptr, nullptr, deleteCPU, Device::cpu());
  }

  // aligned_alloc requires size to be a multiple of alignment.
  size_t alloc_size = roundUpToAlignment(nbytes, kCPUAlignment);
  void* ptr = std::aligned_alloc(kCPUAlignment, alloc_size);

  if (!ptr) {
    // Allocation failed.  In a production framework you'd throw;
    // for now return a null DataPtr so the caller can check.
    return DataPtr(nullptr, nullptr, deleteCPU, Device::cpu());
  }

  // Zero-initialize for safety — prevents reading uninitialized memory.
  std::memset(ptr, 0, alloc_size);

  return DataPtr(ptr, deleteCPU, Device::cpu());
}

Deleter CPUAllocator::raw_deleter() const {
  return deleteCPU;
}

// ----------------------------------------------------------------------------
// Global singleton and self-registration
// ----------------------------------------------------------------------------

CPUAllocator* GetCPUAllocator() {
  // Meyer's singleton — constructed on first call, destroyed at exit.
  static CPUAllocator instance;
  return &instance;
}

namespace {

// Self-registration at static init time.
// Before main() runs, the CPUAllocator registers itself into the
// AllocatorRegistry for DeviceType::CPU.
struct CPUAllocatorRegistrar {
  CPUAllocatorRegistrar() {
    RegisterAllocator(DeviceType::CPU, GetCPUAllocator());
  }
};

static CPUAllocatorRegistrar g_cpu_allocator_registrar;

}  // namespace

}  // namespace c10
