// ============================================================================
// Axion / c10 — AllocatorRegistry implementation
// ============================================================================
//
// This is the first .cpp file in c10/core/.  After this, CMake will
// build c10 as a STATIC library instead of INTERFACE (header-only).
//
// The registry is a fixed-size array — one slot per DeviceType.
// Currently supports CPU(0) and CUDA(1), with room for 8 total.
// No mutex — registration happens at static init time before any
// concurrent access.

#include "c10/core/AllocatorRegistry.h"

#include <array>
#include <cassert>

namespace c10 {

namespace {

// Fixed-size array indexed by DeviceType underlying int8_t value.
// 8 slots is more than enough for foreseeable device types.
constexpr int kMaxDeviceTypes = 8;
std::array<Allocator*, kMaxDeviceTypes> g_allocators = {};

}  // namespace

void RegisterAllocator(DeviceType t, Allocator* allocator) {
  auto idx = static_cast<int>(t);
  assert(idx >= 0 && idx < kMaxDeviceTypes &&
         "DeviceType index out of range for allocator registry");
  g_allocators[static_cast<size_t>(idx)] = allocator;
}

Allocator* GetAllocator(DeviceType t) {
  auto idx = static_cast<int>(t);
  if (idx < 0 || idx >= kMaxDeviceTypes) {
    return nullptr;
  }
  return g_allocators[static_cast<size_t>(idx)];
}

}  // namespace c10
