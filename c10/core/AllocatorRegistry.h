#pragma once

// ============================================================================
// Axion / c10 — AllocatorRegistry
// ============================================================================
//
// Global lookup table: DeviceType → Allocator*.
//
// At startup, before any tensor is created, concrete allocators register
// themselves via RegisterAllocator().  From then on the rest of the framework
// calls GetAllocator(DeviceType::CPU) and gets back whatever was registered.
//
// Implementation: fixed-size array indexed by the DeviceType integer value.
// Not a map, not a hash table.  DeviceType has at most a handful of values.
// Array lookup is zero overhead — no reason to pay map overhead for
// something called on every tensor construction.

#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/macros/Macros.h"

namespace c10 {

/// Register an allocator for a device type.
/// Only one allocator per DeviceType — second registration overwrites
/// the first.  Typically called at static init time.
C10_API void RegisterAllocator(DeviceType t, Allocator* allocator);

/// Retrieve the allocator for a device type.
/// Returns nullptr if none registered.
C10_API Allocator* GetAllocator(DeviceType t);

/// Convenience overload — takes a full Device, extracts the type.
inline Allocator* GetAllocator(Device d) {
  return GetAllocator(d.type);
}

}  // namespace c10
