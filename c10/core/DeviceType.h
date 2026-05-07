#pragma once
#include <cstdint>
#include "c10/macros/Macros.h"

namespace c10 {

// DeviceType defines the supported hardware backends.
// We use a fixed-width integer (int8_t) for compact storage in the Device struct.
enum class C10_API DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,
  // Future-proofing: HIP, XPU, Metal, etc. can be added here.
};

// Returns a human-readable string representation of the DeviceType.
// Useful for logging, error messages, and debugging.
inline const char* DeviceTypeName(DeviceType t) noexcept {
  switch (t) {
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::CUDA:
      return "cuda";
    default:
      return "unknown";
  }
}

} // namespace c10
