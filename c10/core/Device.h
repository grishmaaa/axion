#pragma once

#include <cstdint>
#include "c10/core/DeviceType.h"
#include "c10/macros/Macros.h"

namespace c10 {

// Device is a lightweight value type that represents a physical hardware location.
// It consists of a type (CPU, CUDA, etc.) and an index (for multi-device backends).
//
// DESIGN DECISIONS:
// 1. Lightweight: 2 bytes total (int8_t type + int8_t index).
// 2. Value Semantics: Copying a Device is as cheap as copying a short.
// 3. No Polymorphism: Avoids virtual dispatch overhead for such a fundamental type.
struct C10_API Device {
  DeviceType type;
  int8_t index; // -1 means default device, >= 0 means specific device index.

  // Constructor
  // For CPU, index is generally ignored but stored as 0 for consistency.
  explicit Device(DeviceType t, int8_t i = 0) noexcept : type(t), index(i) {}

  // Convenience static factories
  static Device cpu() noexcept { return Device(DeviceType::CPU, 0); }
  static Device cuda(int8_t i = 0) noexcept { return Device(DeviceType::CUDA, i); }

  // Type queries
  bool is_cpu() const noexcept { return type == DeviceType::CPU; }
  bool is_cuda() const noexcept { return type == DeviceType::CUDA; }

  // Equality operators
  bool operator==(const Device& other) const noexcept {
    return type == other.type && index == other.index;
  }
  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

  // String representation of the device type
  const char* str() const noexcept { return DeviceTypeName(type); }
};

} // namespace c10
