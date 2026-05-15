#pragma once

#include <cstdint>
#include <cstddef>

#include "c10/macros/Macros.h"

namespace c10 {

/**
 * ScalarType represents the data type of the elements in a Tensor.
 */
enum class ScalarType : int8_t {
  Float32,
  Float64,
  Int32,
  Int64,
  Int8,
  UInt8,
  BFloat16,
  Float16,
  Undefined,
};

/**
 * Returns the size in bytes for a given ScalarType.
 */
inline size_t elementSize(ScalarType type) {
  switch (type) {
    case ScalarType::Float32:
      return 4;
    case ScalarType::Float64:
      return 8;
    case ScalarType::Int32:
      return 4;
    case ScalarType::Int64:
      return 8;
    case ScalarType::Int8:
      return 1;
    case ScalarType::UInt8:
      return 1;
    case ScalarType::BFloat16:
      return 2;
    case ScalarType::Float16:
      return 2;
    default:
      return 0;
  }
}

/**
 * Returns a human-readable name for the ScalarType.
 */
inline const char* toString(ScalarType type) {
  switch (type) {
    case ScalarType::Float32:
      return "Float32";
    case ScalarType::Float64:
      return "Float64";
    case ScalarType::Int32:
      return "Int32";
    case ScalarType::Int64:
      return "Int64";
    case ScalarType::Int8:
      return "Int8";
    case ScalarType::UInt8:
      return "UInt8";
    case ScalarType::BFloat16:
      return "BFloat16";
    case ScalarType::Float16:
      return "Float16";
    default:
      return "Undefined";
  }
}

} // namespace c10
