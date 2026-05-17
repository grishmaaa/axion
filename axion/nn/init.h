#pragma once

// ============================================================================
// Axion / nn — Weight initialization
// ============================================================================

#include <cmath>
#include <random>
#include "c10/core/Tensor.h"

namespace nn {
namespace init {

/// Fill tensor with uniform random values in [low, high).
inline void uniform_(c10::Tensor& t, double low, double high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(
      static_cast<float>(low), static_cast<float>(high));
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(gen);
}

/// Fill tensor with normal random values N(mean, std).
inline void normal_(c10::Tensor& t, double mean = 0.0, double std = 1.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(
      static_cast<float>(mean), static_cast<float>(std));
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(gen);
}

/// Kaiming uniform initialization (He et al.)
/// For layers with ReLU activation.
inline void kaiming_uniform_(c10::Tensor& t, double a = 0.0) {
  // fan_in = product of all dims except first
  auto sizes = t.sizes();
  int64_t fan_in = 1;
  for (size_t i = 1; i < sizes.size(); ++i) fan_in *= sizes[i];

  double gain = std::sqrt(2.0 / (1.0 + a * a));
  double bound = gain * std::sqrt(3.0 / static_cast<double>(fan_in));
  uniform_(t, -bound, bound);
}

/// Xavier uniform initialization (Glorot).
/// For layers with linear/tanh/sigmoid activation.
inline void xavier_uniform_(c10::Tensor& t) {
  auto sizes = t.sizes();
  int64_t fan_in = sizes.size() > 1 ? sizes[1] : sizes[0];
  int64_t fan_out = sizes[0];
  double bound = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
  uniform_(t, -bound, bound);
}

/// Fill with zeros.
inline void zeros_(c10::Tensor& t) {
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = 0.0f;
}

}  // namespace init
}  // namespace nn
