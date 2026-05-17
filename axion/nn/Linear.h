#pragma once

// ============================================================================
// Axion / nn — Linear layer
// ============================================================================
//
// y = x @ W^T + b
//
// Parameters:
//   weight: (out_features, in_features)
//   bias:   (1, out_features) — optional

#include "axion/nn/Module.h"
#include "axion/nn/init.h"

namespace nn {

class Linear : public Module {
 public:
  Linear(int64_t in_features, int64_t out_features, bool bias = true);

  c10::Tensor forward(const c10::Tensor& input) override;

  int64_t in_features() const { return in_features_; }
  int64_t out_features() const { return out_features_; }

 private:
  int64_t in_features_;
  int64_t out_features_;
  bool has_bias_;

  /// Safe accessors — look up by index, avoid dangling pointers
  /// from vector reallocation.
  Parameter& weight() { return params_[0].second; }
  Parameter& bias()   { return params_[1].second; }
};

}  // namespace nn
