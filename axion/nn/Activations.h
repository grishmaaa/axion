#pragma once

// ============================================================================
// Axion / nn — Functional modules (ReLU, GELU, etc.)
// ============================================================================
//
// Wrapper modules for activation functions so they can be used in
// Sequential containers.

#include "axion/nn/Module.h"
#include "autograd/Variable.h"

namespace nn {

class ReLU : public Module {
 public:
  c10::Tensor forward(const c10::Tensor& input) override {
    return autograd::relu(input);
  }
};

class GELU : public Module {
 public:
  c10::Tensor forward(const c10::Tensor& input) override {
    return autograd::gelu(input);
  }
};

class Sigmoid : public Module {
 public:
  c10::Tensor forward(const c10::Tensor& input) override {
    return autograd::sigmoid(input);
  }
};

class Tanh : public Module {
 public:
  c10::Tensor forward(const c10::Tensor& input) override {
    return autograd::tanh(input);
  }
};

}  // namespace nn
