#pragma once

// ============================================================================
// Axion / nn — Sequential container
// ============================================================================
//
// Chains modules in order. forward() runs input through each module
// sequentially.

#include <initializer_list>
#include <memory>
#include <string>

#include "axion/nn/Module.h"

namespace nn {

class Sequential : public Module {
 public:
  Sequential() = default;

  /// Variadic constructor: Sequential(make_shared<Linear>(2,4), ...)
  Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    for (auto& m : modules) {
      add(m);
    }
  }

  /// Add a module to the end of the sequence.
  void add(std::shared_ptr<Module> module) {
    std::string name = std::to_string(modules_.size());
    modules_.push_back(register_module(name, std::move(module)));
  }

  c10::Tensor forward(const c10::Tensor& input) override {
    c10::Tensor x = input;
    for (auto& m : modules_) {
      x = m->forward(x);
    }
    return x;
  }

  size_t size() const { return modules_.size(); }

 private:
  std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace nn
