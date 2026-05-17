#pragma once

// ============================================================================
// Axion / nn — Module (base class for all neural network layers)
// ============================================================================
//
// Provides:
//   - Parameter registration and traversal
//   - Submodule registration
//   - Training/eval mode
//   - zero_grad() for all parameters
//
// Subclasses implement forward() with their computation logic.

#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "c10/core/Tensor.h"
#include "axion/nn/Parameter.h"

namespace nn {

class Module {
 public:
  virtual ~Module() = default;

  /// Forward pass. Subclasses override this.
  virtual c10::Tensor forward(const c10::Tensor& input) = 0;

  // ------------------------------------------------------------------
  // Parameter management
  // ------------------------------------------------------------------

  /// Register a named parameter. Returns a reference for storage.
  Parameter& register_parameter(const std::string& name, Parameter param) {
    params_.emplace_back(name, std::move(param));
    return params_.back().second;
  }

  /// All parameters in this module (non-recursive).
  std::vector<Parameter*> parameters() {
    std::vector<Parameter*> result;
    for (auto& [name, param] : params_) {
      result.push_back(&param);
    }
    // Recurse into children
    for (auto& [name, child] : children_) {
      auto child_params = child->parameters();
      result.insert(result.end(), child_params.begin(), child_params.end());
    }
    return result;
  }

  // ------------------------------------------------------------------
  // Submodule management
  // ------------------------------------------------------------------

  /// Register a named submodule.
  template <typename T>
  std::shared_ptr<T> register_module(
      const std::string& name, std::shared_ptr<T> module) {
    children_.emplace_back(name, module);
    return module;
  }

  // ------------------------------------------------------------------
  // Training mode
  // ------------------------------------------------------------------

  void train(bool mode = true) {
    training_ = mode;
    for (auto& [name, child] : children_) {
      child->train(mode);
    }
  }

  void eval() { train(false); }

  bool is_training() const { return training_; }

  // ------------------------------------------------------------------
  // Gradient utilities
  // ------------------------------------------------------------------

  /// Zero all gradients in this module and children.
  void zero_grad() {
    for (auto* p : parameters()) {
      p->zero_grad();
    }
  }

 protected:
  std::vector<std::pair<std::string, Parameter>> params_;
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
  bool training_ = true;
};

}  // namespace nn
