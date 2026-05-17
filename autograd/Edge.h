#pragma once

// ============================================================================
// Axion / autograd — Edge
// ============================================================================
//
// An Edge connects two nodes in the computation graph.
// It says: "gradient should flow into input #input_nr of function."

#include <cstdint>
#include <memory>

namespace autograd {

class Node;  // forward declaration

/// An edge in the computation graph.
struct Edge {
  std::shared_ptr<Node> function;
  uint32_t input_nr = 0;

  Edge() = default;
  Edge(std::shared_ptr<Node> fn, uint32_t nr)
      : function(std::move(fn)), input_nr(nr) {}

  /// True if this edge points to a valid node.
  explicit operator bool() const { return static_cast<bool>(function); }
};

}  // namespace autograd
