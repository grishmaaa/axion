#pragma once

// ============================================================================
// Axion / autograd — Node (base class for backward functions)
// ============================================================================
//
// Every differentiable operation creates a Node subclass that knows how
// to compute the backward pass (local Jacobian-vector product).
//
// The computation graph is a DAG of Nodes connected by Edges.
// Leaf tensors (parameters) terminate at AccumulateGrad nodes.

#include <vector>
#include <memory>

#include "autograd/Edge.h"
#include "c10/core/Tensor.h"

namespace autograd {

class Node : public std::enable_shared_from_this<Node> {
 public:
  virtual ~Node() = default;

  /// Compute input gradients from output gradients.
  /// grads[i] is the gradient of the loss wrt output i of this op.
  /// Returns gradients for each input (one per next_edge).
  virtual std::vector<c10::Tensor> apply(
      std::vector<c10::Tensor> grads) = 0;

  // ------------------------------------------------------------------
  // Graph connectivity
  // ------------------------------------------------------------------

  void set_next_edges(std::vector<Edge> edges) {
    next_edges_ = std::move(edges);
  }

  void add_next_edge(Edge edge) {
    next_edges_.push_back(std::move(edge));
  }

  const std::vector<Edge>& next_edges() const { return next_edges_; }
  const Edge& next_edge(size_t i) const { return next_edges_[i]; }
  size_t num_inputs() const { return next_edges_.size(); }

 protected:
  /// Edges to predecessor nodes in the backward graph.
  /// next_edges_[i] = where to send the gradient for input i.
  std::vector<Edge> next_edges_;
};

}  // namespace autograd
