// ============================================================================
// Axion / autograd / engine — Backward Engine implementation
// ============================================================================
//
// ALGORITHM (Kahn's / BFS topological traversal):
//
//   1. Walk the graph from root to count dependencies (in-degree) of
//      each node — how many edges point TO it from other nodes.
//
//   2. Root has 0 dependencies (we feed it the initial gradient).
//
//   3. BFS: dequeue a node, call apply() with accumulated gradients,
//      distribute output gradients to next_edges, decrement their
//      dependency counts. When a node reaches 0, enqueue it.
//
//   4. AccumulateGrad nodes (leaves) store the gradient on the tensor.

#include "autograd/engine/Engine.h"

#include <cassert>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace autograd {

namespace {

/// Walk the graph from root and count how many edges point to each node.
void compute_dependencies(
    Node* root,
    std::unordered_map<Node*, int>& deps) {
  std::queue<Node*> q;
  std::unordered_set<Node*> visited;

  q.push(root);
  visited.insert(root);
  deps[root] = 0;  // root is immediately ready

  while (!q.empty()) {
    Node* node = q.front();
    q.pop();

    for (const auto& edge : node->next_edges()) {
      if (!edge.function) continue;
      Node* next = edge.function.get();

      deps[next]++;  // one more edge pointing to this node

      if (visited.find(next) == visited.end()) {
        visited.insert(next);
        q.push(next);
      }
    }
  }
}

}  // namespace

void Engine::backward(
    const c10::Tensor& root,
    const c10::Tensor& grad_output) {
  assert(root.defined() && "backward: root tensor is undefined");
  assert(root.numel() == 1 && "backward: root must be a scalar tensor");

  auto* meta = get_autograd_meta(root);
  assert(meta && "backward: root has no autograd metadata");
  assert(meta->grad_fn_ && "backward: root has no grad_fn (is it a leaf?)");

  auto root_fn = meta->grad_fn_;

  // Disable gradient recording during backward to prevent
  // the backward ops from building a second graph.
  NoGradGuard no_grad;

  // 1. Compute dependencies
  std::unordered_map<Node*, int> deps;
  compute_dependencies(root_fn.get(), deps);

  // 2. Initialize gradients
  c10::Tensor initial_grad = grad_output;
  if (!initial_grad.defined()) {
    initial_grad = aten::ones({}, root.dtype());
  }

  // node_grads[node] = accumulated input gradients for that node
  std::unordered_map<Node*, std::vector<c10::Tensor>> node_grads;
  node_grads[root_fn.get()].push_back(initial_grad);

  // 3. BFS backward traversal
  std::queue<Node*> ready;
  ready.push(root_fn.get());

  while (!ready.empty()) {
    Node* node = ready.front();
    ready.pop();

    // Call the backward function
    auto outputs = node->apply(node_grads[node]);

    // Distribute output gradients to next_edges
    const auto& edges = node->next_edges();
    for (size_t i = 0; i < outputs.size() && i < edges.size(); ++i) {
      const auto& edge = edges[i];
      if (!edge.function || !outputs[i].defined()) continue;

      Node* next = edge.function.get();

      // Accumulate gradient for this input_nr
      auto& next_grads = node_grads[next];
      if (next_grads.size() <= edge.input_nr) {
        next_grads.resize(edge.input_nr + 1);
      }

      if (next_grads[edge.input_nr].defined()) {
        next_grads[edge.input_nr] =
            aten::add(next_grads[edge.input_nr], outputs[i]);
      } else {
        next_grads[edge.input_nr] = outputs[i];
      }

      // Decrement dependency and enqueue if ready
      deps[next]--;
      if (deps[next] == 0) {
        ready.push(next);
      }
    }

    // Free processed gradients to save memory
    node_grads.erase(node);
  }
}

}  // namespace autograd
