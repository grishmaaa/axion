// ============================================================================
// Axion / nn — Linear layer implementation
// ============================================================================

#include "axion/nn/Linear.h"

#include "autograd/Variable.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/functions/ReduceOps.h"
#include "aten/ops/Ops.h"

namespace nn {

namespace {

/// Backward for y = input @ weight^T
/// where weight is (out, in) and input is (batch, in).
///
/// dy/d(input) = grad @ weight        (grad is batch,out; weight is out,in)
/// dy/d(weight) = grad^T @ input      (grad^T is out,batch; input is batch,in)
struct LinearBackward : autograd::Node {
  c10::Tensor input_, weight_;

  LinearBackward(c10::Tensor input, c10::Tensor weight)
      : input_(std::move(input)), weight_(std::move(weight)) {}

  std::vector<c10::Tensor> apply(
      std::vector<c10::Tensor> grads) override {
    // grad is (batch, out)
    auto& grad = grads[0];

    // d_input = grad @ weight  (batch,out) @ (out,in) = (batch,in)
    auto d_input = aten::matmul(grad, weight_);

    // d_weight = grad^T @ input  (out,batch) @ (batch,in) = (out,in)
    auto grad_t = aten::contiguous(aten::transpose(grad, 0, 1));
    auto d_weight = aten::matmul(grad_t, input_);

    return {d_input, d_weight};
  }
};

}  // namespace

Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias) {
  params_.reserve(bias ? 2 : 1);

  // weight: (out_features, in_features)
  auto w = c10::Tensor::empty({out_features, in_features},
                               c10::ScalarType::Float32);
  init::kaiming_uniform_(w);
  register_parameter("weight", Parameter(std::move(w)));

  if (has_bias_) {
    auto b = c10::Tensor::empty({1, out_features}, c10::ScalarType::Float32);
    init::zeros_(b);
    register_parameter("bias", Parameter(std::move(b)));
  }
}

c10::Tensor Linear::forward(const c10::Tensor& input) {
  assert(input.defined() && "Linear::forward: input undefined");
  assert(input.ndim() == 2 &&
         "Linear::forward: input must be 2D (batch, features)");
  assert(input.size(1) == in_features_ &&
         "Linear::forward: input features don't match");

  // Compute y = input @ weight^T  (using aten:: for the actual compute)
  auto wt = aten::contiguous(aten::transpose(weight().data(), 0, 1));
  auto y = aten::matmul(input, wt);

  // Record on autograd graph — connect gradient flow to the ORIGINAL
  // weight parameter, not the transposed copy.
  if (autograd::GradMode::is_enabled()) {
    auto node = std::make_shared<LinearBackward>(input, weight().data());
    node->add_next_edge(autograd::gradient_edge(input));
    node->add_next_edge(autograd::gradient_edge(weight().data()));
    autograd::set_grad_fn(y, node);
  }

  if (has_bias_) {
    // bias is (1, out_features). Expand to (batch, out_features).
    int64_t batch = y.size(0);
    int64_t out = y.size(1);

    auto bias_expanded = c10::Tensor::empty({batch, out}, y.dtype());
    const float* bp = bias().data().data_ptr<float>();
    float* ep = bias_expanded.data_ptr<float>();
    for (int64_t i = 0; i < batch; ++i) {
      for (int64_t j = 0; j < out; ++j) {
        ep[i * out + j] = bp[j];
      }
    }
    // Connect bias to graph
    autograd::set_requires_grad(bias_expanded, true);
    y = autograd::add(y, bias_expanded);
  }

  return y;
}

}  // namespace nn
