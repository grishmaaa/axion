// ============================================================================
// Axion / autograd — Variable (autograd-aware operations) implementation
// ============================================================================
//
// Pattern for each op:
//   1. Execute the aten:: forward op
//   2. If GradMode enabled AND any input requires_grad:
//      a. Create backward node, save needed tensors
//      b. Set up edges from the node to input gradient sources
//      c. Set grad_fn on the output tensor
//   3. Return the result

#include "autograd/Variable.h"

#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/functions/BasicOps.h"
#include "autograd/functions/UnaryOps.h"
#include "autograd/functions/ReduceOps.h"
#include "aten/ops/Ops.h"

namespace autograd {

namespace {

/// Check if we should record this op on the graph.
inline bool should_record(const c10::Tensor& a) {
  return GradMode::is_enabled() && a.requires_grad();
}

inline bool should_record(const c10::Tensor& a, const c10::Tensor& b) {
  return GradMode::is_enabled() &&
         (a.requires_grad() || b.requires_grad());
}

}  // namespace

// ============================================================================
// Binary ops
// ============================================================================

c10::Tensor add(const c10::Tensor& a, const c10::Tensor& b) {
  auto result = aten::add(a, b);
  if (should_record(a, b)) {
    auto node = std::make_shared<AddBackward>();
    node->add_next_edge(gradient_edge(a));
    node->add_next_edge(gradient_edge(b));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor sub(const c10::Tensor& a, const c10::Tensor& b) {
  auto result = aten::sub(a, b);
  if (should_record(a, b)) {
    auto node = std::make_shared<SubBackward>();
    node->add_next_edge(gradient_edge(a));
    node->add_next_edge(gradient_edge(b));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor mul(const c10::Tensor& a, const c10::Tensor& b) {
  auto result = aten::mul(a, b);
  if (should_record(a, b)) {
    auto node = std::make_shared<MulBackward>(a, b);
    node->add_next_edge(gradient_edge(a));
    node->add_next_edge(gradient_edge(b));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor div(const c10::Tensor& a, const c10::Tensor& b) {
  auto result = aten::div(a, b);
  if (should_record(a, b)) {
    auto node = std::make_shared<DivBackward>(a, b);
    node->add_next_edge(gradient_edge(a));
    node->add_next_edge(gradient_edge(b));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor pow(const c10::Tensor& a, const c10::Tensor& b) {
  auto result = aten::pow(a, b);
  if (should_record(a, b)) {
    auto node = std::make_shared<PowBackward>(a, b);
    node->add_next_edge(gradient_edge(a));
    node->add_next_edge(gradient_edge(b));
    set_grad_fn(result, node);
  }
  return result;
}

// ============================================================================
// Unary ops
// ============================================================================

c10::Tensor neg(const c10::Tensor& a) {
  auto result = aten::neg(a);
  if (should_record(a)) {
    auto node = std::make_shared<NegBackward>();
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor relu(const c10::Tensor& a) {
  auto result = aten::relu(a);
  if (should_record(a)) {
    auto node = std::make_shared<ReluBackward>(a);
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor abs(const c10::Tensor& a) {
  auto result = aten::abs(a);
  if (should_record(a)) {
    auto node = std::make_shared<AbsBackward>(a);
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor exp(const c10::Tensor& a) {
  auto result = aten::exp(a);
  if (should_record(a)) {
    auto node = std::make_shared<ExpBackward>(result);  // save output
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor log(const c10::Tensor& a) {
  auto result = aten::log(a);
  if (should_record(a)) {
    auto node = std::make_shared<LogBackward>(a);  // save input
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor sqrt(const c10::Tensor& a) {
  auto result = aten::sqrt(a);
  if (should_record(a)) {
    auto node = std::make_shared<SqrtBackward>(result);  // save output
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor tanh(const c10::Tensor& a) {
  auto result = aten::tanh(a);
  if (should_record(a)) {
    auto node = std::make_shared<TanhBackward>(result);  // save output
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor sigmoid(const c10::Tensor& a) {
  auto result = aten::sigmoid(a);
  if (should_record(a)) {
    auto node = std::make_shared<SigmoidBackward>(result);  // save output
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor gelu(const c10::Tensor& a) {
  auto result = aten::gelu(a);
  if (should_record(a)) {
    auto node = std::make_shared<GeluBackward>(a);  // save input
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

// ============================================================================
// Scalar ops
// ============================================================================

c10::Tensor add_scalar(const c10::Tensor& a, double s) {
  auto result = aten::add_scalar(a, s);
  if (should_record(a)) {
    auto node = std::make_shared<PassthroughBackward>();
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor sub_scalar(const c10::Tensor& a, double s) {
  auto result = aten::sub_scalar(a, s);
  if (should_record(a)) {
    auto node = std::make_shared<PassthroughBackward>();
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor mul_scalar(const c10::Tensor& a, double s) {
  auto result = aten::mul_scalar(a, s);
  if (should_record(a)) {
    auto node = std::make_shared<MulScalarBackward>(s);
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor div_scalar(const c10::Tensor& a, double s) {
  auto result = aten::div_scalar(a, s);
  if (should_record(a)) {
    auto node = std::make_shared<DivScalarBackward>(s);
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

// ============================================================================
// Reduce ops
// ============================================================================

c10::Tensor sum(const c10::Tensor& a) {
  auto result = aten::sum(a);
  if (should_record(a)) {
    auto node = std::make_shared<SumBackward>(
        std::vector<int64_t>(a.sizes().begin(), a.sizes().end()),
        a.dtype());
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

c10::Tensor mean(const c10::Tensor& a) {
  auto result = aten::mean(a);
  if (should_record(a)) {
    auto node = std::make_shared<MeanBackward>(
        std::vector<int64_t>(a.sizes().begin(), a.sizes().end()),
        a.numel(), a.dtype());
    node->add_next_edge(gradient_edge(a));
    set_grad_fn(result, node);
  }
  return result;
}

// ============================================================================
// Matmul
// ============================================================================

c10::Tensor matmul(const c10::Tensor& a, const c10::Tensor& b) {
  auto result = aten::matmul(a, b);
  if (should_record(a, b)) {
    auto node = std::make_shared<MatmulBackward>(a, b);
    node->add_next_edge(gradient_edge(a));
    node->add_next_edge(gradient_edge(b));
    set_grad_fn(result, node);
  }
  return result;
}

}  // namespace autograd
