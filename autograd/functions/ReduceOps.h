#pragma once

// ============================================================================
// Axion / autograd / functions — Reduce ops backward nodes
// ============================================================================

#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace autograd {

struct SumBackward : Node {
  std::vector<int64_t> input_sizes_;
  c10::ScalarType dtype_;

  SumBackward(std::vector<int64_t> sizes, c10::ScalarType dt)
      : input_sizes_(std::move(sizes)), dtype_(dt) {}

  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // da = grad * ones(input_shape)  — expand scalar grad to input shape
    return {aten::full(input_sizes_, g[0].data_ptr<float>()[0], dtype_)};
  }
};

struct MeanBackward : Node {
  std::vector<int64_t> input_sizes_;
  int64_t numel_;
  c10::ScalarType dtype_;

  MeanBackward(std::vector<int64_t> sizes, int64_t numel, c10::ScalarType dt)
      : input_sizes_(std::move(sizes)), numel_(numel), dtype_(dt) {}

  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // da = grad / numel * ones(input_shape)
    float grad_val = g[0].data_ptr<float>()[0];
    return {aten::full(input_sizes_, grad_val / static_cast<float>(numel_),
                       dtype_)};
  }
};

struct MatmulBackward : Node {
  c10::Tensor a_, b_;

  MatmulBackward(c10::Tensor a, c10::Tensor b)
      : a_(std::move(a)), b_(std::move(b)) {}

  std::vector<c10::Tensor> apply(std::vector<c10::Tensor> g) override {
    // C = A @ B
    // dA = grad @ B^T
    // dB = A^T @ grad
    auto bt = aten::transpose(b_, 0, 1);
    auto at = aten::transpose(a_, 0, 1);

    // contiguous() needed because transpose creates non-contiguous views
    auto bt_c = aten::contiguous(bt);
    auto at_c = aten::contiguous(at);

    return {aten::matmul(g[0], bt_c), aten::matmul(at_c, g[0])};
  }
};

}  // namespace autograd
