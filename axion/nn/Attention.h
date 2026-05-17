#pragma once

// ============================================================================
// Axion / nn — Causal Self-Attention (GPT-2 style)
// ============================================================================
//
// Multi-head causal (masked) self-attention.
//
// Input:  x of shape (seq_len, d_model)
// Output: (seq_len, d_model)
//
// Architecture:
//   Q, K, V = Linear(x) each (seq_len, d_model)
//   Split into n_heads of (seq_len, d_head)
//   For each head:
//     attn = softmax(Q @ K^T / sqrt(d_head) + causal_mask)
//     attn = dropout(attn)
//     head_out = attn @ V
//   Concat heads -> Linear -> output
//
// This implementation processes heads sequentially to stay within
// 2D tensor constraints. For batch support, call forward per sample.

#include <cmath>
#include <cassert>
#include <cstring>
#include <vector>
#include <memory>

#include "axion/nn/Module.h"
#include "axion/nn/Linear.h"
#include "axion/nn/Dropout.h"
#include "axion/nn/Softmax.h"
#include "autograd/Variable.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace nn {

class CausalSelfAttention : public Module {
 public:
  CausalSelfAttention(int64_t d_model, int64_t n_heads,
                       float dropout_p = 0.0f)
      : d_model_(d_model),
        n_heads_(n_heads),
        d_head_(d_model / n_heads) {
    assert(d_model % n_heads == 0 &&
           "d_model must be divisible by n_heads");

    // Q, K, V projections (combined into one big linear for efficiency)
    qkv_proj_ = register_module("qkv_proj",
        std::make_shared<Linear>(d_model, 3 * d_model, false));

    // Output projection
    out_proj_ = register_module("out_proj",
        std::make_shared<Linear>(d_model, d_model, false));

    // Attention dropout
    attn_dropout_ = std::make_shared<Dropout>(dropout_p);
  }

  /// Forward pass.
  /// Input: (seq_len, d_model)
  /// Output: (seq_len, d_model)
  c10::Tensor forward(const c10::Tensor& input) override {
    assert(input.defined() && "CausalSelfAttention: input undefined");
    assert(input.ndim() == 2 && "CausalSelfAttention: input must be 2D");
    assert(input.size(1) == d_model_ &&
           "CausalSelfAttention: feature dim != d_model");

    int64_t seq_len = input.size(0);

    // 1. Project to Q, K, V  (seq_len, 3*d_model)
    auto qkv = qkv_proj_->forward(input);

    // 2. Split Q, K, V  (each seq_len, d_model)
    auto Q = slice_cols(qkv, 0, d_model_);
    auto K = slice_cols(qkv, d_model_, 2 * d_model_);
    auto V = slice_cols(qkv, 2 * d_model_, 3 * d_model_);

    // 3. Multi-head attention (process heads sequentially)
    auto concat = c10::Tensor::empty(
        {seq_len, d_model_}, c10::ScalarType::Float32);
    float* cp = concat.data_ptr<float>();

    for (int64_t h = 0; h < n_heads_; ++h) {
      int64_t offset = h * d_head_;

      // Extract head slices: (seq_len, d_head)
      auto Qh = slice_cols(Q, offset, offset + d_head_);
      auto Kh = slice_cols(K, offset, offset + d_head_);
      auto Vh = slice_cols(V, offset, offset + d_head_);

      // scores = Qh @ Kh^T / sqrt(d_head)  -> (seq_len, seq_len)
      auto Kt = aten::contiguous(aten::transpose(Kh, 0, 1));
      auto scores = autograd::matmul(Qh, Kt);
      float scale = 1.0f / std::sqrt(static_cast<float>(d_head_));
      scores = autograd::mul_scalar(scores, scale);

      // Apply causal mask: set future positions to -inf
      apply_causal_mask(scores, seq_len);

      // Softmax over last dim (columns)
      auto attn_weights = nn::functional::softmax(scores);

      // Dropout on attention weights
      if (is_training()) {
        attn_weights = attn_dropout_->forward(attn_weights);
      }

      // Weighted sum: attn_weights @ Vh -> (seq_len, d_head)
      auto head_out = autograd::matmul(attn_weights, Vh);

      // Copy into concat buffer
      const float* hp = head_out.data_ptr<float>();
      for (int64_t s = 0; s < seq_len; ++s) {
        std::memcpy(cp + s * d_model_ + offset,
                    hp + s * d_head_,
                    d_head_ * sizeof(float));
      }
    }

    // 4. Output projection
    // For the concatenated output to flow through autograd, we need to
    // connect it. Since we manually assembled it from head_out tensors,
    // we set requires_grad so the output projection records.
    autograd::set_requires_grad(concat, true);

    return out_proj_->forward(concat);
  }

 private:
  int64_t d_model_;
  int64_t n_heads_;
  int64_t d_head_;

  std::shared_ptr<Linear> qkv_proj_;
  std::shared_ptr<Linear> out_proj_;
  std::shared_ptr<Dropout> attn_dropout_;

  /// Slice columns [col_start, col_end) from a 2D tensor.
  /// Returns a NEW contiguous tensor (not a view).
  static c10::Tensor slice_cols(const c10::Tensor& t,
                                 int64_t col_start, int64_t col_end) {
    int64_t rows = t.size(0);
    int64_t total_cols = t.size(1);
    int64_t out_cols = col_end - col_start;

    auto result = c10::Tensor::empty({rows, out_cols}, t.dtype());
    const float* sp = t.data_ptr<float>();
    float* dp = result.data_ptr<float>();

    for (int64_t i = 0; i < rows; ++i) {
      std::memcpy(dp + i * out_cols,
                  sp + i * total_cols + col_start,
                  out_cols * sizeof(float));
    }

    // Propagate requires_grad
    if (t.requires_grad()) {
      autograd::set_requires_grad(result, true);
    }

    return result;
  }

  /// Apply causal mask: set scores[i][j] = -1e9 where j > i
  static void apply_causal_mask(c10::Tensor& scores, int64_t seq_len) {
    float* sp = scores.data_ptr<float>();
    for (int64_t i = 0; i < seq_len; ++i) {
      for (int64_t j = i + 1; j < seq_len; ++j) {
        sp[i * seq_len + j] = -1e9f;
      }
    }
  }
};

}  // namespace nn
