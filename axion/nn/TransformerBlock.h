#pragma once

// ============================================================================
// Axion / nn — Transformer Block (GPT-2 style)
// ============================================================================
//
// Pre-norm Transformer block (GPT-2 architecture):
//
//   x = x + Attention(LayerNorm(x))
//   x = x + FFN(LayerNorm(x))
//
// Where FFN = Linear(d_model, 4*d_model) -> GELU -> Linear(4*d_model, d_model)
//
// Input/Output: (seq_len, d_model)

#include "axion/nn/Module.h"
#include "axion/nn/Linear.h"
#include "axion/nn/LayerNorm.h"
#include "axion/nn/Attention.h"
#include "axion/nn/Dropout.h"
#include "axion/nn/Activations.h"
#include "autograd/Variable.h"
#include "aten/ops/Ops.h"

namespace nn {

class TransformerBlock : public Module {
 public:
  TransformerBlock(int64_t d_model, int64_t n_heads,
                    float dropout_p = 0.0f)
      : d_model_(d_model) {
    // Pre-norm for attention
    ln1_ = register_module("ln1",
        std::make_shared<LayerNorm>(d_model));

    // Self-attention
    attn_ = register_module("attn",
        std::make_shared<CausalSelfAttention>(d_model, n_heads, dropout_p));

    // Pre-norm for FFN
    ln2_ = register_module("ln2",
        std::make_shared<LayerNorm>(d_model));

    // FFN: d_model -> 4*d_model -> d_model
    ffn1_ = register_module("ffn1",
        std::make_shared<Linear>(d_model, 4 * d_model, false));

    ffn2_ = register_module("ffn2",
        std::make_shared<Linear>(4 * d_model, d_model, false));

    // Residual dropout
    resid_dropout_ = std::make_shared<Dropout>(dropout_p);
  }

  /// Forward pass.
  /// Input: (seq_len, d_model)
  /// Output: (seq_len, d_model)
  c10::Tensor forward(const c10::Tensor& input) override {
    assert(input.defined() && "TransformerBlock: input undefined");
    assert(input.ndim() == 2 && "TransformerBlock: input must be 2D");
    assert(input.size(1) == d_model_ &&
           "TransformerBlock: feature dim != d_model");

    // --- Attention sub-layer with residual ---
    // x = x + dropout(attn(ln1(x)))
    auto normed1 = ln1_->forward(input);
    auto attn_out = attn_->forward(normed1);
    if (is_training()) {
      attn_out = resid_dropout_->forward(attn_out);
    }
    auto x = autograd::add(input, attn_out);

    // --- FFN sub-layer with residual ---
    // x = x + dropout(ffn(ln2(x)))
    auto normed2 = ln2_->forward(x);
    auto ffn_out = ffn1_->forward(normed2);
    ffn_out = autograd::gelu(ffn_out);
    ffn_out = ffn2_->forward(ffn_out);
    if (is_training()) {
      ffn_out = resid_dropout_->forward(ffn_out);
    }
    x = autograd::add(x, ffn_out);

    return x;
  }

 private:
  int64_t d_model_;

  std::shared_ptr<LayerNorm> ln1_;
  std::shared_ptr<CausalSelfAttention> attn_;
  std::shared_ptr<LayerNorm> ln2_;
  std::shared_ptr<Linear> ffn1_;
  std::shared_ptr<Linear> ffn2_;
  std::shared_ptr<Dropout> resid_dropout_;
};

}  // namespace nn
