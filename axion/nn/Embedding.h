#pragma once

// ============================================================================
// Axion / nn — Embedding layer
// ============================================================================
//
// Lookup table: given integer indices, return corresponding embedding vectors.
//
//   output[i] = weight[indices[i]]
//
// Parameters:
//   weight: (num_embeddings, embedding_dim)
//
// Input:
//   indices: (batch, seq_len) — integer indices stored as float
//
// Output:
//   (batch, seq_len, embedding_dim)

#include "axion/nn/Module.h"
#include "axion/nn/init.h"
#include "autograd/AutogradMeta.h"
#include "autograd/GradMode.h"
#include "autograd/Node.h"
#include "aten/ops/Ops.h"

namespace nn {

namespace {

/// Backward: scatter gradient back to embedding rows.
/// grad_output is (batch, seq_len, embed_dim)
/// We accumulate into d_weight (num_embed, embed_dim).
struct EmbeddingBackward : autograd::Node {
  c10::Tensor indices_;   // (batch, seq_len) stored as float
  int64_t num_embeddings_;
  int64_t embedding_dim_;

  EmbeddingBackward(c10::Tensor indices, int64_t num_embed, int64_t embed_dim)
      : indices_(std::move(indices)),
        num_embeddings_(num_embed),
        embedding_dim_(embed_dim) {}

  std::vector<c10::Tensor> apply(
      std::vector<c10::Tensor> grads) override {
    auto& grad = grads[0];  // (batch, seq_len, embed_dim)

    auto d_weight = aten::zeros(
        {num_embeddings_, embedding_dim_}, grad.dtype());

    const float* gp = grad.data_ptr<float>();
    const float* ip = indices_.data_ptr<float>();
    float* wp = d_weight.data_ptr<float>();

    int64_t batch = indices_.size(0);
    int64_t seq_len = indices_.size(1);

    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        int64_t idx = static_cast<int64_t>(ip[b * seq_len + s]);
        // Accumulate gradient into the correct embedding row
        for (int64_t d = 0; d < embedding_dim_; ++d) {
          wp[idx * embedding_dim_ + d] +=
              gp[(b * seq_len + s) * embedding_dim_ + d];
        }
      }
    }

    return {d_weight};
  }
};

}  // namespace

class Embedding : public Module {
 public:
  Embedding(int64_t num_embeddings, int64_t embedding_dim)
      : num_embeddings_(num_embeddings),
        embedding_dim_(embedding_dim) {
    auto w = c10::Tensor::empty(
        {num_embeddings, embedding_dim}, c10::ScalarType::Float32);
    init::normal_(w, 0.0, 1.0);
    register_parameter("weight", Parameter(std::move(w)));
  }

  /// Forward: indices is (batch, seq_len) with float-encoded integer indices.
  /// Returns: (batch, seq_len, embedding_dim)
  c10::Tensor forward(const c10::Tensor& indices) override {
    assert(indices.defined() && "Embedding: indices undefined");
    assert(indices.ndim() == 2 && "Embedding: indices must be 2D (batch, seq)");

    int64_t batch = indices.size(0);
    int64_t seq_len = indices.size(1);

    auto output = c10::Tensor::empty(
        {batch * seq_len, embedding_dim_}, c10::ScalarType::Float32);

    const float* ip = indices.data_ptr<float>();
    const float* wp = weight().data().data_ptr<float>();
    float* op = output.data_ptr<float>();

    for (int64_t i = 0; i < batch * seq_len; ++i) {
      int64_t idx = static_cast<int64_t>(ip[i]);
      assert(idx >= 0 && idx < num_embeddings_ && "Embedding: index OOB");
      // Copy embedding row
      for (int64_t d = 0; d < embedding_dim_; ++d) {
        op[i * embedding_dim_ + d] = wp[idx * embedding_dim_ + d];
      }
    }

    // Reshape to (batch, seq_len, embedding_dim) — manual reshape
    // since we don't have 3D support yet, keep as (batch*seq_len, embed_dim)
    // and let callers handle the reshape. But for correctness, we store
    // the 3D view.
    auto result = c10::Tensor::empty(
        {batch, seq_len * embedding_dim_}, c10::ScalarType::Float32);
    std::memcpy(result.data_ptr<float>(), op,
                batch * seq_len * embedding_dim_ * sizeof(float));

    // Record on autograd graph
    if (autograd::GradMode::is_enabled()) {
      auto node = std::make_shared<EmbeddingBackward>(
          indices, num_embeddings_, embedding_dim_);
      node->add_next_edge(autograd::gradient_edge(weight().data()));
      autograd::set_grad_fn(result, node);
    }

    return result;
  }

  int64_t num_embeddings() const { return num_embeddings_; }
  int64_t embedding_dim() const { return embedding_dim_; }

 private:
  int64_t num_embeddings_;
  int64_t embedding_dim_;

  Parameter& weight() { return params_[0].second; }
};

}  // namespace nn
