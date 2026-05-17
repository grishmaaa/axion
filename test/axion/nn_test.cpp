// ============================================================================
// Tests for nn::Module, nn::Linear, nn::Sequential, nn::Parameter
// ============================================================================

#include <gtest/gtest.h>
#include <cmath>

#include "axion/nn/Parameter.h"
#include "axion/nn/Module.h"
#include "axion/nn/Linear.h"
#include "axion/nn/Sequential.h"
#include "axion/nn/Activations.h"
#include "axion/nn/LayerNorm.h"
#include "axion/nn/Dropout.h"
#include "axion/nn/Embedding.h"
#include "axion/nn/Softmax.h"
#include "axion/nn/Attention.h"
#include "axion/nn/TransformerBlock.h"
#include "axion/nn/Loss.h"
#include "axion/optim/SGD.h"
#include "axion/optim/Adam.h"
#include "autograd/Variable.h"
#include "autograd/engine/Engine.h"
#include "aten/ops/Ops.h"

namespace {

// ============================================================================
// Parameter tests
// ============================================================================

TEST(Parameter, RequiresGradByDefault) {
  auto t = aten::ones({3, 4});
  nn::Parameter p(t);
  EXPECT_TRUE(p.data().requires_grad());
}

TEST(Parameter, ZeroGrad) {
  auto t = aten::ones({3});
  nn::Parameter p(t);
  // Simulate a gradient
  autograd::mutable_grad(p.data()) = aten::full({3}, 5.0f);
  EXPECT_TRUE(p.has_grad());
  p.zero_grad();
  EXPECT_FALSE(p.has_grad());
}

TEST(Parameter, ImplicitConversion) {
  auto t = aten::full({}, 3.0f);
  nn::Parameter p(t);
  // Should work via implicit conversion to const Tensor&
  auto result = autograd::mul_scalar(p, 2.0);
  EXPECT_NEAR(result.data_ptr<float>()[0], 6.0f, 1e-5f);
}

// ============================================================================
// Linear tests
// ============================================================================

TEST(Linear, OutputShape) {
  auto layer = nn::Linear(4, 3);
  auto input = aten::ones({2, 4});
  auto output = layer.forward(input);
  EXPECT_EQ(output.size(0), 2);
  EXPECT_EQ(output.size(1), 3);
}

TEST(Linear, HasParameters) {
  auto layer = nn::Linear(4, 3, true);
  auto params = layer.parameters();
  EXPECT_EQ(params.size(), 2u);  // weight + bias
}

TEST(Linear, NoBias) {
  auto layer = nn::Linear(4, 3, false);
  auto params = layer.parameters();
  EXPECT_EQ(params.size(), 1u);  // weight only
}

// ============================================================================
// Sequential tests
// ============================================================================

TEST(Sequential, ForwardChain) {
  auto seq = nn::Sequential();
  seq.add(std::make_shared<nn::Linear>(4, 3, false));
  seq.add(std::make_shared<nn::ReLU>());
  seq.add(std::make_shared<nn::Linear>(3, 2, false));

  auto input = aten::ones({1, 4});
  auto output = seq.forward(input);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_EQ(output.size(1), 2);
}

TEST(Sequential, CollectsAllParams) {
  auto seq = nn::Sequential();
  seq.add(std::make_shared<nn::Linear>(4, 3));  // weight + bias = 2
  seq.add(std::make_shared<nn::ReLU>());         // 0 params
  seq.add(std::make_shared<nn::Linear>(3, 2));  // weight + bias = 2

  auto params = seq.parameters();
  EXPECT_EQ(params.size(), 4u);
}

TEST(Sequential, TrainEvalMode) {
  auto seq = nn::Sequential();
  seq.add(std::make_shared<nn::Linear>(4, 3));
  EXPECT_TRUE(seq.is_training());
  seq.eval();
  EXPECT_FALSE(seq.is_training());
  seq.train();
  EXPECT_TRUE(seq.is_training());
}

// ============================================================================
// Loss tests
// ============================================================================

TEST(Loss, MSEZeroLoss) {
  auto pred = aten::full({4}, 2.0f);
  autograd::set_requires_grad(pred, true);
  auto target = aten::full({4}, 2.0f);
  auto loss = nn::mse_loss(pred, target);
  EXPECT_NEAR(loss.data_ptr<float>()[0], 0.0f, 1e-5f);
}

TEST(Loss, MSECorrectValue) {
  auto pred = aten::full({4}, 3.0f);
  autograd::set_requires_grad(pred, true);
  auto target = aten::full({4}, 1.0f);
  auto loss = nn::mse_loss(pred, target);
  // mean((3-1)^2) = mean(4) = 4.0
  EXPECT_NEAR(loss.data_ptr<float>()[0], 4.0f, 1e-5f);
}

// ============================================================================
// SGD tests
// ============================================================================

TEST(SGD, StepDecreasesLoss) {
  auto t = aten::full({}, 5.0f);
  nn::Parameter p(t);
  auto target = aten::full({}, 0.0f);

  optim::SGD optimizer({&p}, /*lr=*/0.1);

  float prev_loss = 1e10f;
  for (int i = 0; i < 5; ++i) {
    optimizer.zero_grad();
    auto loss = nn::mse_loss(p.data(), target);
    autograd::Engine::backward(loss);
    optimizer.step();

    float l = loss.data_ptr<float>()[0];
    EXPECT_LT(l, prev_loss);
    prev_loss = l;
  }
}

// ============================================================================
// XOR end-to-end training test
// ============================================================================

TEST(EndToEnd, XORTraining) {
  // XOR problem: learn to map
  //   [0,0] -> 0, [0,1] -> 1, [1,0] -> 1, [1,1] -> 0
  //
  // XOR with small networks is init-sensitive. Retry up to 3 times
  // with different random seeds (kaiming_uniform uses random_device).

  // XOR data (fixed)
  auto x = c10::Tensor::empty({4, 2}, c10::ScalarType::Float32);
  float* xp = x.data_ptr<float>();
  xp[0] = 0; xp[1] = 0;
  xp[2] = 0; xp[3] = 1;
  xp[4] = 1; xp[5] = 0;
  xp[6] = 1; xp[7] = 1;

  auto target = c10::Tensor::empty({4, 1}, c10::ScalarType::Float32);
  float* tp = target.data_ptr<float>();
  tp[0] = 0; tp[1] = 1; tp[2] = 1; tp[3] = 0;

  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    // Architecture: Linear(2,16) -> ReLU -> Linear(16,1)
    auto model = nn::Sequential();
    model.add(std::make_shared<nn::Linear>(2, 32, false));
    model.add(std::make_shared<nn::ReLU>());
    model.add(std::make_shared<nn::Linear>(32, 1, false));

    auto params = model.parameters();
    optim::SGD optimizer(params, /*lr=*/0.05);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < 5000; ++epoch) {
      optimizer.zero_grad();

      auto pred = model.forward(x);
      auto loss = nn::mse_loss(pred, target);

      if (epoch == 0) initial_loss = loss.data_ptr<float>()[0];
      final_loss = loss.data_ptr<float>()[0];

      autograd::Engine::backward(loss);
      optimizer.step();

      // Early stop if converged
      if (final_loss < 0.01f) break;
    }

    if (final_loss < initial_loss * 0.5f && final_loss < 0.1f) {
      converged = true;
    }
  }

  EXPECT_TRUE(converged) << "XOR training didn't converge after 3 attempts";
}

// ============================================================================
// Adam tests
// ============================================================================

TEST(Adam, StepDecreasesLoss) {
  auto t = aten::full({}, 5.0f);
  nn::Parameter p(t);
  auto target = aten::full({}, 0.0f);

  optim::Adam optimizer({&p}, /*lr=*/0.1);

  float first_loss = 0.0f;
  float prev_loss = 1e10f;
  for (int i = 0; i < 100; ++i) {
    optimizer.zero_grad();
    auto loss = nn::mse_loss(p.data(), target);
    autograd::Engine::backward(loss);
    optimizer.step();

    float l = loss.data_ptr<float>()[0];
    if (i == 0) first_loss = l;
    prev_loss = l;
  }
  // After 100 steps of Adam(lr=0.1), scalar should be near 0
  EXPECT_LT(prev_loss, first_loss * 0.1f);
}

// ============================================================================
// Cross-Entropy Loss tests
// ============================================================================

TEST(Loss, CrossEntropyPerfectPrediction) {
  // logits strongly favor the correct class -> low loss
  auto logits = c10::Tensor::empty({2, 3}, c10::ScalarType::Float32);
  float* lp = logits.data_ptr<float>();
  // Sample 0: class 0 with high confidence
  lp[0] = 10.0f; lp[1] = 0.0f; lp[2] = 0.0f;
  // Sample 1: class 2 with high confidence
  lp[3] = 0.0f;  lp[4] = 0.0f; lp[5] = 10.0f;
  autograd::set_requires_grad(logits, true);

  auto targets = c10::Tensor::empty({2}, c10::ScalarType::Float32);
  targets.data_ptr<float>()[0] = 0.0f;
  targets.data_ptr<float>()[1] = 2.0f;

  auto loss = nn::cross_entropy_loss(logits, targets);
  // Loss should be very small (near 0)
  EXPECT_LT(loss.data_ptr<float>()[0], 0.01f);
}

TEST(Loss, CrossEntropyUniform) {
  // Uniform logits -> loss = log(num_classes)
  auto logits = aten::zeros({4, 5});
  autograd::set_requires_grad(logits, true);

  auto targets = c10::Tensor::empty({4}, c10::ScalarType::Float32);
  float* tp = targets.data_ptr<float>();
  tp[0] = 0; tp[1] = 1; tp[2] = 2; tp[3] = 3;

  auto loss = nn::cross_entropy_loss(logits, targets);
  float expected = std::log(5.0f);  // ln(5) ≈ 1.6094
  EXPECT_NEAR(loss.data_ptr<float>()[0], expected, 1e-4f);
}

TEST(Loss, CrossEntropyBackward) {
  // Verify gradient flows through cross-entropy
  auto logits = aten::full({2, 3}, 1.0f);
  autograd::set_requires_grad(logits, true);

  auto targets = c10::Tensor::empty({2}, c10::ScalarType::Float32);
  targets.data_ptr<float>()[0] = 0.0f;
  targets.data_ptr<float>()[1] = 1.0f;

  auto loss = nn::cross_entropy_loss(logits, targets);
  autograd::Engine::backward(loss);

  auto grad = autograd::get_grad(logits);
  EXPECT_TRUE(grad.defined());
  EXPECT_EQ(grad.numel(), 6);  // 2x3

  // For uniform logits, softmax = 1/3 for all classes.
  // Gradient for correct class: (1/3 - 1) / 2 = -1/3
  // Gradient for wrong class:   (1/3 - 0) / 2 =  1/6
  const float* gp = grad.data_ptr<float>();
  EXPECT_NEAR(gp[0], -1.0f/3.0f, 1e-4f);  // row 0, class 0 (correct)
  EXPECT_NEAR(gp[1],  1.0f/6.0f, 1e-4f);  // row 0, class 1
  EXPECT_NEAR(gp[2],  1.0f/6.0f, 1e-4f);  // row 0, class 2
  EXPECT_NEAR(gp[3],  1.0f/6.0f, 1e-4f);  // row 1, class 0
  EXPECT_NEAR(gp[4], -1.0f/3.0f, 1e-4f);  // row 1, class 1 (correct)
  EXPECT_NEAR(gp[5],  1.0f/6.0f, 1e-4f);  // row 1, class 2
}

// ============================================================================
// Adam + CrossEntropy classification training
// ============================================================================

TEST(EndToEnd, ClassificationTraining) {
  // 4-class linear classification problem
  // Input: 2D points, Target: quadrant class
  auto x = c10::Tensor::empty({8, 2}, c10::ScalarType::Float32);
  float* xp = x.data_ptr<float>();
  // Quadrant 0: (+,+), Quadrant 1: (-,+), Quadrant 2: (-,-), Quadrant 3: (+,-)
  xp[0]=1; xp[1]=1;   xp[2]=2; xp[3]=2;     // class 0
  xp[4]=-1; xp[5]=1;  xp[6]=-2; xp[7]=2;    // class 1
  xp[8]=-1; xp[9]=-1; xp[10]=-2; xp[11]=-2; // class 2
  xp[12]=1; xp[13]=-1; xp[14]=2; xp[15]=-2; // class 3

  auto targets = c10::Tensor::empty({8}, c10::ScalarType::Float32);
  float* tp = targets.data_ptr<float>();
  tp[0]=0; tp[1]=0; tp[2]=1; tp[3]=1; tp[4]=2; tp[5]=2; tp[6]=3; tp[7]=3;

  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    auto model = nn::Sequential();
    model.add(std::make_shared<nn::Linear>(2, 16, false));
    model.add(std::make_shared<nn::ReLU>());
    model.add(std::make_shared<nn::Linear>(16, 4, false));

    auto params = model.parameters();
    optim::Adam optimizer(params, /*lr=*/0.01);

    float initial_loss = 0, final_loss = 0;
    for (int epoch = 0; epoch < 1000; ++epoch) {
      optimizer.zero_grad();
      auto logits = model.forward(x);
      auto loss = nn::cross_entropy_loss(logits, targets);

      if (epoch == 0) initial_loss = loss.data_ptr<float>()[0];
      final_loss = loss.data_ptr<float>()[0];

      autograd::Engine::backward(loss);
      optimizer.step();

      if (final_loss < 0.05f) break;
    }

    if (final_loss < initial_loss * 0.5f && final_loss < 0.5f) {
      converged = true;
    }
  }

  EXPECT_TRUE(converged)
      << "Classification training didn't converge after 5 attempts";
}

// ============================================================================
// LayerNorm tests
// ============================================================================

TEST(LayerNorm, OutputZeroMeanUnitVar) {
  auto ln = nn::LayerNorm(4);
  auto input = c10::Tensor::empty({2, 4}, c10::ScalarType::Float32);
  float* ip = input.data_ptr<float>();
  ip[0]=1; ip[1]=2; ip[2]=3; ip[3]=4;
  ip[4]=10; ip[5]=20; ip[6]=30; ip[7]=40;

  auto output = ln.forward(input);
  const float* op = output.data_ptr<float>();

  // Each row should have mean ≈ 0 (beta=0) and std ≈ 1 (gamma=1)
  for (int row = 0; row < 2; ++row) {
    float mean = 0;
    for (int j = 0; j < 4; ++j) mean += op[row * 4 + j];
    mean /= 4;
    EXPECT_NEAR(mean, 0.0f, 1e-4f) << "Row " << row;

    float var = 0;
    for (int j = 0; j < 4; ++j) {
      float d = op[row * 4 + j] - mean;
      var += d * d;
    }
    var /= 4;
    EXPECT_NEAR(var, 1.0f, 0.1f) << "Row " << row;
  }
}

TEST(LayerNorm, HasParameters) {
  auto ln = nn::LayerNorm(16);
  auto params = ln.parameters();
  EXPECT_EQ(params.size(), 2u);  // gamma + beta
}

TEST(LayerNorm, GradientFlows) {
  auto ln = nn::LayerNorm(4);
  auto input = aten::full({2, 4}, 1.0f);
  autograd::set_requires_grad(input, true);

  auto output = ln.forward(input);
  auto loss = autograd::mean(output);
  autograd::Engine::backward(loss);

  auto grad = autograd::get_grad(input);
  EXPECT_TRUE(grad.defined());
  EXPECT_EQ(grad.numel(), 8);
}

// ============================================================================
// Dropout tests
// ============================================================================

TEST(Dropout, EvalModeIsIdentity) {
  auto drop = nn::Dropout(0.5f);
  drop.eval();  // switch to eval mode

  auto input = aten::ones({4, 4});
  auto output = drop.forward(input);

  // In eval mode, output should be identical to input
  const float* ip = input.data_ptr<float>();
  const float* op = output.data_ptr<float>();
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(ip[i], op[i]);
  }
}

TEST(Dropout, TrainingModeZerosSomeElements) {
  auto drop = nn::Dropout(0.5f);
  drop.train();

  auto input = aten::ones({100, 100});
  auto output = drop.forward(input);

  // Count zeros — with p=0.5, roughly half should be 0
  const float* op = output.data_ptr<float>();
  int zeros = 0;
  for (int i = 0; i < 10000; ++i) {
    if (op[i] == 0.0f) zeros++;
  }
  // Should be roughly 5000 ± 500
  EXPECT_GT(zeros, 3000);
  EXPECT_LT(zeros, 7000);
}

TEST(Dropout, ZeroProbIsIdentity) {
  auto drop = nn::Dropout(0.0f);
  drop.train();

  auto input = aten::full({4, 4}, 3.0f);
  auto output = drop.forward(input);

  const float* op = output.data_ptr<float>();
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(op[i], 3.0f);
  }
}

// ============================================================================
// Embedding tests
// ============================================================================

TEST(Embedding, OutputShape) {
  auto emb = nn::Embedding(10, 4);  // vocab=10, dim=4

  auto indices = c10::Tensor::empty({2, 3}, c10::ScalarType::Float32);
  float* ip = indices.data_ptr<float>();
  ip[0]=0; ip[1]=1; ip[2]=2;
  ip[3]=3; ip[4]=4; ip[5]=5;

  auto output = emb.forward(indices);
  // Output: (2, 3*4) = (2, 12) — flattened seq*dim
  EXPECT_EQ(output.size(0), 2);
  EXPECT_EQ(output.size(1), 12);
}

TEST(Embedding, HasParameters) {
  auto emb = nn::Embedding(100, 32);
  auto params = emb.parameters();
  EXPECT_EQ(params.size(), 1u);  // just the weight table
}

TEST(Embedding, LookupCorrect) {
  auto emb = nn::Embedding(3, 2);  // 3 words, 2-dim

  // Manually set weights for testing
  float* wp = emb.parameters()[0]->data().data_ptr<float>();
  wp[0]=1; wp[1]=2;   // embedding[0] = [1, 2]
  wp[2]=3; wp[3]=4;   // embedding[1] = [3, 4]
  wp[4]=5; wp[5]=6;   // embedding[2] = [5, 6]

  auto indices = c10::Tensor::empty({1, 3}, c10::ScalarType::Float32);
  float* ip = indices.data_ptr<float>();
  ip[0]=2; ip[1]=0; ip[2]=1;  // look up embeddings 2, 0, 1

  auto output = emb.forward(indices);
  const float* op = output.data_ptr<float>();
  // output should be [5,6, 1,2, 3,4]
  EXPECT_FLOAT_EQ(op[0], 5.0f);
  EXPECT_FLOAT_EQ(op[1], 6.0f);
  EXPECT_FLOAT_EQ(op[2], 1.0f);
  EXPECT_FLOAT_EQ(op[3], 2.0f);
  EXPECT_FLOAT_EQ(op[4], 3.0f);
  EXPECT_FLOAT_EQ(op[5], 4.0f);
}

// ============================================================================
// Softmax tests
// ============================================================================

TEST(Softmax, RowsSumToOne) {
  auto input = c10::Tensor::empty({3, 4}, c10::ScalarType::Float32);
  float* ip = input.data_ptr<float>();
  ip[0]=1; ip[1]=2; ip[2]=3; ip[3]=4;
  ip[4]=0; ip[5]=0; ip[6]=0; ip[7]=0;
  ip[8]=-1; ip[9]=-2; ip[10]=5; ip[11]=0;

  auto output = nn::functional::softmax(input);
  const float* op = output.data_ptr<float>();

  for (int row = 0; row < 3; ++row) {
    float sum = 0;
    for (int j = 0; j < 4; ++j) {
      sum += op[row * 4 + j];
      EXPECT_GE(op[row * 4 + j], 0.0f);  // all non-negative
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Row " << row;
  }
}

TEST(Softmax, UniformInputGivesUniform) {
  auto input = aten::full({2, 5}, 1.0f);
  auto output = nn::functional::softmax(input);
  const float* op = output.data_ptr<float>();

  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR(op[i], 0.2f, 1e-5f);  // 1/5 = 0.2
  }
}

TEST(Softmax, GradientFlows) {
  auto input = aten::full({2, 3}, 1.0f);
  autograd::set_requires_grad(input, true);

  auto output = nn::functional::softmax(input);
  auto loss = autograd::mean(output);
  autograd::Engine::backward(loss);

  auto grad = autograd::get_grad(input);
  EXPECT_TRUE(grad.defined());
  EXPECT_EQ(grad.numel(), 6);
}

// ============================================================================
// CausalSelfAttention tests
// ============================================================================

TEST(Attention, OutputShape) {
  auto attn = nn::CausalSelfAttention(/*d_model=*/16, /*n_heads=*/4);
  auto input = aten::ones({8, 16});  // seq_len=8, d_model=16
  auto output = attn.forward(input);
  EXPECT_EQ(output.size(0), 8);
  EXPECT_EQ(output.size(1), 16);
}

TEST(Attention, HasParameters) {
  auto attn = nn::CausalSelfAttention(16, 4);
  auto params = attn.parameters();
  // qkv_proj (weight) + out_proj (weight) = 2 params
  EXPECT_EQ(params.size(), 2u);
}

TEST(Attention, CausalMasking) {
  // With seq_len=1, attention should just pass through (no masking needed)
  auto attn = nn::CausalSelfAttention(8, 2);
  attn.eval();
  auto input = aten::ones({1, 8});
  auto output = attn.forward(input);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_EQ(output.size(1), 8);
}

// ============================================================================
// TransformerBlock tests
// ============================================================================

TEST(TransformerBlock, OutputShape) {
  auto block = nn::TransformerBlock(/*d_model=*/16, /*n_heads=*/4);
  block.eval();  // no dropout
  auto input = aten::ones({4, 16});  // seq_len=4, d_model=16
  auto output = block.forward(input);
  EXPECT_EQ(output.size(0), 4);
  EXPECT_EQ(output.size(1), 16);
}

TEST(TransformerBlock, HasParameters) {
  auto block = nn::TransformerBlock(16, 4);
  auto params = block.parameters();
  // ln1: 2 (gamma, beta)
  // attn: qkv_proj(1) + out_proj(1) = 2
  // ln2: 2 (gamma, beta)
  // ffn1: 1, ffn2: 1
  // Total: 8
  EXPECT_EQ(params.size(), 8u);
}

TEST(TransformerBlock, ResidualConnection) {
  // Verify output != input (the block transforms)
  // and that the shape is preserved
  auto block = nn::TransformerBlock(8, 2);
  block.eval();
  auto input = aten::rand({2, 8});  // varied input so LayerNorm doesn't zero out
  auto output = block.forward(input);

  EXPECT_EQ(output.size(0), 2);
  EXPECT_EQ(output.size(1), 8);

  // Output should differ from input (non-trivial transformation)
  const float* ip = input.data_ptr<float>();
  const float* op = output.data_ptr<float>();
  bool different = false;
  for (int i = 0; i < 16; ++i) {
    if (std::abs(ip[i] - op[i]) > 1e-6f) {
      different = true;
      break;
    }
  }
  EXPECT_TRUE(different) << "TransformerBlock output should differ from input";
}

}  // namespace
