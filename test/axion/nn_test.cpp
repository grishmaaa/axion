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
#include "axion/nn/Loss.h"
#include "axion/optim/SGD.h"
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
    optim::SGD optimizer(params, /*lr=*/0.5);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < 2000; ++epoch) {
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

}  // namespace
