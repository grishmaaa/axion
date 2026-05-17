// ============================================================================
// Axion Example — XOR MLP
// ============================================================================
//
// A 2-layer MLP learns the XOR function:
//   [0,0] -> 0    [0,1] -> 1    [1,0] -> 1    [1,1] -> 0
//
// Architecture:  Linear(2, 32) -> ReLU -> Linear(32, 1)
// Loss:          MSE
// Optimizer:     Adam(lr=0.01)
//
// This example demonstrates the complete Axion training loop:
//   1. Model construction via nn::Sequential
//   2. Forward pass through autograd-aware ops
//   3. Loss computation
//   4. Backward pass via autograd::Engine
//   5. Parameter update via optim::Adam
//
// Build:  cmake --build build --target xor_mlp
// Run:    ./build/examples/xor_mlp
//

#include <cstdio>
#include <cmath>

#include "axion/nn/Linear.h"
#include "axion/nn/Sequential.h"
#include "axion/nn/Activations.h"
#include "axion/nn/Loss.h"
#include "axion/optim/Adam.h"
#include "autograd/engine/Engine.h"
#include "aten/ops/Ops.h"

int main() {
  printf("╔══════════════════════════════════════════╗\n");
  printf("║     Axion — XOR MLP Training Example     ║\n");
  printf("╚══════════════════════════════════════════╝\n\n");

  // ---------------------------------------------------------------
  // 1. Data
  // ---------------------------------------------------------------
  auto x = c10::Tensor::empty({4, 2}, c10::ScalarType::Float32);
  float* xp = x.data_ptr<float>();
  xp[0] = 0; xp[1] = 0;   // -> 0
  xp[2] = 0; xp[3] = 1;   // -> 1
  xp[4] = 1; xp[5] = 0;   // -> 1
  xp[6] = 1; xp[7] = 1;   // -> 0

  auto target = c10::Tensor::empty({4, 1}, c10::ScalarType::Float32);
  float* tp = target.data_ptr<float>();
  tp[0] = 0; tp[1] = 1; tp[2] = 1; tp[3] = 0;

  // ---------------------------------------------------------------
  // 2. Model
  // ---------------------------------------------------------------
  auto model = nn::Sequential();
  model.add(std::make_shared<nn::Linear>(2, 32, false));
  model.add(std::make_shared<nn::ReLU>());
  model.add(std::make_shared<nn::Linear>(32, 1, false));

  auto params = model.parameters();
  printf("Model: Linear(2,32) -> ReLU -> Linear(32,1)\n");
  printf("Parameters: %zu tensors\n\n", params.size());

  // ---------------------------------------------------------------
  // 3. Optimizer
  // ---------------------------------------------------------------
  optim::Adam optimizer(params, /*lr=*/0.01);

  // ---------------------------------------------------------------
  // 4. Training loop
  // ---------------------------------------------------------------
  const int epochs = 2000;
  printf("Training for %d epochs with Adam(lr=0.01)...\n\n", epochs);
  printf("  Epoch  |   Loss\n");
  printf("  -------+----------\n");

  for (int epoch = 0; epoch < epochs; ++epoch) {
    optimizer.zero_grad();

    auto pred = model.forward(x);
    auto loss = nn::mse_loss(pred, target);

    float l = loss.data_ptr<float>()[0];

    if (epoch % 200 == 0 || epoch == epochs - 1) {
      printf("  %5d  |  %.6f\n", epoch, l);
    }

    autograd::Engine::backward(loss);
    optimizer.step();

    if (l < 1e-5f) {
      printf("  %5d  |  %.6f  <-- converged!\n", epoch, l);
      break;
    }
  }

  // ---------------------------------------------------------------
  // 5. Inference
  // ---------------------------------------------------------------
  printf("\n  Predictions:\n");
  printf("  ──────────────────────────\n");

  model.eval();  // switch to eval mode
  auto final_pred = model.forward(x);
  const float* pp = final_pred.data_ptr<float>();

  const char* inputs[] = {"[0, 0]", "[0, 1]", "[1, 0]", "[1, 1]"};
  const int targets[] = {0, 1, 1, 0};

  for (int i = 0; i < 4; ++i) {
    float predicted = pp[i];
    int rounded = predicted > 0.5f ? 1 : 0;
    const char* status = (rounded == targets[i]) ? "✓" : "✗";
    printf("  %s -> %.4f (≈%d) %s\n", inputs[i], predicted, rounded, status);
  }

  printf("\n══════════════════════════════════════════\n");
  printf("Done.\n");

  return 0;
}
