# Axion ⚛️

> A pure C++ deep learning framework built from scratch.

Axion is a modular deep learning framework written entirely in C++17.
It follows a layered architecture inspired by PyTorch's internal design,
where each layer builds on the one below it — from raw hardware abstraction
all the way up to a user-friendly neural network API.

**Zero Python. Pure C++.**

---

## Architecture

The framework is organized as a strict dependency stack:

```
┌─────────────────────────────────────────┐
│            axion/  (Layer 4)            │  User-facing API
│         nn · optim · data               │  What you #include to train models
├─────────────────────────────────────────┤
│          autograd/  (Layer 3)           │  Automatic differentiation
│       engine · functions                │  Builds & evaluates the grad graph
├─────────────────────────────────────────┤
│            aten/  (Layer 2)             │  Tensor library
│      core · ops · native/cpu · cuda     │  Tensor storage + math kernels
├─────────────────────────────────────────┤
│            c10/  (Layer 1)              │  Core runtime
│    core · util · macros                 │  Device, types, allocator, threading
└─────────────────────────────────────────┘
              ▼  hardware  ▼
```

**Dependency rule:** each layer may only depend on layers *below* it. `axion/` depends
on `autograd/`, which depends on `aten/`, which depends on `c10/`. Nothing goes upward.

---

## Directory Structure

```
Axion/
│
├── c10/                    # Layer 1: Core runtime library
│   ├── core/               #   Device, DeviceType, ScalarType, Allocator,
│   │                       #   Storage, TensorOptions, DispatchKey
│   ├── util/               #   SmallVector, intrusive_ptr, ArrayRef,
│   │                       #   Logging, Exception helpers
│   ├── macros/             #   Export macros, platform detection
│   └── test/               #   Unit tests for c10
│
├── aten/                   # Layer 2: Tensor library ("A Tensor")
│   ├── core/               #   Tensor, TensorImpl, TensorAccessor
│   ├── ops/                #   Op registration & dispatch tables
│   ├── native/             #   Actual math kernel implementations
│   │   ├── cpu/            #     CPU kernels (add, matmul, conv, relu, etc.)
│   │   └── cuda/           #     CUDA kernels (same ops, GPU versions)
│   └── test/               #   Unit tests for aten
│
├── autograd/               # Layer 3: Automatic differentiation
│   ├── engine/             #   Backward engine, topological sort, graph exec
│   ├── functions/          #   Backward functions (AddBackward, MulBackward, etc.)
│   └── test/               #   Unit tests for autograd
│
├── axion/                  # Layer 4: User-facing C++ API
│   ├── nn/                 #   Module, Linear, Conv2d, ReLU, Sequential, loss fns
│   ├── optim/              #   SGD, Adam, LR schedulers
│   ├── data/               #   Dataset, DataLoader, Sampler, transforms
│   ├── serialize/          #   Model save/load (checkpointing)
│   └── test/               #   Unit tests for axion API
│
├── test/                   # Integration & cross-layer tests
│   ├── c10/
│   ├── aten/
│   ├── autograd/
│   └── axion/
│
├── examples/               # Example programs showing the API in action
├── benchmarks/             # Performance benchmarks
├── tools/                  # Build scripts, code generators, CI helpers
├── third_party/            # External dependencies (googletest, etc.)
├── docs/                   # Documentation
│
├── CMakeLists.txt          # Root build file (builds layers bottom-up)
├── .clang-format           # Code style config
└── .gitignore
```

---

## Layer Details

### Layer 1 — `c10/` (Core Ten)
The foundation everything else is built on. Defines the vocabulary types
that the rest of the framework speaks:

| Component | What it does |
|-----------|-------------|
| `Device` / `DeviceType` | Abstraction over CPU, CUDA, etc. |
| `ScalarType` | Enum of numeric types (Float32, Int64, etc.) |
| `Allocator` | Interface for memory allocation per device |
| `Storage` | Ref-counted raw memory buffer |
| `TensorOptions` | Bundle of (dtype, device, layout) for tensor creation |
| `DispatchKey` | Tags for routing ops to the right kernel |

### Layer 2 — `aten/` (A Tensor Library)
The tensor math engine. Owns `Tensor` and all the operations on it:

| Component | What it does |
|-----------|-------------|
| `TensorImpl` | The actual data structure behind `Tensor` (metadata + Storage) |
| `Tensor` | Thin, copyable handle wrapping `TensorImpl` via intrusive_ptr |
| `ops/` | Operator dispatch tables — maps `(op, dispatch_key) → kernel` |
| `native/cpu/` | CPU kernel implementations (loops, vectorized math) |
| `native/cuda/` | GPU kernel implementations (CUDA kernels) |

### Layer 3 — `autograd/`
Reverse-mode automatic differentiation bolted on top of `aten/`:

| Component | What it does |
|-----------|-------------|
| `engine/` | Runs backward pass — topological sort of the computation graph |
| `functions/` | One class per differentiable op, defining `backward()` for each |

### Layer 4 — `axion/` (User API)
The friendly face of the framework — what users actually `#include`:

| Component | What it does |
|-----------|-------------|
| `nn/` | Neural network modules (`Module`, `Linear`, `Conv2d`, `ReLU`, losses) |
| `optim/` | Optimizers (`SGD`, `Adam`) and learning rate schedulers |
| `data/` | `Dataset` / `DataLoader` for batched data loading |
| `serialize/` | Model checkpointing (save & load) |

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## Design Principles

1. **Strict layering** — no upward dependencies, each layer is testable in isolation
2. **Value semantics** — `Tensor` is a lightweight handle (like `shared_ptr`)
3. **Dispatch-based extensibility** — new backends (e.g. Metal, XLA) plug in via `DispatchKey`
4. **No Python** — this is a pure C++ framework, end to end
5. **Modern C++17** — `std::optional`, `std::variant`, structured bindings, constexpr
