# ============================================================================
# Axion — A Pure C++ Deep Learning Framework
# ============================================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Axion VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(AXION_BUILD_TESTS      "Build unit tests"              ON)
option(AXION_BUILD_BENCHMARKS "Build benchmarks"              OFF)
option(AXION_BUILD_EXAMPLES   "Build example programs"        ON)
option(AXION_USE_CUDA         "Enable CUDA backend"           OFF)

# ---------------------------------------------------------------------------
# Include paths — mirror the layered architecture
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Core libraries (bottom-up dependency order)
# ---------------------------------------------------------------------------
# Layer 1: c10  — device/type/allocator runtime
# Layer 2: aten — tensor operations & compute kernels
# Layer 3: autograd — automatic differentiation engine
# Layer 4: axion — user-facing API (nn, optim, data)

add_subdirectory(c10)
add_subdirectory(aten)
add_subdirectory(autograd)
add_subdirectory(axion)

# ---------------------------------------------------------------------------
# Optional targets
# ---------------------------------------------------------------------------
if(AXION_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(AXION_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(AXION_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

message(STATUS "")
message(STATUS "Axion v${PROJECT_VERSION} configuration:")
message(STATUS "  C++ standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:          ${AXION_USE_CUDA}")
message(STATUS "  Tests:         ${AXION_BUILD_TESTS}")
message(STATUS "  Benchmarks:    ${AXION_BUILD_BENCHMARKS}")
message(STATUS "  Examples:      ${AXION_BUILD_EXAMPLES}")
message(STATUS "")
