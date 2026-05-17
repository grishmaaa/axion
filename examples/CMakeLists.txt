# ============================================================================
# examples — Standalone executables demonstrating Axion
# ============================================================================

add_executable(xor_mlp xor_mlp.cpp)
target_link_libraries(xor_mlp PRIVATE
  axion autograd aten c10
)
target_include_directories(xor_mlp PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(xor_mlp PRIVATE cxx_std_17)
