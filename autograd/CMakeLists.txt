# ============================================================================
# autograd — Automatic differentiation (placeholder)
# ============================================================================
add_library(autograd INTERFACE)
target_include_directories(autograd INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(autograd INTERFACE aten)
