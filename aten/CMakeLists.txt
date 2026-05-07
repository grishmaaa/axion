# ============================================================================
# aten — Tensor library (placeholder)
# ============================================================================
add_library(aten INTERFACE)
target_include_directories(aten INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(aten INTERFACE c10)
