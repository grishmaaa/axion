# ============================================================================
# axion — User-facing API (placeholder)
# ============================================================================
add_library(axion INTERFACE)
target_include_directories(axion INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(axion INTERFACE autograd)
