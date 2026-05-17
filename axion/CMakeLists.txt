# ============================================================================
# axion — User-facing API (nn, optim, etc.)
# ============================================================================

file(GLOB_RECURSE AXION_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/nn/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/optim/*.cpp"
)

if(AXION_SRCS)
  add_library(axion OBJECT ${AXION_SRCS})
  target_include_directories(axion PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(axion PUBLIC autograd aten c10)
  target_compile_features(axion PUBLIC cxx_std_17)
else()
  add_library(axion INTERFACE)
  target_include_directories(axion INTERFACE ${CMAKE_SOURCE_DIR})
  target_link_libraries(axion INTERFACE autograd aten c10)
  target_compile_features(axion INTERFACE cxx_std_17)
endif()
