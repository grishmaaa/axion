# ============================================================================
# autograd — Automatic differentiation
# ============================================================================
#
# OBJECT library (like aten) to prevent linker stripping of static
# registrars and to ensure all TUs are linked.

file(GLOB_RECURSE AUTOGRAD_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/functions/*.cpp"
)

if(AUTOGRAD_SRCS)
  add_library(autograd OBJECT ${AUTOGRAD_SRCS})
  target_include_directories(autograd PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(autograd PUBLIC aten c10)
  target_compile_features(autograd PUBLIC cxx_std_17)
else()
  add_library(autograd INTERFACE)
  target_include_directories(autograd INTERFACE ${CMAKE_SOURCE_DIR})
  target_link_libraries(autograd INTERFACE aten c10)
  target_compile_features(autograd INTERFACE cxx_std_17)
endif()
