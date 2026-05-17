# ============================================================================
# aten — Tensor operations library
# ============================================================================
#
# Now a STATIC library — Ops.cpp provides the first translation unit.

file(GLOB_RECURSE ATEN_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/native/*.cpp"
)

if(ATEN_SRCS)
  add_library(aten STATIC ${ATEN_SRCS})
  target_include_directories(aten PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(aten PUBLIC c10)
  target_compile_features(aten PUBLIC cxx_std_17)
else()
  add_library(aten INTERFACE)
  target_include_directories(aten INTERFACE ${CMAKE_SOURCE_DIR})
  target_link_libraries(aten INTERFACE c10)
  target_compile_features(aten INTERFACE cxx_std_17)
endif()
