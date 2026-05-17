# ============================================================================
# aten — Tensor operations library
# ============================================================================
#
# Now a STATIC library — Ops.cpp provides the first translation unit.
# Native CPU kernel files use static registration (RegisterKernel) to
# populate the Dispatcher at init time.  These registrations have no
# direct symbol references from other TUs, so we must use OBJECT library
# linkage to prevent the linker from stripping them.

file(GLOB_RECURSE ATEN_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/native/*.cpp"
)

if(ATEN_SRCS)
  # OBJECT library ensures all .o files (including static registrars)
  # are linked into every target that depends on aten.
  add_library(aten OBJECT ${ATEN_SRCS})
  target_include_directories(aten PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(aten PUBLIC c10)
  target_compile_features(aten PUBLIC cxx_std_17)
else()
  add_library(aten INTERFACE)
  target_include_directories(aten INTERFACE ${CMAKE_SOURCE_DIR})
  target_link_libraries(aten INTERFACE c10)
  target_compile_features(aten INTERFACE cxx_std_17)
endif()
