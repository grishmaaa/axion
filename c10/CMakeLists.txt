# ============================================================================
# c10 — Core runtime library
# ============================================================================
#
# This is a header-only library for now.  As we add .cpp translation units
# (e.g., Allocator.cpp, DeviceGuard.cpp) it will become a STATIC library.

# Collect sources — currently empty; will grow as we add .cpp files.
file(GLOB_RECURSE C10_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp"
)

# If no .cpp files exist yet, create an INTERFACE (header-only) target.
# Once we have real sources, switch to a STATIC library.
if(C10_SRCS)
  add_library(c10 STATIC ${C10_SRCS})
  target_include_directories(c10 PUBLIC ${CMAKE_SOURCE_DIR})
  target_compile_features(c10 PUBLIC cxx_std_17)
else()
  add_library(c10 INTERFACE)
  target_include_directories(c10 INTERFACE ${CMAKE_SOURCE_DIR})
  target_compile_features(c10 INTERFACE cxx_std_17)
endif()
