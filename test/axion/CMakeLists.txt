# ============================================================================
# Axion tests (nn, optim, end-to-end)
# ============================================================================

add_executable(nn_test nn_test.cpp)
target_link_libraries(nn_test PRIVATE axion autograd aten c10 GTest::gtest_main)
gtest_discover_tests(nn_test)
