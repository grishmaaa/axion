# ============================================================================
# ATen tests
# ============================================================================

add_executable(tensor_iterator_test tensor_iterator_test.cpp)
target_link_libraries(tensor_iterator_test PRIVATE aten c10 GTest::gtest_main)
gtest_discover_tests(tensor_iterator_test)
