# ============================================================================
# ATen tests
# ============================================================================

add_executable(tensor_iterator_test tensor_iterator_test.cpp)
target_link_libraries(tensor_iterator_test PRIVATE aten c10 GTest::gtest_main)
gtest_discover_tests(tensor_iterator_test)

add_executable(ops_test ops_test.cpp)
target_link_libraries(ops_test PRIVATE aten c10 GTest::gtest_main)
gtest_discover_tests(ops_test)

