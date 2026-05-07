# ============================================================================
# c10 tests
# ============================================================================

add_executable(test_intrusive_ptr intrusive_ptr_test.cpp)
target_link_libraries(test_intrusive_ptr PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(test_intrusive_ptr)
