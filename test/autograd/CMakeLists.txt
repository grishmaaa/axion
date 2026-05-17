# ============================================================================
# Autograd tests
# ============================================================================

add_executable(autograd_meta_test autograd_meta_test.cpp)
target_link_libraries(autograd_meta_test PRIVATE autograd aten c10 GTest::gtest_main)
gtest_discover_tests(autograd_meta_test)

add_executable(engine_test engine_test.cpp)
target_link_libraries(engine_test PRIVATE autograd aten c10 GTest::gtest_main)
gtest_discover_tests(engine_test)

add_executable(variable_test variable_test.cpp)
target_link_libraries(variable_test PRIVATE autograd aten c10 GTest::gtest_main)
gtest_discover_tests(variable_test)
