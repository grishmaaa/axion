# ============================================================================
# c10 tests
# ============================================================================

add_executable(test_intrusive_ptr intrusive_ptr_test.cpp)
target_link_libraries(test_intrusive_ptr PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(test_intrusive_ptr)

add_executable(device_test device_test.cpp)
target_link_libraries(device_test PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(device_test)

add_executable(dataptr_test dataptr_test.cpp)
target_link_libraries(dataptr_test PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(dataptr_test)

add_executable(allocator_test allocator_test.cpp)
target_link_libraries(allocator_test PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(allocator_test)

add_executable(cpu_allocator_test cpu_allocator_test.cpp)
target_link_libraries(cpu_allocator_test PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(cpu_allocator_test)
