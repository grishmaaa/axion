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

add_executable(storage_impl_test storage_impl_test.cpp)
target_link_libraries(storage_impl_test PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(storage_impl_test)

add_executable(storage_test storage_test.cpp)
target_link_libraries(storage_test PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(storage_test)

add_executable(tensor_impl_test tensor_impl_test.cpp)
target_link_libraries(tensor_impl_test PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(tensor_impl_test)

add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE c10 GTest::gtest_main)
gtest_discover_tests(tensor_test)

add_executable(dispatch_test dispatch_test.cpp)
target_link_libraries(dispatch_test PRIVATE aten c10 GTest::gtest_main)
gtest_discover_tests(dispatch_test)
