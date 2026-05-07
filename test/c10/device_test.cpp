#include <gtest/gtest.h>
#include "c10/core/Device.h"

namespace {

using namespace c10;

// 1. CPU device type is correct
TEST(DeviceTest, CPUDeviceTypeIsCorrect) {
  Device d(DeviceType::CPU);
  EXPECT_EQ(d.type, DeviceType::CPU);
}

// 2. CPU device index is 0
TEST(DeviceTest, CPUDeviceIndexIsZero) {
  Device d(DeviceType::CPU);
  EXPECT_EQ(d.index, 0);
}

// 3. is_cpu() true for CPU
TEST(DeviceTest, IsCPUTrueForCPU) {
  Device d(DeviceType::CPU);
  EXPECT_TRUE(d.is_cpu());
}

// 4. is_cuda() false for CPU
TEST(DeviceTest, IsCUDAFalseForCPU) {
  Device d(DeviceType::CPU);
  EXPECT_FALSE(d.is_cuda());
}

// 5. CUDA device type is correct
TEST(DeviceTest, CUDADeviceTypeIsCorrect) {
  Device d(DeviceType::CUDA);
  EXPECT_EQ(d.type, DeviceType::CUDA);
}

// 6. CUDA device default index is 0
TEST(DeviceTest, CUDADeviceDefaultIndexIsZero) {
  Device d(DeviceType::CUDA);
  EXPECT_EQ(d.index, 0);
}

// 7. CUDA device explicit index stored correctly
TEST(DeviceTest, CUDADeviceExplicitIndexIsStored) {
  Device d(DeviceType::CUDA, 1);
  EXPECT_EQ(d.index, 1);
}

// 8. is_cuda() true for CUDA
TEST(DeviceTest, IsCUDATrueForCUDA) {
  Device d(DeviceType::CUDA);
  EXPECT_TRUE(d.is_cuda());
}

// 9. is_cpu() false for CUDA
TEST(DeviceTest, IsCPUFalseForCUDA) {
  Device d(DeviceType::CUDA);
  EXPECT_FALSE(d.is_cpu());
}

// 10. CPU == CPU same index
TEST(DeviceTest, CPUEquality) {
  Device d1(DeviceType::CPU);
  Device d2(DeviceType::CPU);
  EXPECT_EQ(d1, d2);
}

// 11. CPU != CUDA
TEST(DeviceTest, CPUCUDANotEqual) {
  Device d1(DeviceType::CPU);
  Device d2(DeviceType::CUDA);
  EXPECT_NE(d1, d2);
}

// 12. CUDA index 0 != CUDA index 1
TEST(DeviceTest, CUDAIndexEquality) {
  Device d1(DeviceType::CUDA, 0);
  Device d2(DeviceType::CUDA, 1);
  EXPECT_NE(d1, d2);
}

// 13. Device::cpu() convenience constructor works
TEST(DeviceTest, CPUFactory) {
  Device d = Device::cpu();
  EXPECT_TRUE(d.is_cpu());
  EXPECT_EQ(d.index, 0);
}

// 14. Device::cuda() convenience constructor works
TEST(DeviceTest, CUDAFactory) {
  Device d = Device::cuda();
  EXPECT_TRUE(d.is_cuda());
  EXPECT_EQ(d.index, 0);
}

// 15. Device::cuda(2) stores index 2 correctly
TEST(DeviceTest, CUDAFactoryWithIndex) {
  Device d = Device::cuda(2);
  EXPECT_TRUE(d.is_cuda());
  EXPECT_EQ(d.index, 2);
}

// 16. str() returns "cpu" for CPU device
TEST(DeviceTest, CPUStr) {
  Device d = Device::cpu();
  EXPECT_STREQ(d.str(), "cpu");
}

// 17. str() returns "cuda" for CUDA device
TEST(DeviceTest, CUDAStr) {
  Device d = Device::cuda();
  EXPECT_STREQ(d.str(), "cuda");
}

} // namespace
