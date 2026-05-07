// ============================================================================
// Tests for c10::DataPtr
// ============================================================================
//
// 20 test cases covering:
//   - Null construction
//   - CPU construction with malloc
//   - Destructor behavior / deleter firing
//   - Deleter fires exactly once
//   - Move semantics (constructor and assignment)
//   - Stateful deleter receiving ctx_ not data_
//   - release_context() preventing destructor from firing
//   - clear() freeing immediately
//   - Double clear safety
//   - Device stored and retrieved correctly

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

#include "c10/core/DataPtr.h"

namespace {

using namespace c10;

// ---------------------------------------------------------------------------
// Helpers — tracking deleters
// ---------------------------------------------------------------------------

// Tracks whether the deleter was called, how many times, and with what pointer.
struct DeleterTracker {
  int call_count = 0;
  void* last_ctx = nullptr;
};

void trackingDeleter(void* ctx) {
  // ctx points to a DeleterTracker.
  auto* tracker = static_cast<DeleterTracker*>(ctx);
  tracker->call_count++;
  tracker->last_ctx = ctx;
}

// A deleter that records the pointer it was called with into an external slot.
static void* g_received_ctx = nullptr;
static int g_deleter_calls = 0;

void spyDeleter(void* ctx) {
  g_received_ctx = ctx;
  g_deleter_calls++;
}

void resetSpy() {
  g_received_ctx = nullptr;
  g_deleter_calls = 0;
}

// ============================================================================
// 1. Default constructor is null
// ============================================================================
TEST(DataPtr, DefaultConstructorIsNull) {
  DataPtr dp;
  EXPECT_EQ(dp.get(), nullptr);
  EXPECT_EQ(dp.get_context(), nullptr);
  EXPECT_EQ(dp.get_deleter(), nullptr);
  EXPECT_FALSE(static_cast<bool>(dp));
}

// ============================================================================
// 2. Default constructor device is CPU
// ============================================================================
TEST(DataPtr, DefaultConstructorDeviceIsCPU) {
  DataPtr dp;
  EXPECT_TRUE(dp.device().is_cpu());
  EXPECT_EQ(dp.device().index, 0);
}

// ============================================================================
// 3. CPU construction with malloc
// ============================================================================
TEST(DataPtr, CPUConstruction) {
  void* raw = std::malloc(64);
  ASSERT_NE(raw, nullptr);
  std::memset(raw, 0xAB, 64);

  DataPtr dp(raw, deleteCPU, Device::cpu());
  EXPECT_EQ(dp.get(), raw);
  EXPECT_EQ(dp.get_context(), raw);  // convenience ctor: ctx == data
  EXPECT_EQ(dp.get_deleter(), deleteCPU);
  EXPECT_TRUE(static_cast<bool>(dp));
  // dp's destructor will call free(raw) — ASan verifies no leak.
}

// ============================================================================
// 4. Full constructor — data and ctx are separate
// ============================================================================
TEST(DataPtr, FullConstructorSeparateDataAndCtx) {
  void* block = std::malloc(256);
  ASSERT_NE(block, nullptr);
  void* user = static_cast<char*>(block) + 64;  // user sees offset

  DataPtr dp(user, block, deleteCPU, Device::cpu());
  EXPECT_EQ(dp.get(), user);
  EXPECT_EQ(dp.get_context(), block);
  EXPECT_NE(dp.get(), dp.get_context());
  // Destructor will call free(block), not free(user). ASan validates.
}

// ============================================================================
// 5. Destructor fires deleter
// ============================================================================
TEST(DataPtr, DestructorFiresDeleter) {
  DeleterTracker tracker;
  {
    DataPtr dp(&tracker, &tracker, trackingDeleter, Device::cpu());
    EXPECT_EQ(tracker.call_count, 0);
  }  // dp goes out of scope
  EXPECT_EQ(tracker.call_count, 1);
}

// ============================================================================
// 6. Destructor does NOT fire when both ctx and del are null
// ============================================================================
TEST(DataPtr, DestructorSkipsWhenNull) {
  resetSpy();
  {
    DataPtr dp;  // null
  }
  EXPECT_EQ(g_deleter_calls, 0);
}

// ============================================================================
// 7. Deleter fires exactly once
// ============================================================================
TEST(DataPtr, DeleterFiresExactlyOnce) {
  DeleterTracker tracker;
  {
    DataPtr dp(&tracker, &tracker, trackingDeleter, Device::cpu());
  }
  EXPECT_EQ(tracker.call_count, 1);
}

// ============================================================================
// 8. deleteNothing does not free
// ============================================================================
TEST(DataPtr, DeleteNothingDoesNotFree) {
  int stack_var = 42;
  {
    DataPtr dp(&stack_var, deleteNothing, Device::cpu());
    EXPECT_EQ(dp.get(), &stack_var);
  }
  // If deleteNothing tried to free a stack variable, ASan would scream.
  EXPECT_EQ(stack_var, 42);
}

// ============================================================================
// 9. Move constructor transfers ownership
// ============================================================================
TEST(DataPtr, MoveConstructorTransfersOwnership) {
  void* raw = std::malloc(32);
  ASSERT_NE(raw, nullptr);

  DataPtr src(raw, deleteCPU, Device::cpu());
  DataPtr dst(std::move(src));

  EXPECT_EQ(dst.get(), raw);
  EXPECT_EQ(dst.get_context(), raw);
  EXPECT_EQ(dst.get_deleter(), deleteCPU);
  // src should be null after move.
  EXPECT_EQ(src.get(), nullptr);          // NOLINT: intentional use-after-move
  EXPECT_EQ(src.get_context(), nullptr);  // NOLINT
  EXPECT_EQ(src.get_deleter(), nullptr);  // NOLINT
  // dst's destructor will free. ASan validates exactly one free.
}

// ============================================================================
// 10. Move constructor — source becomes null
// ============================================================================
TEST(DataPtr, MoveConstructorNullsSource) {
  DeleterTracker tracker;
  DataPtr src(&tracker, &tracker, trackingDeleter, Device::cpu());
  DataPtr dst(std::move(src));

  EXPECT_FALSE(static_cast<bool>(src));   // NOLINT
  EXPECT_TRUE(static_cast<bool>(dst));
  EXPECT_EQ(tracker.call_count, 0);  // not freed yet
}

// ============================================================================
// 11. Move assignment transfers ownership
// ============================================================================
TEST(DataPtr, MoveAssignmentTransfersOwnership) {
  void* raw = std::malloc(16);
  ASSERT_NE(raw, nullptr);

  DataPtr src(raw, deleteCPU, Device::cpu());
  DataPtr dst;

  dst = std::move(src);
  EXPECT_EQ(dst.get(), raw);
  EXPECT_EQ(src.get(), nullptr);  // NOLINT
}

// ============================================================================
// 12. Move assignment frees previous target
// ============================================================================
TEST(DataPtr, MoveAssignmentFreesPrevious) {
  DeleterTracker tracker1;
  DeleterTracker tracker2;

  DataPtr dp1(&tracker1, &tracker1, trackingDeleter, Device::cpu());
  DataPtr dp2(&tracker2, &tracker2, trackingDeleter, Device::cpu());

  // Assigning dp2 into dp1 should free dp1's old allocation first.
  dp1 = std::move(dp2);
  EXPECT_EQ(tracker1.call_count, 1);  // old target freed
  EXPECT_EQ(tracker2.call_count, 0);  // new target still alive
}

// ============================================================================
// 13. Stateful deleter receives ctx_, not data_
// ============================================================================
TEST(DataPtr, DeleterReceivesCtxNotData) {
  resetSpy();

  int user_data = 0;
  int context_data = 0;
  void* user_ptr = &user_data;
  void* ctx_ptr = &context_data;

  {
    DataPtr dp(user_ptr, ctx_ptr, spyDeleter, Device::cpu());
    EXPECT_EQ(dp.get(), user_ptr);
    EXPECT_EQ(dp.get_context(), ctx_ptr);
  }

  // The deleter should have received ctx_ptr, NOT user_ptr.
  EXPECT_EQ(g_received_ctx, ctx_ptr);
  EXPECT_NE(g_received_ctx, user_ptr);
  EXPECT_EQ(g_deleter_calls, 1);
}

// ============================================================================
// 14. release_context returns the context pointer
// ============================================================================
TEST(DataPtr, ReleaseContextReturnsCtx) {
  int ctx_obj = 99;
  DataPtr dp(&ctx_obj, &ctx_obj, deleteNothing, Device::cpu());

  void* released = dp.release_context();
  EXPECT_EQ(released, &ctx_obj);
}

// ============================================================================
// 15. release_context prevents destructor from firing
// ============================================================================
TEST(DataPtr, ReleaseContextPreventsDeleter) {
  DeleterTracker tracker;
  {
    DataPtr dp(&tracker, &tracker, trackingDeleter, Device::cpu());
    void* ctx = dp.release_context();
    EXPECT_EQ(ctx, &tracker);
    EXPECT_EQ(dp.get(), nullptr);
    EXPECT_EQ(dp.get_context(), nullptr);
    EXPECT_EQ(dp.get_deleter(), nullptr);
  }  // dp destructor runs but should NOT call deleter
  EXPECT_EQ(tracker.call_count, 0);
}

// ============================================================================
// 16. clear frees immediately
// ============================================================================
TEST(DataPtr, ClearFreesImmediately) {
  DeleterTracker tracker;
  DataPtr dp(&tracker, &tracker, trackingDeleter, Device::cpu());
  EXPECT_EQ(tracker.call_count, 0);

  dp.clear();
  EXPECT_EQ(tracker.call_count, 1);  // freed by clear()
  EXPECT_EQ(dp.get(), nullptr);
  EXPECT_EQ(dp.get_context(), nullptr);
  EXPECT_EQ(dp.get_deleter(), nullptr);
}

// ============================================================================
// 17. clear nulls all fields
// ============================================================================
TEST(DataPtr, ClearNullsAllFields) {
  void* raw = std::malloc(8);
  DataPtr dp(raw, deleteCPU, Device::cpu());

  dp.clear();
  EXPECT_EQ(dp.get(), nullptr);
  EXPECT_EQ(dp.get_context(), nullptr);
  EXPECT_EQ(dp.get_deleter(), nullptr);
  EXPECT_FALSE(static_cast<bool>(dp));
}

// ============================================================================
// 18. Double clear is safe — no double free
// ============================================================================
TEST(DataPtr, DoubleClearIsSafe) {
  DeleterTracker tracker;
  DataPtr dp(&tracker, &tracker, trackingDeleter, Device::cpu());

  dp.clear();
  EXPECT_EQ(tracker.call_count, 1);

  dp.clear();  // second clear — should be a no-op
  EXPECT_EQ(tracker.call_count, 1);  // still 1, not 2
}

// ============================================================================
// 19. Device stored correctly — CPU
// ============================================================================
TEST(DataPtr, DeviceStoredCPU) {
  int dummy = 0;
  DataPtr dp(&dummy, deleteNothing, Device::cpu());
  EXPECT_TRUE(dp.device().is_cpu());
  EXPECT_FALSE(dp.device().is_cuda());
  EXPECT_EQ(dp.device().index, 0);
}

// ============================================================================
// 20. Device stored correctly — CUDA with index
// ============================================================================
TEST(DataPtr, DeviceStoredCUDA) {
  int dummy = 0;
  DataPtr dp(&dummy, deleteNothing, Device::cuda(3));
  EXPECT_TRUE(dp.device().is_cuda());
  EXPECT_FALSE(dp.device().is_cpu());
  EXPECT_EQ(dp.device().index, 3);
}

}  // namespace
