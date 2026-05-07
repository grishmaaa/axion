// ============================================================================
// Tests for c10::intrusive_ptr<T>
// ============================================================================

#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "c10/util/intrusive_ptr.h"

// ---------------------------------------------------------------------------
// Test target class
// ---------------------------------------------------------------------------

namespace {

class TestObject : public c10::intrusive_ptr_target {
 public:
  explicit TestObject(int v = 0) : value(v) {}
  int value;
};

// A derived class — tests polymorphic conversions.
class DerivedObject : public TestObject {
 public:
  explicit DerivedObject(int v, float extra)
      : TestObject(v), extra_data(extra) {}
  float extra_data;
};

// An object that sets a flag on destruction — tests lifetime correctness.
class DestTracker : public c10::intrusive_ptr_target {
 public:
  explicit DestTracker(bool* flag) : destroyed(flag) { *destroyed = false; }
  ~DestTracker() override { *destroyed = true; }
  bool* destroyed;
};

}  // namespace

// ============================================================================
// Basic semantics
// ============================================================================

TEST(IntrusivePtr, DefaultConstructorIsNull) {
  c10::intrusive_ptr<TestObject> p;
  EXPECT_EQ(p.get(), nullptr);
  EXPECT_FALSE(static_cast<bool>(p));
  EXPECT_EQ(p.use_count(), 0u);
}

TEST(IntrusivePtr, MakeIntrusive) {
  auto p = c10::make_intrusive<TestObject>(42);
  ASSERT_NE(p.get(), nullptr);
  EXPECT_EQ(p->value, 42);
  EXPECT_EQ(p.use_count(), 1u);
  EXPECT_TRUE(p.unique());
}

TEST(IntrusivePtr, ExplicitRawPointer) {
  auto* raw = new TestObject(7);
  c10::intrusive_ptr<TestObject> p(raw);
  EXPECT_EQ(p.get(), raw);
  EXPECT_EQ(p->value, 7);
  EXPECT_EQ(p.use_count(), 1u);
}

// ============================================================================
// Copy semantics
// ============================================================================

TEST(IntrusivePtr, CopyIncrementsRefcount) {
  auto p1 = c10::make_intrusive<TestObject>(10);
  EXPECT_EQ(p1.use_count(), 1u);

  auto p2 = p1;  // copy
  EXPECT_EQ(p1.use_count(), 2u);
  EXPECT_EQ(p2.use_count(), 2u);
  EXPECT_EQ(p1.get(), p2.get());
}

TEST(IntrusivePtr, CopyAssignment) {
  auto p1 = c10::make_intrusive<TestObject>(1);
  auto p2 = c10::make_intrusive<TestObject>(2);

  TestObject* old_p2 = p2.get();
  p2 = p1;

  EXPECT_EQ(p1.use_count(), 2u);
  EXPECT_EQ(p2.get(), p1.get());
  // old_p2 object should have been destroyed (use_count was 1, now 0).
}

TEST(IntrusivePtr, SelfAssignmentIsSafe) {
  auto p = c10::make_intrusive<TestObject>(99);
  p = p;  // self-assign
  EXPECT_EQ(p.use_count(), 1u);
  EXPECT_EQ(p->value, 99);
}

// ============================================================================
// Move semantics
// ============================================================================

TEST(IntrusivePtr, MoveLeaveSourceNull) {
  auto p1 = c10::make_intrusive<TestObject>(5);
  auto* raw = p1.get();

  auto p2 = std::move(p1);
  EXPECT_EQ(p1.get(), nullptr);  // NOLINT: intentional use-after-move
  EXPECT_EQ(p2.get(), raw);
  EXPECT_EQ(p2.use_count(), 1u);
}

TEST(IntrusivePtr, MoveAssignment) {
  auto p1 = c10::make_intrusive<TestObject>(1);
  auto p2 = c10::make_intrusive<TestObject>(2);

  p2 = std::move(p1);
  EXPECT_EQ(p1.get(), nullptr);  // NOLINT
  EXPECT_EQ(p2->value, 1);
  EXPECT_EQ(p2.use_count(), 1u);
}

// ============================================================================
// Lifetime / destructor tracking
// ============================================================================

TEST(IntrusivePtr, DestructorCalledWhenLastRefDies) {
  bool destroyed = false;
  {
    auto p = c10::make_intrusive<DestTracker>(&destroyed);
    EXPECT_FALSE(destroyed);
    EXPECT_EQ(p.use_count(), 1u);
  }  // p goes out of scope
  EXPECT_TRUE(destroyed);
}

TEST(IntrusivePtr, DestructorNotCalledWhileRefsRemain) {
  bool destroyed = false;
  auto p1 = c10::make_intrusive<DestTracker>(&destroyed);
  {
    auto p2 = p1;  // copy
    EXPECT_EQ(p1.use_count(), 2u);
  }  // p2 dies — refcount drops to 1
  EXPECT_FALSE(destroyed);
  EXPECT_EQ(p1.use_count(), 1u);
}

// ============================================================================
// reset / release / swap
// ============================================================================

TEST(IntrusivePtr, Reset) {
  bool destroyed = false;
  auto p = c10::make_intrusive<DestTracker>(&destroyed);
  p.reset();
  EXPECT_TRUE(destroyed);
  EXPECT_EQ(p.get(), nullptr);
}

TEST(IntrusivePtr, ResetWithNewTarget) {
  bool d1 = false, d2 = false;
  auto p = c10::make_intrusive<DestTracker>(&d1);
  auto* raw2 = new DestTracker(&d2);

  p.reset(raw2);
  EXPECT_TRUE(d1);    // old object destroyed
  EXPECT_FALSE(d2);   // new object alive
  EXPECT_EQ(p.get(), raw2);
}

TEST(IntrusivePtr, Release) {
  auto p = c10::make_intrusive<TestObject>(42);
  auto* raw = p.release();
  EXPECT_EQ(p.get(), nullptr);
  EXPECT_EQ(raw->value, 42);
  // Manually clean up — we took ownership away from the smart pointer.
  // The refcount is still 1, so we wrap it again to trigger cleanup.
  c10::intrusive_ptr<TestObject> cleanup(raw);
  // cleanup's destructor will delete it.
}

TEST(IntrusivePtr, Swap) {
  auto p1 = c10::make_intrusive<TestObject>(1);
  auto p2 = c10::make_intrusive<TestObject>(2);
  auto* raw1 = p1.get();
  auto* raw2 = p2.get();

  p1.swap(p2);
  EXPECT_EQ(p1.get(), raw2);
  EXPECT_EQ(p2.get(), raw1);
}

TEST(IntrusivePtr, FreeStandingSwap) {
  auto p1 = c10::make_intrusive<TestObject>(10);
  auto p2 = c10::make_intrusive<TestObject>(20);
  auto* raw1 = p1.get();
  auto* raw2 = p2.get();

  swap(p1, p2);  // ADL
  EXPECT_EQ(p1.get(), raw2);
  EXPECT_EQ(p2.get(), raw1);
}

// ============================================================================
// Comparisons
// ============================================================================

TEST(IntrusivePtr, EqualityComparison) {
  auto p1 = c10::make_intrusive<TestObject>(1);
  auto p2 = p1;
  auto p3 = c10::make_intrusive<TestObject>(1);

  EXPECT_EQ(p1, p2);     // same object
  EXPECT_NE(p1, p3);     // different objects (even if same value)
}

TEST(IntrusivePtr, NullptrComparison) {
  c10::intrusive_ptr<TestObject> null;
  auto live = c10::make_intrusive<TestObject>(0);

  EXPECT_EQ(null, nullptr);
  EXPECT_NE(live, nullptr);
}

// ============================================================================
// Polymorphic (derived → base) conversions
// ============================================================================

TEST(IntrusivePtr, DerivedToBaseCopy) {
  auto derived = c10::make_intrusive<DerivedObject>(5, 3.14f);
  c10::intrusive_ptr<TestObject> base = derived;  // generalized copy

  EXPECT_EQ(base.use_count(), 2u);
  EXPECT_EQ(base->value, 5);
}

TEST(IntrusivePtr, DerivedToBaseMove) {
  auto derived = c10::make_intrusive<DerivedObject>(7, 2.71f);
  c10::intrusive_ptr<TestObject> base = std::move(derived);  // generalized move

  EXPECT_EQ(derived.get(), nullptr);  // NOLINT
  EXPECT_EQ(base.use_count(), 1u);
  EXPECT_EQ(base->value, 7);
}

// ============================================================================
// weak_intrusive_ptr basics
// ============================================================================

TEST(WeakIntrusivePtr, DefaultIsExpired) {
  c10::weak_intrusive_ptr<TestObject> w;
  EXPECT_TRUE(w.expired());
  EXPECT_EQ(w.use_count(), 0u);
  auto locked = w.lock();
  EXPECT_EQ(locked.get(), nullptr);
}

TEST(WeakIntrusivePtr, LockSucceedsWhileStrongExists) {
  auto strong = c10::make_intrusive<TestObject>(100);
  c10::weak_intrusive_ptr<TestObject> weak(strong);

  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(weak.use_count(), 1u);

  auto promoted = weak.lock();
  ASSERT_NE(promoted.get(), nullptr);
  EXPECT_EQ(promoted->value, 100);
  EXPECT_EQ(strong.use_count(), 2u);  // strong + promoted
}

TEST(WeakIntrusivePtr, LockFailsAfterStrongDies) {
  c10::weak_intrusive_ptr<TestObject> weak;
  {
    auto strong = c10::make_intrusive<TestObject>(42);
    weak = c10::weak_intrusive_ptr<TestObject>(strong);
    EXPECT_FALSE(weak.expired());
  }  // strong dies
  EXPECT_TRUE(weak.expired());
  auto promoted = weak.lock();
  EXPECT_EQ(promoted.get(), nullptr);
}

TEST(WeakIntrusivePtr, CopySemantics) {
  auto strong = c10::make_intrusive<TestObject>(1);
  c10::weak_intrusive_ptr<TestObject> w1(strong);
  auto w2 = w1;  // copy

  EXPECT_EQ(w1.use_count(), 1u);
  EXPECT_EQ(w2.use_count(), 1u);
  EXPECT_FALSE(w1.expired());
  EXPECT_FALSE(w2.expired());
}

TEST(WeakIntrusivePtr, MoveSemantics) {
  auto strong = c10::make_intrusive<TestObject>(2);
  c10::weak_intrusive_ptr<TestObject> w1(strong);
  auto w2 = std::move(w1);

  EXPECT_FALSE(w2.expired());
  // w1 is moved-from — its target_ is null
}

// ============================================================================
// Thread safety (basic smoke test)
// ============================================================================

TEST(IntrusivePtr, ConcurrentCopyAndDestroy) {
  // Create an object and hammer it with concurrent copies/destroys.
  auto original = c10::make_intrusive<TestObject>(0);
  constexpr int kThreads = 8;
  constexpr int kIters = 10000;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&original]() {
      for (int i = 0; i < kIters; ++i) {
        auto copy = original;  // increment
        EXPECT_GE(copy.use_count(), 2u);
      }  // decrement
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  // After all threads complete, only `original` should remain.
  EXPECT_EQ(original.use_count(), 1u);
}
