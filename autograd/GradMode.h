#pragma once

// ============================================================================
// Axion / autograd — GradMode
// ============================================================================
//
// Thread-local flag controlling whether ops record onto the autograd
// graph.  Use NoGradGuard to temporarily disable recording (e.g.
// during inference or inside backward()).

namespace autograd {

class GradMode {
 public:
  static bool is_enabled();
  static void set_enabled(bool enabled);

 private:
  static thread_local bool enabled_;
};

/// RAII guard that disables gradient recording for its lifetime.
///
///   {
///     autograd::NoGradGuard no_grad;
///     auto y = aten::add(x, x);  // NOT recorded on the graph
///   }
///
class NoGradGuard {
 public:
  NoGradGuard() : prev_(GradMode::is_enabled()) {
    GradMode::set_enabled(false);
  }
  ~NoGradGuard() { GradMode::set_enabled(prev_); }

  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool prev_;
};

}  // namespace autograd
