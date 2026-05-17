// ============================================================================
// Axion / autograd — GradMode implementation
// ============================================================================

#include "autograd/GradMode.h"

namespace autograd {

thread_local bool GradMode::enabled_ = true;

bool GradMode::is_enabled() { return enabled_; }
void GradMode::set_enabled(bool enabled) { enabled_ = enabled; }

}  // namespace autograd
