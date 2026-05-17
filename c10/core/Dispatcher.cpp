// ============================================================================
// Axion / c10 — Dispatcher implementation
// ============================================================================

#include "c10/core/Dispatcher.h"

namespace c10 {

Dispatcher& Dispatcher::singleton() {
  static Dispatcher instance;
  return instance;
}

}  // namespace c10
