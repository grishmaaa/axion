#pragma once

// ============================================================================
// Axion / c10 — Platform & Export Macros
// ============================================================================
//
// These macros handle:
//   1. DLL export/import on Windows (dllexport / dllimport)
//   2. Symbol visibility on Unix (__attribute__((visibility("default"))))
//   3. Convenience annotations used throughout the c10 library
//
// Usage:
//   class C10_API MyClass { ... };
//   C10_API void myFunction();

// ---------------------------------------------------------------------------
// Platform detection
// ---------------------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
#define C10_PLATFORM_WINDOWS
#elif defined(__APPLE__)
#define C10_PLATFORM_APPLE
#elif defined(__linux__)
#define C10_PLATFORM_LINUX
#endif

// ---------------------------------------------------------------------------
// Compiler detection
// ---------------------------------------------------------------------------
#if defined(__clang__)
#define C10_COMPILER_CLANG
#elif defined(__GNUC__)
#define C10_COMPILER_GCC
#elif defined(_MSC_VER)
#define C10_COMPILER_MSVC
#endif

// ---------------------------------------------------------------------------
// Export / import
// ---------------------------------------------------------------------------
#ifdef C10_PLATFORM_WINDOWS
#ifdef C10_BUILD_SHARED_LIBS
#ifdef C10_BUILD_MAIN_LIB
#define C10_EXPORT __declspec(dllexport)
#else
#define C10_EXPORT __declspec(dllimport)
#endif
#else
#define C10_EXPORT
#endif
#define C10_HIDDEN
#else  // Unix / macOS
#define C10_EXPORT __attribute__((visibility("default")))
#define C10_HIDDEN __attribute__((visibility("hidden")))
#endif

// Convenience alias — every public c10 symbol is tagged with this.
#define C10_API C10_EXPORT

// ---------------------------------------------------------------------------
// Utility macros
// ---------------------------------------------------------------------------

// Suppress unused-variable warnings.
#if defined(C10_COMPILER_GCC) || defined(C10_COMPILER_CLANG)
#define C10_UNUSED __attribute__((unused))
#elif defined(C10_COMPILER_MSVC)
#define C10_UNUSED __pragma(warning(suppress : 4100 4101))
#else
#define C10_UNUSED
#endif

// Hint to the compiler that a branch is likely/unlikely.
#if defined(C10_COMPILER_GCC) || defined(C10_COMPILER_CLANG)
#define C10_LIKELY(x) __builtin_expect(!!(x), 1)
#define C10_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define C10_LIKELY(x) (x)
#define C10_UNLIKELY(x) (x)
#endif

// Mark a class as non-copyable.
#define C10_DISABLE_COPY(Class) \
  Class(const Class&) = delete;  \
  Class& operator=(const Class&) = delete

// Mark a class as non-copyable and non-movable.
#define C10_DISABLE_COPY_AND_MOVE(Class) \
  C10_DISABLE_COPY(Class);               \
  Class(Class&&) = delete;               \
  Class& operator=(Class&&) = delete
