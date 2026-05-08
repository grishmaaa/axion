// ============================================================================
// Axion / c10 — StorageImpl implementation
// ============================================================================

#include "c10/core/StorageImpl.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace c10 {

// ----------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------

StorageImpl::StorageImpl(size_t nbytes, Allocator* allocator)
    : data_(allocator ? allocator->allocate(nbytes) : DataPtr()),
      nbytes_(nbytes),
      allocator_(allocator),
      resizable_(true) {}

StorageImpl::StorageImpl(size_t nbytes, DataPtr data, Allocator* allocator)
    : data_(std::move(data)),
      nbytes_(nbytes),
      allocator_(allocator),
      resizable_(allocator != nullptr) {}

// ----------------------------------------------------------------------------
// Move semantics
// ----------------------------------------------------------------------------

StorageImpl::StorageImpl(StorageImpl&& other) noexcept
    : intrusive_ptr_target(std::move(other)),
      data_(std::move(other.data_)),
      nbytes_(other.nbytes_),
      allocator_(other.allocator_),
      resizable_(other.resizable_) {
  other.nbytes_ = 0;
  other.allocator_ = nullptr;
  other.resizable_ = false;
}

StorageImpl& StorageImpl::operator=(StorageImpl&& other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    nbytes_ = other.nbytes_;
    allocator_ = other.allocator_;
    resizable_ = other.resizable_;
    other.nbytes_ = 0;
    other.allocator_ = nullptr;
    other.resizable_ = false;
  }
  return *this;
}

// ----------------------------------------------------------------------------
// Resize
// ----------------------------------------------------------------------------

void StorageImpl::resize(size_t new_nbytes) {
  assert(resizable_ && "Cannot resize non-resizable storage");
  assert(allocator_ && "Cannot resize storage without an allocator");

  if (new_nbytes == nbytes_) {
    return;  // no-op
  }

  // Allocate new buffer.
  DataPtr new_data = allocator_->allocate(new_nbytes);

  // Copy existing data (up to the smaller of old and new sizes).
  if (data_.get() && new_data.get()) {
    size_t copy_bytes = std::min(nbytes_, new_nbytes);
    std::memcpy(new_data.get(), data_.get(), copy_bytes);
  }

  // Replace — old DataPtr's destructor frees the old memory.
  data_ = std::move(new_data);
  nbytes_ = new_nbytes;
}

}  // namespace c10
