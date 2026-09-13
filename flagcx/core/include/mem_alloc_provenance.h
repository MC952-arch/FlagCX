/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Plain allocation provenance types shared by host lifecycle code. Keep this
 * header free of STL types so device host compilers can parse it cheaply.
 ************************************************************************/

#ifndef FLAGCX_MEM_ALLOC_PROVENANCE_H_
#define FLAGCX_MEM_ALLOC_PROVENANCE_H_

#include "flagcx.h"

typedef enum {
  // Native platform allocation. This currently maps to the device adaptor's
  // gdrMemAlloc/gdrMemFree callbacks for heterogeneous communicators.
  flagcxMemAllocBackendNative = 0,
  flagcxMemAllocBackendCcl = 1,
  flagcxMemAllocBackendShmem = 2,
} flagcxMemAllocBackend_t;

struct flagcxMemAllocationInfo {
  void *base;
  size_t size;
  flagcxMemAllocator_t allocator;
  flagcxMemAllocBackend_t backend;
};

#endif // FLAGCX_MEM_ALLOC_PROVENANCE_H_
