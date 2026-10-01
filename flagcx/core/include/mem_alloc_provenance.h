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
  // Captured at allocation time. Registration must not infer this later from
  // the current FLAGCX_VMM_ENABLE value because the environment can change.
  bool isVmm;
  // Number of live symmetric windows that may still expose this allocation
  // through an IPC/VMM mapping or network MR. flagcxMemFree must not release
  // the allocation until every window (including cleanup-required tokens) has
  // dropped its lease.
  size_t windowRefs;
};

#endif // FLAGCX_MEM_ALLOC_PROVENANCE_H_
