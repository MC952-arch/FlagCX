/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * SHMEM Adaptor Interface — struct of function pointers for SHMEM backends.
 * Implementations (e.g., nvshmem_adaptor.cc) fill these pointers.
 ************************************************************************/

#ifndef FLAGCX_SHMEM_ADAPTOR_H_
#define FLAGCX_SHMEM_ADAPTOR_H_

#include "flagcx.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for SHMEM-backed device comm state
struct flagcxShmemCommInternal;
typedef struct flagcxShmemCommInternal *flagcxShmemComm_t;

// Forward declarations
struct flagcxDevCommRequirements;
struct flagcxDevMemInternal;

struct flagcxShmemAdaptor {
  const char *name;

  // Lifecycle (reference-counted)
  flagcxResult_t (*init)(int rank, int nRanks);
  flagcxResult_t (*finalize)();

  // Symmetric memory management
  flagcxResult_t (*symMalloc)(void **ptr, size_t size);
  flagcxResult_t (*symFree)(void *ptr);

  // Device comm setup
  flagcxResult_t (*devCommCreate)(flagcxComm_t comm,
                                  const struct flagcxDevCommRequirements *reqs,
                                  flagcxShmemComm_t *shmemComm);
  flagcxResult_t (*devCommDestroy)(flagcxShmemComm_t shmemComm);

  // Device mem setup
  flagcxResult_t (*devMemCreate)(flagcxShmemComm_t shmemComm, void *buff,
                                 size_t size,
                                 struct flagcxDevMemInternal *devMem);
  flagcxResult_t (*devMemDestroy)(flagcxShmemComm_t shmemComm,
                                  struct flagcxDevMemInternal *devMem);
};

typedef struct flagcxShmemAdaptor flagcxShmemAdaptor_t;

// Global adaptor instance (set at load time)
extern flagcxShmemAdaptor_t *shmemAdaptor;

#ifdef __cplusplus
}
#endif

#endif // FLAGCX_SHMEM_ADAPTOR_H_
