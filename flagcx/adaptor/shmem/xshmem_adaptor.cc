/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * XSHMEM Adaptor — implementation of flagcxShmemAdaptor_t for XSHMEM.
 * Manages XSHMEM lifecycle, symmetric heap allocations, and device comm
 * state (signals, counters, barriers, teams).
 ************************************************************************/

#include "xshmem_adaptor.h"
#include "shmem_adaptor.h"

#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include "kunlunxin_adaptor.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <xshmem/xshmem.h>
#include <xshmem/xshmemx.h>


// ============================================================
// Lifecycle: reference-counted init/finalize
// ============================================================
static int shmemInitRefCount = 0;

static flagcxResult_t xshmemAdaptorInit(flagcxComm_t comm) {
#ifdef USE_KUNLUNXIN_ADAPTOR
  if (comm == nullptr || comm->homoComm == nullptr)
    return flagcxInternalError;

  // xccl xshmem initializes against the BKCL context created by
  // bkcl_init_rank (stored in flagcxInnerComm::base). There is no no-arg
  // xshmem_init(), so this is the only way to obtain the context.
  BKCLContext_t ctx = comm->homoComm->base;
  if (xshmem_init(ctx) != 0)
    return flagcxInternalError;

  if (xshmem_my_pe() != comm->rank || xshmem_n_pes() != comm->nranks)
    return flagcxInternalError;
#else
  (void)comm;
  return flagcxInternalError; // xshmem requires the Kunlunxin BKCL context
#endif
  shmemInitRefCount++;
  return flagcxSuccess;
}

static flagcxResult_t xshmemAdaptorFinalize() {
  if (shmemInitRefCount > 0)
    --shmemInitRefCount;
  return flagcxSuccess;
}

// ============================================================
// Symmetric memory management
// ============================================================
static flagcxResult_t xshmemAdaptorMalloc(void **ptr, size_t size) {
  *ptr = xshmem_malloc(size);
  if (*ptr == nullptr)
    return flagcxSystemError;
  cudaMemset(*ptr, 0, size);
  return flagcxSuccess;
}

static flagcxResult_t xshmemAdaptorFree(void *ptr) {
  xshmem_free(ptr);
  return flagcxSuccess;
}

// ============================================================
// Device Comm Create
// ============================================================
static flagcxResult_t xshmemAdaptorDevCommDestroy(flagcxShmemComm_t shmemComm);

  static flagcxResult_t
  xshmemAdaptorDevCommCreate(flagcxComm_t comm,
                            const struct flagcxDevCommRequirements *reqs,
                            flagcxShmemComm_t *shmemComm) {
  auto *sc = new flagcxShmemCommInternal();
  memset(sc, 0, sizeof(*sc));
  sc->intraTeam = XSHMEM_TEAM_INVALID;
  sc->interTeam = XSHMEM_TEAM_INVALID;

  sc->rank = comm->rank;
  sc->nRanks = comm->nranks;
  sc->intraRank = comm->localRank;
  sc->intraSize = comm->localRanks;

  sc->signalCount = reqs->interSignalCount;
  sc->counterCount = reqs->interCounterCount;

  // Signal buffer (symmetric heap, remote-writable)
  if (sc->signalCount > 0) {
    sc->signalBuffer =
        (uint64_t *)xshmem_malloc(sc->signalCount * sizeof(uint64_t));
    if (!sc->signalBuffer) {
      delete sc;
      return flagcxSystemError;
    }
    cudaMemset(sc->signalBuffer, 0, sc->signalCount * sizeof(uint64_t));
  }

  // Counter buffer (local device memory)
  if (sc->counterCount > 0) {
    if (cudaMalloc(&sc->counterBuffer, sc->counterCount * sizeof(uint64_t)) !=
        cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->counterBuffer, 0, sc->counterCount * sizeof(uint64_t));
  }

  // Shadow buffer (local device memory)
  if (sc->signalCount > 0) {
    if (cudaMalloc(&sc->shadowBuffer, sc->signalCount * sizeof(uint64_t)) !=
        cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->shadowBuffer, 0, sc->signalCount * sizeof(uint64_t));
  }

  // Validate topology
  {
    if (sc->intraSize > 0 && sc->nRanks % sc->intraSize != 0) {
      WARN(
          "xshmem devCommCreate: nRanks (%d) not divisible by intraSize (%d); "
          "non-uniform topologies are not supported",
          sc->nRanks, sc->intraSize);
      goto fail;
    }
    int interSize = (sc->intraSize > 0) ? sc->nRanks / sc->intraSize : 1;

    // Grid sync state for multi-block barrier coordination
    // 3 barriers x (arrive[CTA_COUNT] + release[CTA_COUNT]) = 6*CTA_COUNT
    size_t gridSyncSize = 6 * FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
    if (cudaMalloc(&sc->gridSyncState, gridSyncSize) != cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->gridSyncState, 0, gridSyncSize);

    // Team assignment.
    (void)interSize;
    sc->intraTeam = XSHMEMX_TEAM_NODE;

    sc->interTeam = XSHMEM_TEAM_INVALID;

    sc->worldTeam = XSHMEM_TEAM_WORLD;
  }

  *shmemComm = sc;
  return flagcxSuccess;

fail:
  xshmemAdaptorDevCommDestroy(sc);
  return flagcxSystemError;
}

// ============================================================
// Device Comm Destroy
// ============================================================
static flagcxResult_t
xshmemAdaptorDevCommDestroy(flagcxShmemComm_t shmemComm) {
  if (shmemComm == nullptr)
    return flagcxSuccess;

  // Free symmetric heap allocations
  if (shmemComm->signalBuffer)
    xshmem_free(shmemComm->signalBuffer);

  // Free local device allocations
  if (shmemComm->counterBuffer)
    cudaFree(shmemComm->counterBuffer);
  if (shmemComm->shadowBuffer)
    cudaFree(shmemComm->shadowBuffer);
  if (shmemComm->gridSyncState)
    cudaFree(shmemComm->gridSyncState);

  delete shmemComm;
  return flagcxSuccess;
}

// ============================================================
// Global adaptor instance
// ============================================================
static flagcxShmemAdaptor_t xshmemAdaptorInstance = {
    .name = "xshmem",
    .init = xshmemAdaptorInit,
    .finalize = xshmemAdaptorFinalize,
    .malloc = xshmemAdaptorMalloc,
    .free = xshmemAdaptorFree,
    .devCommCreate = xshmemAdaptorDevCommCreate,
    .devCommDestroy = xshmemAdaptorDevCommDestroy,
};

flagcxShmemAdaptor_t *shmemAdaptor = &xshmemAdaptorInstance;
