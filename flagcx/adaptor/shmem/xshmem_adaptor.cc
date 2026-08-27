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
#include <mutex>
#include <xshmem/xshmem.h>
#include <xshmem/xshmemx.h>

// ============================================================
// Internal state for one devComm backed by XSHMEM
// ============================================================

// ============================================================
// Lifecycle
// ============================================================
namespace {

std::mutex g_shmemInitLock;
int g_shmemUseCount = 0;
bool g_shmemInitDone = false;
#ifdef USE_KUNLUNXIN_ADAPTOR
BKCLContext_t g_shmemInitCtx = nullptr;
#endif
int g_shmemMyPe = -1;
int g_shmemNPes = -1;

} // namespace

static flagcxResult_t xshmemAdaptorInit(int rank, int nranks, void *handle) {
#ifdef USE_KUNLUNXIN_ADAPTOR
  if (handle == nullptr)
    return flagcxInvalidArgument;
  BKCLContext_t ctx = (BKCLContext_t)handle;

  std::lock_guard<std::mutex> lock(g_shmemInitLock);

  if (!g_shmemInitDone) {
    if (xshmem_init(ctx) != 0)
      return flagcxInternalError;
    g_shmemInitDone = true;
    g_shmemInitCtx = ctx;
    g_shmemMyPe = xshmem_my_pe();
    g_shmemNPes = xshmem_n_pes();
  } else if (ctx != g_shmemInitCtx) {
    WARN("xshmem init: already initialized with a different BKCL context; "
         "xshmem has no finalize/re-init, so one process supports one world");
    return flagcxInvalidUsage;
  }

  if (g_shmemMyPe != rank || g_shmemNPes != nranks) {
    WARN("xshmem init: caller (rank %d/%d) does not match the initialized "
         "xshmem world (pe %d/%d)",
         rank, nranks, g_shmemMyPe, g_shmemNPes);
    return flagcxInvalidUsage;
  }

  ++g_shmemUseCount;
  return flagcxSuccess;
#else
  (void)rank;
  (void)nranks;
  (void)handle;
  return flagcxInternalError;
#endif
}

static flagcxResult_t xshmemAdaptorFinalize() {
  std::lock_guard<std::mutex> lock(g_shmemInitLock);
  if (g_shmemUseCount > 0)
    --g_shmemUseCount;
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
      WARN("xshmem devCommCreate: nRanks (%d) not divisible by intraSize (%d); "
           "non-uniform topologies are not supported",
           sc->nRanks, sc->intraSize);
      goto fail;
    }
    int interSize = (sc->intraSize > 0) ? sc->nRanks / sc->intraSize : 1;

    size_t gridSyncSize = 6 * FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
    if (cudaMalloc(&sc->gridSyncState, gridSyncSize) != cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->gridSyncState, 0, gridSyncSize);

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
static flagcxResult_t xshmemAdaptorDevCommDestroy(flagcxShmemComm_t shmemComm) {
  if (shmemComm == nullptr)
    return flagcxSuccess;

  if (shmemComm->signalBuffer)
    xshmem_free(shmemComm->signalBuffer);

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
