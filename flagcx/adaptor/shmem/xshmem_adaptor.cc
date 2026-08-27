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
#include "kunlunxin_adaptor.h" // flagcxInnerComm::base (BKCLContext_t)

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
// xccl xshmem has NO xshmem_finalize() and no init-status query: once
// xshmem_init(ctx) succeeds, the runtime and its symmetric heap live for the
// whole process. (`destroy_heap()` exists but is not a finalize: it would
// invalidate every symmetric pointer still held by live devComms/kernels, and
// there is no supported re-init path afterwards.)
//
// So this adaptor deliberately implements ONE PROCESS-LIFETIME init:
//   - xshmem_init() is called exactly once, under a lock, by the first caller;
//   - later callers only validate that they describe the SAME world and bump a
//     use count;
//   - finalize() drops the use count and, by design, tears nothing down.
// The use count is therefore diagnostic, not a destruction trigger -- it must
// not pretend otherwise. This keeps repeated devComm create/destroy cycles
// (e.g. test_multi_fifo) safe instead of re-entering xshmem_init every time.
namespace {

std::mutex g_shmemInitLock;
int g_shmemUseCount = 0;      // live users (does NOT drive teardown)
bool g_shmemInitDone = false; // xshmem_init(ctx) has been attempted+succeeded
#ifdef USE_KUNLUNXIN_ADAPTOR
BKCLContext_t g_shmemInitCtx = nullptr; // ctx the runtime was initialized with
#endif
int g_shmemMyPe = -1;
int g_shmemNPes = -1;

} // namespace

static flagcxResult_t xshmemAdaptorInit(flagcxComm_t comm) {
#ifdef USE_KUNLUNXIN_ADAPTOR
  if (comm == nullptr || comm->homoComm == nullptr)
    return flagcxInternalError;

  // xccl xshmem initializes against the BKCL context created by
  // bkcl_init_rank (stored in flagcxInnerComm::base). There is no no-arg
  // xshmem_init(), so this is the only way to obtain the context.
  BKCLContext_t ctx = comm->homoComm->base;

  std::lock_guard<std::mutex> lock(g_shmemInitLock);

  if (!g_shmemInitDone) {
    if (xshmem_init(ctx) != 0)
      return flagcxInternalError; // nothing was claimed; state untouched
    // The runtime cannot be un-initialized, so record it even if the checks
    // below reject THIS comm -- otherwise a later call would init twice.
    g_shmemInitDone = true;
    g_shmemInitCtx = ctx;
    g_shmemMyPe = xshmem_my_pe();
    g_shmemNPes = xshmem_n_pes();
  } else if (ctx != g_shmemInitCtx) {
    WARN("xshmem init: already initialized with a different BKCL context; "
         "xshmem has no finalize/re-init, so one process supports one world");
    return flagcxInvalidUsage;
  }

  // Reject a comm that does not describe the initialized world. No cleanup is
  // possible (nor needed): we claim no use count and leave the runtime as the
  // first successful init left it.
  if (g_shmemMyPe != comm->rank || g_shmemNPes != comm->nranks) {
    WARN("xshmem init: comm (rank %d/%d) does not match the initialized xshmem "
         "world (pe %d/%d)",
         comm->rank, comm->nranks, g_shmemMyPe, g_shmemNPes);
    return flagcxInvalidUsage;
  }

  ++g_shmemUseCount;
  return flagcxSuccess;
#else
  (void)comm;
  return flagcxInternalError; // xshmem requires the Kunlunxin BKCL context
#endif
}

static flagcxResult_t xshmemAdaptorFinalize() {
  std::lock_guard<std::mutex> lock(g_shmemInitLock);
  if (g_shmemUseCount > 0)
    --g_shmemUseCount;
  // Intentionally no teardown on the last reference: see the note above.
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

    // Grid sync state for multi-block barrier coordination
    // 3 barriers x (arrive[CTA_COUNT] + release[CTA_COUNT]) = 6*CTA_COUNT
    size_t gridSyncSize = 6 * FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
    if (cudaMalloc(&sc->gridSyncState, gridSyncSize) != cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->gridSyncState, 0, gridSyncSize);

    // Team assignment.
    // xccl xshmem provides no dynamic team creation (no team_split_strided);
    // only 4 fixed teams exist (WORLD/SHARED/NODE/SAME_MYPE_NODE, see
    // xshmem_common.h). Map intra-node to the built-in XSHMEMX_TEAM_NODE.
    (void)interSize;
    sc->intraTeam = XSHMEMX_TEAM_NODE;

    // No inter-node sub-team is available in xshmem, and device-side barrier
    // only supports XSHMEM_TEAM_WORLD (see non_abi/.../coll/barrier.h). Leave
    // interTeam invalid; cross-node collectives must go through WORLD.
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

  // Teams are fixed built-ins in xccl xshmem; there is no xshmem_team_destroy.

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
