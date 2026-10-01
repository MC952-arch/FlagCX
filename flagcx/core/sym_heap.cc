/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Symmetric memory coordination for the default (non-vendor) path.
 * Implements VMM-based flat VA mapping with IPC fallback.
 ************************************************************************/

#include "sym_heap.h"
#include "adaptor.h"
#include "alloc.h"
#include "bootstrap.h"
#include "check.h"
#include "comm.h"
#include "global_comm.h"
#include "ipcsocket.h"
#include "mem_alloc_registry.h"
#include "onesided.h"
#include "param.h"
#include "transport.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

constexpr int kSymStatusGatherTag = 0x5950;
constexpr int kSymStatusBroadcastTag = 0x5951;
constexpr int kSymPendingGatherTag = 0x5952;
constexpr int kSymPendingBroadcastTag = 0x5953;
constexpr int kSymNetworkRouteGatherTag = 0x5954;
constexpr int kSymNetworkRouteBroadcastTag = 0x5955;

enum flagcxSymScalarReduction {
  flagcxSymFirstError,
  flagcxSymLogicalOr,
};

// Symmetric-memory rollback may be entered because a rank-local allocation
// failed. Keep the convergence primitive allocation-free so reporting that
// failure cannot itself strand peers in a collective. Rank 0 gathers one
// scalar from each peer and broadcasts the deterministic result.
static flagcxResult_t flagcxSymGlobalConvergeScalar(
    flagcxHeteroComm_t comm, int localValue, int gatherTag, int broadcastTag,
    flagcxSymScalarReduction reduction, int *commonValue) {
  if (comm == nullptr || comm->bootstrap == nullptr || commonValue == nullptr ||
      comm->rank < 0 || comm->rank >= comm->nRanks || comm->nRanks <= 0)
    return flagcxInvalidArgument;

  constexpr int root = 0;
  int common = localValue;
  if (comm->rank == root) {
    for (int peer = 1; peer < comm->nRanks; peer++) {
      int peerValue = 0;
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, peer, gatherTag, &peerValue,
                                sizeof(peerValue)));
      if (reduction == flagcxSymLogicalOr) {
        common = common != 0 || peerValue != 0;
      } else if (common == flagcxSuccess && peerValue != flagcxSuccess) {
        // Peers are visited in rank order, preserving the existing
        // lowest-failing-rank error selection.
        common = peerValue;
      }
    }
    for (int peer = 1; peer < comm->nRanks; peer++) {
      FLAGCXCHECK(bootstrapSend(comm->bootstrap, peer, broadcastTag, &common,
                                sizeof(common)));
    }
  } else {
    FLAGCXCHECK(bootstrapSend(comm->bootstrap, root, gatherTag, &localValue,
                              sizeof(localValue)));
    FLAGCXCHECK(bootstrapRecv(comm->bootstrap, root, broadcastTag, &common,
                              sizeof(common)));
  }

  *commonValue = common;
  return flagcxSuccess;
}

} // namespace

static flagcxResult_t flagcxSymLocalConverge(flagcxHeteroComm_t comm,
                                             flagcxResult_t localStatus,
                                             int tag,
                                             flagcxResult_t *commonStatus) {
  if (comm == nullptr || comm->bootstrap == nullptr ||
      commonStatus == nullptr || comm->localRanks <= 0 || comm->localRank < 0 ||
      comm->localRank >= comm->localRanks || comm->localRankToRank == nullptr)
    return flagcxInvalidArgument;
  int local = static_cast<int>(localStatus);
  int common = local;
  constexpr int localRoot = 0;
  int root = comm->localRankToRank[localRoot];
  if (comm->localRank == localRoot) {
    for (int i = 1; i < comm->localRanks; i++) {
      int peerStatus = static_cast<int>(flagcxSuccess);
      int peer = comm->localRankToRank[i];
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, peer, tag, &peerStatus,
                                sizeof(peerStatus)));
      if (common == flagcxSuccess && peerStatus != flagcxSuccess)
        common = peerStatus;
    }
    for (int i = 1; i < comm->localRanks; i++) {
      int peer = comm->localRankToRank[i];
      FLAGCXCHECK(
          bootstrapSend(comm->bootstrap, peer, tag, &common, sizeof(common)));
    }
  } else {
    FLAGCXCHECK(
        bootstrapSend(comm->bootstrap, root, tag, &local, sizeof(local)));
    FLAGCXCHECK(
        bootstrapRecv(comm->bootstrap, root, tag, &common, sizeof(common)));
  }
  // Local rank zero receives in local-rank order, so different simultaneous
  // failures still converge to one deterministic error on every peer.
  *commonStatus = static_cast<flagcxResult_t>(common);
  return flagcxSuccess;
}

static flagcxResult_t flagcxSymGlobalConverge(flagcxHeteroComm_t comm,
                                              flagcxResult_t localStatus,
                                              flagcxResult_t *commonStatus) {
  if (commonStatus == nullptr)
    return flagcxInvalidArgument;
  int common = static_cast<int>(flagcxSuccess);
  FLAGCXCHECK(flagcxSymGlobalConvergeScalar(
      comm, static_cast<int>(localStatus), kSymStatusGatherTag,
      kSymStatusBroadcastTag, flagcxSymFirstError, &common));
  *commonStatus = static_cast<flagcxResult_t>(common);
  return flagcxSuccess;
}

flagcxResult_t flagcxSymConvergeStatus(flagcxHeteroComm_t comm,
                                       flagcxResult_t localStatus,
                                       flagcxResult_t *commonStatus) {
  return flagcxSymGlobalConverge(comm, localStatus, commonStatus);
}

static bool flagcxSymWindowHasValidNetworkMr(flagcxHeteroComm_t comm,
                                             flagcxSymWindow_t window) {
  if (comm == nullptr || window == nullptr || !window->hasNetworkMrRef ||
      window->mrIndex < 0 || window->mrIndex >= comm->oneSideHandleCount ||
      comm->oneSideHandles == nullptr)
    return false;
  struct flagcxOneSideHandleInfo *handle =
      comm->oneSideHandles[window->mrIndex];
  return handle != nullptr && handle->baseVas != nullptr &&
         handle->regionSizes != nullptr && comm->rank >= 0 &&
         comm->rank < comm->nRanks &&
         handle->baseVas[comm->rank] == (uintptr_t)window->localBase &&
         handle->regionSizes[comm->rank] == window->heapSize;
}

flagcxResult_t flagcxSymWindowEnsureNetworkMr(flagcxHeteroComm_t comm,
                                              flagcxSymWindow_t window) {
  if (comm == nullptr || window == nullptr || window->localBase == nullptr ||
      window->heapSize == 0)
    return flagcxInvalidArgument;
  if (window->hasNetworkMrRef)
    return flagcxSymWindowHasValidNetworkMr(comm, window) ? flagcxSuccess
                                                          : flagcxInternalError;

  int mrIndex = -1;
  bool rollbackPending = false;
  flagcxResult_t result = flagcxOneSideRegisterInternal(
      comm, window->localBase, window->heapSize, window->allocationIsVmm,
      /*acquireWindowRef=*/true, &mrIndex, &rollbackPending);
  if (result != flagcxSuccess) {
    window->hasPendingNetworkCleanup = rollbackPending;
    return result;
  }

  // Take ownership before validating the published metadata. If validation
  // detects an internal inconsistency, the caller's normal rollback path must
  // still release (or retain for retry) the acquired MR reference.
  window->mrIndex = mrIndex;
  window->hasNetworkMrRef = true;
  if (!flagcxSymWindowHasValidNetworkMr(comm, window))
    return flagcxInternalError;
  window->mrBase = comm->oneSideHandles[mrIndex]->baseVas[comm->rank];
  return flagcxSuccess;
}

flagcxResult_t
flagcxSymWindowValidateDataRoutesForMode(flagcxHeteroComm_t comm,
                                         flagcxSymWindow_t window,
                                         bool localPeerTransportEnabled) {
  if (comm == nullptr || window == nullptr || comm->nRanks <= 0 ||
      comm->localRanks <= 0 || comm->localRanks > comm->nRanks)
    return flagcxInvalidArgument;

  const bool hasNetworkMr = flagcxSymWindowHasValidNetworkMr(comm, window);
  const bool needsLocalPeerRoute = comm->localRanks > 1;
  const bool hasLocalPeerRoute =
      !needsLocalPeerRoute || hasNetworkMr ||
      (localPeerTransportEnabled &&
       ((window->isVMM && window->flatBase != nullptr) ||
        window->ipcSlot >= 0));
  const bool hasRemotePeerRoute =
      comm->localRanks == comm->nRanks || hasNetworkMr;
  return hasLocalPeerRoute && hasRemotePeerRoute ? flagcxSuccess
                                                 : flagcxNotSupported;
}

flagcxResult_t flagcxSymWindowValidateDataRoutes(flagcxHeteroComm_t comm,
                                                 flagcxSymWindow_t window) {
  return flagcxSymWindowValidateDataRoutesForMode(comm, window,
                                                  !flagcxParamP2pDisable());
}

static flagcxResult_t flagcxSymCleanupStepConverge(flagcxHeteroComm_t comm,
                                                   flagcxResult_t local,
                                                   int tag,
                                                   flagcxSymCleanupMode mode,
                                                   flagcxResult_t *common) {
  if (common == nullptr)
    return flagcxInvalidArgument;
  if (mode == flagcxSymCleanupCollective && comm != nullptr &&
      comm->bootstrap != nullptr && comm->localRanks > 1)
    return flagcxSymLocalConverge(comm, local, tag, common);
  *common = local;
  return flagcxSuccess;
}

static flagcxResult_t flagcxSymCleanupMappings(flagcxHeteroComm_t comm,
                                               flagcxSymWindow_t d,
                                               flagcxSymCleanupMode mode) {
  const bool hasVmmOwnership =
      d != nullptr && (d->flatMappingOwned || d->flatVaOwned ||
                       d->multicastMappingOwned || d->multicastVaOwned ||
                       d->mcHandle != nullptr || d->physHandle != nullptr);
  if (hasVmmOwnership && deviceAdaptor == nullptr)
    return flagcxInternalError;

  flagcxResult_t local = flagcxSuccess;
  flagcxResult_t common = flagcxSuccess;

  if (hasVmmOwnership && d->multicastMappingOwned) {
    local =
        deviceAdaptor->symMulticastMappingUnmap != nullptr
            ? deviceAdaptor->symMulticastMappingUnmap(d->mcBase, d->mcMapSize)
            : flagcxInternalError;
    if (local == flagcxSuccess)
      d->multicastMappingOwned = false;
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5941, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->multicastVaOwned) {
    local = deviceAdaptor->symMulticastVaFree != nullptr
                ? deviceAdaptor->symMulticastVaFree(d->mcBase, d->mcMapSize)
                : flagcxInternalError;
    if (local == flagcxSuccess) {
      d->multicastVaOwned = false;
      d->mcBase = nullptr;
      d->mcMapSize = 0;
    }
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5942, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->mcHandle != nullptr) {
    local = deviceAdaptor->symMulticastFree != nullptr
                ? deviceAdaptor->symMulticastFree(d->mcHandle)
                : flagcxInternalError;
    if (local == flagcxSuccess)
      d->mcHandle = nullptr;
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5943, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->flatMappingOwned) {
    local = deviceAdaptor->symFlatMappingUnmap != nullptr
                ? deviceAdaptor->symFlatMappingUnmap(d->flatBase, d->allocSize,
                                                     d->localRanks)
                : flagcxInternalError;
    if (local == flagcxSuccess)
      d->flatMappingOwned = false;
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5944, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->flatVaOwned) {
    local = deviceAdaptor->symFlatVaFree != nullptr
                ? deviceAdaptor->symFlatVaFree(d->flatBase, d->allocSize,
                                               d->localRanks)
                : flagcxInternalError;
    if (local == flagcxSuccess) {
      d->flatVaOwned = false;
      d->flatBase = nullptr;
    }
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5945, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->physHandle != nullptr) {
    local = deviceAdaptor->symPhysFree != nullptr
                ? deviceAdaptor->symPhysFree(d->physHandle)
                : flagcxInternalError;
    if (local == flagcxSuccess)
      d->physHandle = nullptr;
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5946, mode, &common));
  if (common == flagcxSuccess && d != nullptr && !d->flatMappingOwned &&
      !d->flatVaOwned && !d->multicastMappingOwned && !d->multicastVaOwned &&
      d->mcHandle == nullptr && d->physHandle == nullptr) {
    d->isVMM = false;
    d->allocSize = 0;
  }
  return common;
}

static flagcxResult_t flagcxSymReleaseAllocationLease(flagcxSymWindow_t d) {
  if (d == nullptr || !d->hasAllocationLease)
    return flagcxSuccess;
  flagcxResult_t result =
      globalMemAllocRegistry.releaseWindow(d->allocationBase);
  if (result == flagcxSuccess) {
    d->hasAllocationLease = false;
    d->allocationBase = nullptr;
  }
  return result;
}

// Mapping teardown is ordered among local peers because multicast and flat VA
// ownership are node-local. Public deregistration is nevertheless a
// communicator-wide collective: no rank may detach/free its window until all
// nodes report that their local teardown completed. Communicator destruction
// uses local mode and deliberately skips this rendezvous.
static flagcxResult_t
flagcxSymCleanupMappingsConverged(flagcxHeteroComm_t comm, flagcxSymWindow_t d,
                                  flagcxSymCleanupMode mode) {
  flagcxResult_t local = flagcxSymCleanupMappings(comm, d, mode);
  if (mode == flagcxSymCleanupLocal)
    return local;

  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxSymGlobalConverge(comm, local, &common));
  return common;
}

static flagcxResult_t flagcxSymCleanupUnpublished(flagcxHeteroComm_t comm,
                                                  flagcxSymWindow_t d,
                                                  flagcxSymCleanupMode mode) {
  if (comm == nullptr)
    return flagcxInvalidArgument;
  flagcxResult_t local = flagcxSuccess;
  if (d != nullptr && d->hasPendingNetworkCleanup) {
    local = flagcxOneSideRetryPendingCleanup(comm);
    if (local == flagcxSuccess)
      d->hasPendingNetworkCleanup = false;
  } else if (d != nullptr && d->hasNetworkMrRef && d->mrIndex >= 0) {
    local = flagcxOneSideDeregisterInternal(comm, d->mrIndex);
    if (local == flagcxSuccess) {
      d->hasNetworkMrRef = false;
      d->mrIndex = -1;
      d->mrBase = 0;
    }
  }
  flagcxResult_t common = flagcxSuccess;
  if (mode == flagcxSymCleanupCollective) {
    FLAGCXCHECK(flagcxSymGlobalConverge(comm, local, &common));
  } else {
    common = local;
  }
  if (common != flagcxSuccess)
    return common;

  return flagcxSymCleanupMappingsConverged(comm, d, mode);
}

flagcxResult_t flagcxSymRetryPendingCleanup(flagcxHeteroComm_t comm,
                                            flagcxSymCleanupMode mode) {
  if (comm == nullptr)
    return flagcxInvalidArgument;
  if (mode == flagcxSymCleanupLocal) {
    while (comm->pendingSymCleanup != nullptr) {
      flagcxSymWindow_t d = comm->pendingSymCleanup;
      FLAGCXCHECK(flagcxSymCleanupUnpublished(comm, d, mode));
      FLAGCXCHECK(flagcxSymReleaseAllocationLease(d));
      comm->pendingSymCleanup = d->cleanupNext;
      flagcxWindow_t owner = d->owner;
      free(d);
      free(owner);
    }
    return flagcxSuccess;
  }

  // Rollback ownership can be asymmetric: only a rank whose local teardown
  // failed needs an entry. Every rank still participates in each retry round
  // so MR and local-mapping status convergence cannot deadlock.
  while (true) {
    int anyPending = 0;
    FLAGCXCHECK(flagcxSymGlobalConvergeScalar(
        comm, comm->pendingSymCleanup != nullptr ? 1 : 0, kSymPendingGatherTag,
        kSymPendingBroadcastTag, flagcxSymLogicalOr, &anyPending));
    if (!anyPending)
      break;

    flagcxSymWindow_t d = comm->pendingSymCleanup;
    FLAGCXCHECK(flagcxSymCleanupUnpublished(comm, d, mode));
    if (d != nullptr) {
      FLAGCXCHECK(flagcxSymReleaseAllocationLease(d));
      comm->pendingSymCleanup = d->cleanupNext;
      flagcxWindow_t owner = d->owner;
      free(d);
      free(owner);
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSymRetainPendingCleanup(flagcxHeteroComm_t comm,
                                             flagcxWindow_t win) {
  if (comm == nullptr || win == nullptr || win->defaultBase == nullptr)
    return flagcxInvalidArgument;
  flagcxSymWindow_t d = win->defaultBase;
  if (d->published)
    return flagcxInvalidUsage;
  for (flagcxSymWindow_t pending = comm->pendingSymCleanup; pending != nullptr;
       pending = pending->cleanupNext) {
    if (pending == d)
      return flagcxSuccess;
  }
  d->cleanupNext = comm->pendingSymCleanup;
  comm->pendingSymCleanup = d;
  d->state = flagcxSymWindowCleanupRequired;
  return flagcxSuccess;
}

flagcxResult_t flagcxSymWindowRegisterInternal(flagcxHeteroComm_t comm,
                                               void *buff, size_t size,
                                               flagcxWindow_t *win,
                                               int winFlags,
                                               bool allocationIsVmm) {
  if (comm == nullptr || buff == nullptr || size == 0 || win == nullptr)
    return flagcxInvalidArgument;
  *win = nullptr;
  // A cleanup token represents an unfinished collective transaction. Starting
  // another registration could give ranks different window/MR slot numbers,
  // so all ranks reject it until the application retries deregistration.
  int anyPendingCleanup = 0;
  FLAGCXCHECK(flagcxSymGlobalConvergeScalar(
      comm, comm->pendingSymCleanup != nullptr ? 1 : 0, kSymPendingGatherTag,
      kSymPendingBroadcastTag, flagcxSymLogicalOr, &anyPendingCleanup));
  if (anyPendingCleanup)
    return flagcxInvalidUsage;

  // Flat VMM and IPC mappings are node-local and the runtime deliberately
  // bypasses both when P2P is disabled. Converge whether any rank needs NET so
  // every rank takes the same collective MR path even if its local environment
  // is inconsistent.
  int needsNetworkMr = 0;
  FLAGCXCHECK(flagcxSymGlobalConvergeScalar(
      comm,
      comm->localRanks < comm->nRanks ||
              (comm->localRanks > 1 && flagcxParamP2pDisable())
          ? 1
          : 0,
      kSymNetworkRouteGatherTag, kSymNetworkRouteBroadcastTag,
      flagcxSymLogicalOr, &needsNetworkMr));

  // A required network route cannot be published while NET is disabled.
  // Converge this prerequisite before any rank enters VMM FD or MR metadata
  // exchange; this also handles inconsistent rank-local environments without
  // splitting collective control flow.
  flagcxResult_t localRouteStatus = needsNetworkMr && flagcxParamIbDisable()
                                        ? flagcxNotSupported
                                        : flagcxSuccess;
  flagcxResult_t commonRouteStatus = flagcxSuccess;
  FLAGCXCHECK(
      flagcxSymGlobalConverge(comm, localRouteStatus, &commonRouteStatus));
  if (commonRouteStatus != flagcxSuccess)
    return commonRouteStatus;

  flagcxResult_t res = flagcxSuccess;
  flagcxWindow_t w = nullptr;
  flagcxSymWindow_t d = nullptr;
  int *allFds = nullptr;
  int *localDevices = nullptr;
  void **peerHandles = nullptr;
  void *physHandle = nullptr;
  void *mcHandle = nullptr;
  int mcFd = -1;
  int shareableFd = -1;
  bool ipcSockOpen = false;
  bool mcIpcSockOpen = false;
  bool retainCleanupForRetry = false;
  bool prepareRollbackConverged = false;
  flagcxResult_t localPrepareStatus = flagcxSuccess;
  void *allocationBase = nullptr;
  flagcxResult_t leaseResult = flagcxSuccess;
  struct flagcxIpcSocket ipcSock;
  struct flagcxIpcSocket mcIpcSock;
  memset(&ipcSock, 0, sizeof(ipcSock));
  memset(&mcIpcSock, 0, sizeof(mcIpcSock));

  // The window wrappers are rank-local preparation. Converge their outcome
  // before any rank starts VMM/IPC collectives; otherwise a single calloc
  // failure can leave its peers blocked in FD exchange.
  flagcxResult_t localInitStatus = flagcxCalloc(&w, 1);
  if (localInitStatus == flagcxSuccess)
    localInitStatus = flagcxCalloc(&d, 1);
  flagcxResult_t commonInitStatus = flagcxSuccess;
  res = flagcxSymGlobalConverge(comm, localInitStatus, &commonInitStatus);
  if (res != flagcxSuccess) {
    free(d);
    free(w);
    return res;
  }
  if (commonInitStatus != flagcxSuccess) {
    free(d);
    free(w);
    return commonInitStatus;
  }

  w->vendorBase = nullptr;
  w->defaultBase = d;
  w->isSymmetricDefault = 1;
  w->winFlags = winFlags;

  d->mrIndex = -1;
  d->mrBase = 0;
  d->hasNetworkMrRef = false;
  d->ipcSlot = -1;
  d->localBase = buff;
  d->owner = w;
  d->allocationIsVmm = allocationIsVmm;
  d->state = flagcxSymWindowPreparing;

  // Buffers owned by flagcxMemAlloc carry a lease for the full lifetime of
  // the window, including failed-registration cleanup. External/internal
  // allocations are intentionally allowed and simply have no registry lease.
  leaseResult =
      globalMemAllocRegistry.retainWindowRange(buff, size, &allocationBase);
  if (leaseResult == flagcxSuccess) {
    d->hasAllocationLease = true;
    d->allocationBase = allocationBase;
  } else if (leaseResult != flagcxInvalidUsage) {
    localPrepareStatus = leaseResult;
  }

  // Registry retention can fail on one rank only (for example, allocation of
  // registry bookkeeping). Do not let successful ranks enter VMM preparation
  // until that status is communicator-wide.
  {
    flagcxResult_t commonLeaseStatus = flagcxSuccess;
    FLAGCXCHECKGOTO(
        flagcxSymGlobalConverge(comm, localPrepareStatus, &commonLeaseStatus),
        res, fail);
    if (commonLeaseStatus != flagcxSuccess) {
      res = commonLeaseStatus;
      goto fail;
    }
  }

  int localRanks;
  localRanks = comm->localRanks;
  int localRank;
  localRank = comm->localRank;
  d->localRanks = localRanks;
  d->heapSize = size;

  // ---- Try VMM path ----
  {
    bool vmmOk = false;
    const bool localVmmAvailable =
        deviceAdaptor->symPhysAlloc != nullptr &&
        deviceAdaptor->symPhysFree != nullptr &&
        deviceAdaptor->symFlatMap != nullptr &&
        deviceAdaptor->symFlatMappingUnmap != nullptr &&
        deviceAdaptor->symFlatVaFree != nullptr && flagcxParamVmmEnable();
    flagcxResult_t commonVmmAvailability = flagcxSuccess;
    FLAGCXCHECKGOTO(flagcxSymLocalConverge(
                        comm,
                        localVmmAvailable ? flagcxSuccess : flagcxNotSupported,
                        /*tag=*/0x592f, &commonVmmAvailability),
                    res, fail);
    // VMM FD exchange is node-local, so every local rank must select the same
    // path even when environment or plugin capabilities differ by process.
    if (commonVmmAvailability == flagcxSuccess) {
      // Allocate every rank-local exchange array before the first provider or
      // bootstrap operation. Their status is converged within the node so no
      // peer can enter FD exchange while another rank exits on ENOMEM.
      flagcxResult_t localScratchStatus = flagcxCalloc(&allFds, localRanks);
      if (localScratchStatus == flagcxSuccess)
        localScratchStatus = flagcxCalloc(&peerHandles, localRanks);
      if (localScratchStatus == flagcxSuccess)
        localScratchStatus = flagcxCalloc(&localDevices, localRanks);
      if (allFds != nullptr) {
        for (int i = 0; i < localRanks; i++)
          allFds[i] = -1;
      }
      flagcxResult_t scratchStatus = flagcxSuccess;
      FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localScratchStatus,
                                             /*tag=*/0x5930, &scratchStatus),
                      res, fail);
      if (scratchStatus != flagcxSuccess) {
        localPrepareStatus = scratchStatus;
      } else {
        size_t handleSize = sizeof(int);
        size_t allocSize = 0;

        flagcxResult_t allocRes = deviceAdaptor->symPhysAlloc(
            buff, size, &physHandle, &shareableFd, &handleSize, &allocSize);
        INFO(FLAGCX_INIT,
             "[symWindowRegister] symPhysAlloc: res=%d physHandle=%p "
             "shareableFd=%d allocSize=%zu buff=%p size=%zu",
             (int)allocRes, physHandle, shareableFd, allocSize, buff, size);
        flagcxResult_t localAllocStatus =
            (allocRes == flagcxSuccess && physHandle != nullptr &&
             shareableFd >= 0 && allocSize >= size)
                ? flagcxSuccess
                : (allocRes == flagcxSuccess ? flagcxInternalError : allocRes);

        // No rank may enter FD exchange until every local rank has completed
        // local preparation successfully.
        flagcxResult_t allocStatus = flagcxSuccess;
        FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localAllocStatus,
                                               /*tag=*/0x5931, &allocStatus),
                        res, fail);
        bool allAllocOk = allocStatus == flagcxSuccess;

        if (allAllocOk) {
          comm->symWindowFdExchangeCount++;
          // Exchange shareable FDs with intra-node peers via Unix Domain Socket
          allFds[localRank] = shareableFd;

          // Hash must be identical across all ranks
          uint64_t ipcHash = comm->commHash ^ size;

          FLAGCXCHECKGOTO(
              flagcxIpcSocketInit(&ipcSock, comm->rank, ipcHash, /*block=*/1),
              res, fail);
          ipcSockOpen = true;

          // Barrier to ensure all sockets are created before sending
          struct bootstrapState *state = comm->bootstrap;
          FLAGCXCHECKGOTO(bootstrapCollIntraNodeBarrier(state,
                                                        comm->localRankToRank,
                                                        localRank, localRanks,
                                                        /*tag=*/0x5932),
                          res, fail);

          // Send our FD to each peer
          for (int i = 0; i < localRanks; i++) {
            if (i == localRank)
              continue;
            int peerGlobalRank = comm->localRankToRank[i];
            FLAGCXCHECKGOTO(
                flagcxIpcSocketSendMsg(&ipcSock, &localRank, sizeof(localRank),
                                       shareableFd, peerGlobalRank, ipcHash),
                res, fail);
          }

          // Receive FDs from each peer
          int received = 0;
          int expected = localRanks - 1;
          while (received < expected) {
            int senderLocalRank = -1;
            int fd = -1;
            FLAGCXCHECKGOTO(flagcxIpcSocketRecvMsg(&ipcSock, &senderLocalRank,
                                                   sizeof(senderLocalRank),
                                                   &fd),
                            res, fail);
            if (senderLocalRank < 0 || senderLocalRank >= localRanks ||
                senderLocalRank == localRank || fd < 0 ||
                allFds[senderLocalRank] >= 0) {
              if (fd >= 0)
                close(fd);
              res = flagcxInternalError;
              goto fail;
            }
            allFds[senderLocalRank] = fd;
            received++;
          }

          flagcxIpcSocketClose(&ipcSock);
          ipcSockOpen = false;

          // Build peer handle pointers for symFlatMap
          for (int i = 0; i < localRanks; i++) {
            peerHandles[i] = &allFds[i];
          }

          void *flatBase = nullptr;
          flagcxResult_t mapRes = deviceAdaptor->symFlatMap
                                      ? deviceAdaptor->symFlatMap(
                                            peerHandles, localRanks, localRank,
                                            physHandle, allocSize, &flatBase)
                                      : flagcxNotSupported;
          INFO(FLAGCX_INIT,
               "[symWindowRegister] symFlatMap: mapRes=%d flatBase=%p "
               "localRanks=%d allocSize=%zu",
               (int)mapRes, flatBase, localRanks, allocSize);

          // Take ownership before convergence. A peer can fail after this rank
          // has mapped successfully, and rollback must retain both the mapping
          // and physical handle if unmap/free fails.
          d->flatBase = flatBase;
          d->physHandle = physHandle;
          d->allocSize = allocSize;
          d->isVMM = true;
          d->flatMappingOwned = flatBase != nullptr;
          d->flatVaOwned = flatBase != nullptr;
          physHandle = nullptr;
          flagcxResult_t localMapStatus =
              mapRes == flagcxSuccess && flatBase != nullptr
                  ? flagcxSuccess
                  : (mapRes == flagcxSuccess ? flagcxInternalError : mapRes);
          flagcxResult_t mapStatus = flagcxSuccess;
          FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localMapStatus,
                                                 /*tag=*/0x5935, &mapStatus),
                          res, fail);
          if (mapStatus == flagcxSuccess) {
            vmmOk = true;

            // Try multicast setup
            d->mcBase = nullptr;
            int mcSupported = 0;
            flagcxResult_t mcSupportResult =
                deviceAdaptor->symMulticastSupported != nullptr
                    ? deviceAdaptor->symMulticastSupported(&mcSupported)
                    : flagcxNotSupported;
            // Rank-local communicator destruction cannot rendezvous with peers.
            // Multicast is therefore enabled only when every process can retain
            // its own provider reference until after its local mapping
            // teardown.
            if (deviceAdaptor->symMulticastCreate == nullptr ||
                deviceAdaptor->symMulticastImport == nullptr ||
                deviceAdaptor->symMulticastBind == nullptr ||
                deviceAdaptor->symMulticastFree == nullptr)
              mcSupported = 0;
            if (deviceAdaptor->symMulticastMappingUnmap == nullptr ||
                deviceAdaptor->symMulticastVaFree == nullptr)
              mcSupported = 0;
            flagcxResult_t commonMcAvailability = flagcxSuccess;
            FLAGCXCHECKGOTO(flagcxSymLocalConverge(
                                comm,
                                mcSupportResult == flagcxSuccess && mcSupported
                                    ? flagcxSuccess
                                    : flagcxNotSupported,
                                /*tag=*/0x592e, &commonMcAvailability),
                            res, fail);
            mcSupported = commonMcAvailability == flagcxSuccess ? 1 : 0;
            if (mcSupported) {
              // Build local device ordinal array from peerInfo
              for (int i = 0; i < localRanks; i++) {
                int globalRank = comm->localRankToRank[i];
                localDevices[i] = comm->peerInfo[globalRank].cudaDev;
              }

              if (localRank == 0) {
                flagcxResult_t mcRes = deviceAdaptor->symMulticastCreate
                                           ? deviceAdaptor->symMulticastCreate(
                                                 allocSize, localRanks,
                                                 localDevices, &mcHandle, &mcFd)
                                           : flagcxNotSupported;
                if (mcRes != flagcxSuccess || mcHandle == nullptr || mcFd < 0) {
                  mcSupported = 0;
                } else {
                  // Retain ownership immediately so every later failure uses
                  // the same ordered cleanup state machine.
                  d->mcHandle = mcHandle;
                  mcHandle = nullptr;
                }
              }
              // A provider may fail after publishing a partial object. Move it
              // into the retryable window state before broadcasting the
              // optional multicast decision; never discard ownership on an
              // error return.
              flagcxResult_t mcCreateCleanup = flagcxSuccess;
              if (localRank == 0 && !mcSupported) {
                if (mcHandle != nullptr) {
                  d->mcHandle = mcHandle;
                  mcHandle = nullptr;
                  mcCreateCleanup =
                      deviceAdaptor->symMulticastFree(d->mcHandle);
                  if (mcCreateCleanup == flagcxSuccess)
                    d->mcHandle = nullptr;
                }
                if (mcFd >= 0) {
                  close(mcFd);
                  mcFd = -1;
                }
              }
              // Broadcast success/failure from rank 0
              struct bootstrapState *mcState = comm->bootstrap;
              if (localRank == 0) {
                for (int i = 1; i < localRanks; i++) {
                  int peerGlobalRank = comm->localRankToRank[i];
                  FLAGCXCHECKGOTO(bootstrapSend(mcState, peerGlobalRank,
                                                /*tag=*/0x5933, &mcSupported,
                                                sizeof(mcSupported)),
                                  res, fail);
                }
              } else {
                int rank0Global = comm->localRankToRank[0];
                FLAGCXCHECKGOTO(bootstrapRecv(mcState, rank0Global,
                                              /*tag=*/0x5933, &mcSupported,
                                              sizeof(mcSupported)),
                                res, fail);
              }
              flagcxResult_t mcCreateCleanupStatus = flagcxSuccess;
              FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, mcCreateCleanup,
                                                     /*tag=*/0x593c,
                                                     &mcCreateCleanupStatus),
                              res, fail);
              if (mcCreateCleanupStatus != flagcxSuccess) {
                localPrepareStatus = mcCreateCleanupStatus;
                retainCleanupForRetry = true;
                mcSupported = 0;
              }

              if (mcSupported) {
                uint64_t mcIpcHash = ipcHash ^ 0x4D43; // "MC"

                FLAGCXCHECKGOTO(flagcxIpcSocketInit(&mcIpcSock, comm->rank,
                                                    mcIpcHash, /*block=*/1),
                                res, fail);
                mcIpcSockOpen = true;

                // Barrier: ensure all peers have created their IPC sockets
                FLAGCXCHECKGOTO(bootstrapCollIntraNodeBarrier(
                                    state, comm->localRankToRank, localRank,
                                    localRanks, /*tag=*/0x5934),
                                res, fail);

                if (localRank == 0) {
                  for (int i = 1; i < localRanks; i++) {
                    int peerGlobalRank = comm->localRankToRank[i];
                    int tag = 0;
                    FLAGCXCHECKGOTO(
                        flagcxIpcSocketSendMsg(&mcIpcSock, &tag, sizeof(tag),
                                               mcFd, peerGlobalRank, mcIpcHash),
                        res, fail);
                  }
                } else {
                  int tag = -1;
                  FLAGCXCHECKGOTO(flagcxIpcSocketRecvMsg(&mcIpcSock, &tag,
                                                         sizeof(tag), &mcFd),
                                  res, fail);
                  if (tag != 0 || mcFd < 0) {
                    res = flagcxInternalError;
                    goto fail;
                  }
                }

                flagcxIpcSocketClose(&mcIpcSock);
                mcIpcSockOpen = false;

                // Every non-owner rank imports and retains its own provider
                // reference. This makes rank-local teardown safe even when the
                // creator rank destroys its communicator first.
                flagcxResult_t localImportStatus = flagcxSuccess;
                if (localRank != 0) {
                  localImportStatus =
                      deviceAdaptor->symMulticastImport(mcFd, &d->mcHandle);
                  if (localImportStatus == flagcxSuccess &&
                      d->mcHandle == nullptr)
                    localImportStatus = flagcxInternalError;
                } else if (d->mcHandle == nullptr) {
                  localImportStatus = flagcxInternalError;
                }
                flagcxResult_t importStatus = flagcxSuccess;
                FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localImportStatus,
                                                       /*tag=*/0x5939,
                                                       &importStatus),
                                res, fail);
                if (importStatus != flagcxSuccess) {
                  flagcxResult_t cleanupRes = flagcxSuccess;
                  if (d->mcHandle != nullptr) {
                    cleanupRes =
                        deviceAdaptor->symMulticastFree != nullptr
                            ? deviceAdaptor->symMulticastFree(d->mcHandle)
                            : flagcxInternalError;
                    if (cleanupRes == flagcxSuccess)
                      d->mcHandle = nullptr;
                  }
                  flagcxResult_t cleanupStatus = flagcxSuccess;
                  FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, cleanupRes,
                                                         /*tag=*/0x593a,
                                                         &cleanupStatus),
                                  res, fail);
                  if (cleanupStatus != flagcxSuccess) {
                    localPrepareStatus = cleanupStatus;
                    retainCleanupForRetry = true;
                  }
                  mcSupported = 0;
                }

                if (mcSupported) {
                  // All ranks bind through their retained local handle.
                  void *mcBaseVa = nullptr;
                  size_t mcMapSize = 0;
                  flagcxResult_t mcRes =
                      deviceAdaptor->symMulticastBind
                          ? deviceAdaptor->symMulticastBind(
                                d->mcHandle, /*importFd=*/-1, d->physHandle,
                                allocSize, localRank, localRanks, &mcBaseVa,
                                &mcMapSize)
                          : flagcxNotSupported;
                  flagcxResult_t localMcStatus =
                      mcRes == flagcxSuccess && mcBaseVa != nullptr
                          ? flagcxSuccess
                          : (mcRes == flagcxSuccess ? flagcxInternalError
                                                    : mcRes);
                  d->mcBase = mcBaseVa;
                  d->mcMapSize = mcMapSize;
                  d->multicastMappingOwned = mcBaseVa != nullptr;
                  d->multicastVaOwned = mcBaseVa != nullptr;
                  flagcxResult_t mcStatus = flagcxSuccess;
                  FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localMcStatus,
                                                         /*tag=*/0x5936,
                                                         &mcStatus),
                                  res, fail);
                  if (mcStatus == flagcxSuccess) {
                  } else {
                    WARN("symMulticastBind failed: res=%d mcBaseVa=%p "
                         "(localRank=%d)",
                         mcStatus, mcBaseVa, localRank);
                    flagcxResult_t cleanupRes = flagcxSuccess;
                    if (d->multicastMappingOwned) {
                      cleanupRes = deviceAdaptor->symMulticastMappingUnmap(
                          d->mcBase, d->mcMapSize);
                      if (cleanupRes == flagcxSuccess)
                        d->multicastMappingOwned = false;
                    }
                    flagcxResult_t cleanupStatus = flagcxSuccess;
                    FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, cleanupRes,
                                                           /*tag=*/0x5937,
                                                           &cleanupStatus),
                                    res, fail);
                    if (cleanupStatus != flagcxSuccess) {
                      localPrepareStatus = cleanupStatus;
                      retainCleanupForRetry = true;
                    } else {
                      cleanupRes = flagcxSuccess;
                      if (d->multicastVaOwned) {
                        cleanupRes = deviceAdaptor->symMulticastVaFree(
                            d->mcBase, d->mcMapSize);
                        if (cleanupRes == flagcxSuccess) {
                          d->multicastVaOwned = false;
                          d->mcBase = nullptr;
                          d->mcMapSize = 0;
                        }
                      }
                      FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, cleanupRes,
                                                             /*tag=*/0x593b,
                                                             &cleanupStatus),
                                      res, fail);
                      if (cleanupStatus != flagcxSuccess) {
                        localPrepareStatus = cleanupStatus;
                        retainCleanupForRetry = true;
                      }
                    }
                    if (cleanupStatus == flagcxSuccess) {
                      cleanupRes = flagcxSuccess;
                      if (d->mcHandle != nullptr) {
                        cleanupRes =
                            deviceAdaptor->symMulticastFree != nullptr
                                ? deviceAdaptor->symMulticastFree(d->mcHandle)
                                : flagcxInternalError;
                        if (cleanupRes == flagcxSuccess)
                          d->mcHandle = nullptr;
                      }
                      FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, cleanupRes,
                                                             /*tag=*/0x5938,
                                                             &cleanupStatus),
                                      res, fail);
                      if (cleanupStatus != flagcxSuccess) {
                        localPrepareStatus = cleanupStatus;
                        retainCleanupForRetry = true;
                      }
                    }
                    mcSupported = 0;
                  }
                }

                // Close the multicast FD
                if (mcFd >= 0) {
                  close(mcFd);
                  mcFd = -1;
                }
              }
              if (mcFd >= 0) {
                close(mcFd);
                mcFd = -1;
              }
            }
          } else {
            // A peer may have mapped successfully before another rank failed.
            // All ranks participate in ordered cleanup. If any rank cannot
            // release its mapping, preserve the complete ownership record for a
            // later retry instead of falling through to IPC.
            flagcxResult_t cleanupRes =
                flagcxSymCleanupMappings(comm, d, flagcxSymCleanupCollective);
            if (cleanupRes != flagcxSuccess) {
              WARN("symFlatMap rollback retained for retry: res=%d",
                   cleanupRes);
              localPrepareStatus = cleanupRes;
              retainCleanupForRetry = true;
            } else {
              d->isVMM = false;
              d->allocSize = 0;
            }
          }

          free(peerHandles);
          peerHandles = nullptr;
          // Close all FDs
          for (int i = 0; i < localRanks; i++) {
            if (allFds[i] >= 0)
              close(allFds[i]);
          }
          free(allFds);
          allFds = nullptr;
          shareableFd = -1;
        } else {
          // Some ranks may already own a physical allocation. Make every local
          // rank enter the same cleanup phases, retaining ownership if release
          // fails, before allowing IPC fallback.
          d->isVMM = true;
          d->allocSize = allocSize;
          if (physHandle != nullptr) {
            d->physHandle = physHandle;
            physHandle = nullptr;
          }
          if (shareableFd >= 0) {
            close(shareableFd);
            shareableFd = -1;
          }
          flagcxResult_t cleanupRes =
              flagcxSymCleanupMappings(comm, d, flagcxSymCleanupCollective);
          if (cleanupRes != flagcxSuccess) {
            WARN("symPhysAlloc rollback retained for retry: res=%d",
                 cleanupRes);
            localPrepareStatus = cleanupRes;
            retainCleanupForRetry = true;
          } else {
            d->isVMM = false;
            d->allocSize = 0;
          }
        }
        if (!allAllocOk && shareableFd >= 0) {
          close(shareableFd);
          shareableFd = -1;
        }
      }
      free(localDevices);
      localDevices = nullptr;
      // The success path normally released these after FD exchange; these
      // guards also cover scratch allocation failure and phys-alloc rollback.
      if (allFds != nullptr) {
        for (int i = 0; i < localRanks; i++) {
          if (allFds[i] >= 0)
            close(allFds[i]);
        }
        free(allFds);
        allFds = nullptr;
      }
      free(peerHandles);
      peerHandles = nullptr;
    }

    // ---- IPC fallback if VMM not available ----
    if (!vmmOk && localPrepareStatus == flagcxSuccess) {
      // A provider error may return a partial physical handle. Route it
      // through the staged cleanup state machine so a failed release remains
      // represented by the window cleanup token.
      if (physHandle != nullptr) {
        d->physHandle = physHandle;
        physHandle = nullptr;
        d->isVMM = true;
        flagcxResult_t cleanupRes =
            flagcxSymCleanupMappings(comm, d, flagcxSymCleanupCollective);
        if (cleanupRes != flagcxSuccess) {
          localPrepareStatus = cleanupRes;
          retainCleanupForRetry = true;
        }
      }
      if (localPrepareStatus == flagcxSuccess) {
        d->flatBase = nullptr;
        d->mcBase = nullptr;
        d->physHandle = nullptr;
        d->isVMM = false;
        d->allocSize = 0;
      }
    }
  }

  // VMM preparation and its local rollback complete independently on each
  // node. Converge the outcome across the whole communicator before any rank
  // enters one-sided MR setup, whose collectives must never race the caller's
  // window-status collective on another node.
  {
    flagcxResult_t globalPrepareStatus = flagcxSuccess;
    FLAGCXCHECKGOTO(
        flagcxSymGlobalConverge(comm, localPrepareStatus, &globalPrepareStatus),
        res, fail);
    if (globalPrepareStatus != flagcxSuccess) {
      // Nodes that prepared successfully roll back now. A node whose earlier
      // rollback failed keeps its exact ownership for the next registration's
      // retry instead of immediately re-running a failed release operation.
      flagcxResult_t localRollbackStatus = localPrepareStatus;
      if (localPrepareStatus == flagcxSuccess) {
        localRollbackStatus =
            flagcxSymCleanupMappings(comm, d, flagcxSymCleanupCollective);
        if (localRollbackStatus == flagcxSuccess) {
          d->isVMM = false;
          d->allocSize = 0;
        } else {
          retainCleanupForRetry = true;
        }
      }

      flagcxResult_t globalRollbackStatus = flagcxSuccess;
      FLAGCXCHECKGOTO(flagcxSymGlobalConverge(comm, localRollbackStatus,
                                              &globalRollbackStatus),
                      res, fail);
      prepareRollbackConverged = true;
      retainCleanupForRetry = globalRollbackStatus != flagcxSuccess;
      res = retainCleanupForRetry ? globalRollbackStatus : globalPrepareStatus;
      goto fail;
    }
  }

  // ---- Network MR registration ----
  // Remote peers always require NET. A local-only communicator also requires
  // NET when P2P is disabled because runtime PUT/signal paths intentionally
  // bypass its flat/IPC mappings.
  if (needsNetworkMr) {
    INFO(FLAGCX_INIT,
         "[symWindowRegister] vmmOk=%d, registering MR for buff=%p size=%zu",
         (int)d->isVMM, buff, size);
    FLAGCXCHECKGOTO(flagcxSymWindowEnsureNetworkMr(comm, d), res, fail);
  }

  *win = w;
  return flagcxSuccess;

fail:
  if (mcIpcSockOpen)
    flagcxIpcSocketClose(&mcIpcSock);
  if (ipcSockOpen)
    flagcxIpcSocketClose(&ipcSock);
  if (shareableFd >= 0 && (allFds == NULL || allFds[localRank] != shareableFd))
    close(shareableFd);
  if (allFds) {
    for (int i = 0; i < comm->localRanks; i++) {
      if (allFds[i] >= 0)
        close(allFds[i]);
    }
    free(allFds);
  }
  free(peerHandles);
  free(localDevices);
  // Move all provider objects into the window before rollback so a failed
  // release can be represented by the returned cleanup token.
  if (d != nullptr && physHandle != nullptr) {
    d->physHandle = physHandle;
    d->isVMM = true;
    physHandle = nullptr;
  }
  if (d != nullptr && mcHandle != nullptr) {
    d->mcHandle = mcHandle;
    d->isVMM = true;
    mcHandle = nullptr;
  }
  if (retainCleanupForRetry && d != nullptr) {
    d->state = flagcxSymWindowCleanupRequired;
    flagcxSymRetainPendingCleanup(comm, w);
    *win = w;
    d = nullptr;
    w = nullptr;
  } else if (!prepareRollbackConverged && d != nullptr &&
             (d->hasNetworkMrRef || d->hasPendingNetworkCleanup ||
              d->flatMappingOwned || d->flatVaOwned ||
              d->multicastMappingOwned || d->multicastVaOwned ||
              d->physHandle != nullptr || d->mcHandle != nullptr)) {
    flagcxResult_t cleanup =
        flagcxSymCleanupUnpublished(comm, d, flagcxSymCleanupCollective);
    if (cleanup != flagcxSuccess) {
      WARN("symWindowRegister rollback retained for retry: res=%d", cleanup);
      d->state = flagcxSymWindowCleanupRequired;
      flagcxSymRetainPendingCleanup(comm, w);
      *win = w;
      d = nullptr;
      w = nullptr;
      res = cleanup;
    }
  }
  if (d != nullptr) {
    flagcxResult_t leaseCleanup = flagcxSymReleaseAllocationLease(d);
    if (leaseCleanup != flagcxSuccess) {
      d->state = flagcxSymWindowCleanupRequired;
      flagcxSymRetainPendingCleanup(comm, w);
      *win = w;
      d = nullptr;
      w = nullptr;
      res = leaseCleanup;
    }
  }
  if (physHandle != nullptr && deviceAdaptor->symPhysFree)
    deviceAdaptor->symPhysFree(physHandle);
  if (mcFd >= 0)
    close(mcFd);
  if (mcHandle != nullptr && deviceAdaptor->symMulticastFree)
    deviceAdaptor->symMulticastFree(mcHandle);
  free(d);
  free(w);
  return res;
}

flagcxResult_t flagcxSymWindowRegister(flagcxHeteroComm_t comm, void *buff,
                                       size_t size, flagcxWindow_t *win,
                                       int winFlags) {
  return flagcxSymWindowRegisterInternal(comm, buff, size, win, winFlags,
                                         flagcxOneSideMemoryIsVmm(buff, size));
}

flagcxResult_t flagcxSymWindowDeregister(flagcxHeteroComm_t comm,
                                         flagcxWindow_t win,
                                         flagcxSymCleanupMode mode) {
  if (win == nullptr)
    return flagcxSuccess;

  flagcxSymWindow_t d = win->defaultBase;
  if (d != nullptr) {
    if (d->published && comm == nullptr)
      return flagcxInvalidArgument;
    d->state = flagcxSymWindowCleanupRequired;
    FLAGCXCHECK(flagcxSymCleanupMappingsConverged(comm, d, mode));
    FLAGCXCHECK(flagcxSymReleaseAllocationLease(d));

    // Only detach after all fallible cleanup has completed. A failure above
    // leaves both the public handle and list ownership intact for retry.
    if (d->published && comm != nullptr) {
      flagcxSymWindow_t *link = &comm->symWindows;
      while (*link != nullptr && *link != d)
        link = &(*link)->next;
      if (*link == d)
        *link = d->next;
      d->published = false;
    }

    if (comm != nullptr) {
      flagcxSymWindow_t *cleanupLink = &comm->pendingSymCleanup;
      while (*cleanupLink != nullptr && *cleanupLink != d)
        cleanupLink = &(*cleanupLink)->cleanupNext;
      if (*cleanupLink == d)
        *cleanupLink = d->cleanupNext;
    }

    free(d);
    win->defaultBase = nullptr;
  }

  free(win);
  return flagcxSuccess;
}

flagcxResult_t flagcxSymWindowPublish(flagcxHeteroComm_t comm,
                                      flagcxWindow_t win) {
  if (comm == nullptr || win == nullptr || win->defaultBase == nullptr)
    return flagcxInvalidArgument;
  flagcxSymWindow_t d = win->defaultBase;
  if (d->state == flagcxSymWindowCleanupRequired)
    return flagcxInvalidUsage;
  if (d->published)
    return flagcxSuccess;
  FLAGCXCHECK(flagcxSymWindowValidateDataRoutes(comm, d));
  d->next = comm->symWindows;
  comm->symWindows = d;
  d->published = true;
  d->state = flagcxSymWindowPublished;
  return flagcxSuccess;
}

flagcxSymWindow_t flagcxSymWindowFind(flagcxHeteroComm_t comm, const void *ptr,
                                      size_t size, size_t *offset) {
  if (comm == nullptr || ptr == nullptr)
    return nullptr;

  uintptr_t address = (uintptr_t)ptr;
  for (flagcxSymWindow_t window = comm->symWindows; window != nullptr;
       window = window->next) {
    if (window->state != flagcxSymWindowPublished)
      continue;
    uintptr_t base = (uintptr_t)window->localBase;
    if (address < base)
      continue;
    size_t localOffset = (size_t)(address - base);
    if (localOffset > window->heapSize || size > window->heapSize - localOffset)
      continue;
    if (offset != nullptr)
      *offset = localOffset;
    return window;
  }
  return nullptr;
}

flagcxResult_t flagcxSymWindowResolveIpcPeerPtr(flagcxHeteroComm_t comm,
                                                flagcxSymWindow_t window,
                                                int peer, size_t offset,
                                                size_t size, void **ptr) {
  if (comm == nullptr || window == nullptr || ptr == nullptr || peer < 0 ||
      peer >= comm->nRanks)
    return flagcxInvalidArgument;
  *ptr = nullptr;
  if (offset > window->heapSize || size > window->heapSize - offset)
    return flagcxInvalidArgument;
  if (comm->rankToNode == nullptr || comm->rankToLocalRank == nullptr ||
      comm->rankToNode[peer] != comm->node)
    return flagcxNotSupported;

  int peerLocalRank = comm->rankToLocalRank[peer];
  if (peerLocalRank < 0 || peerLocalRank >= window->localRanks)
    return flagcxInternalError;

  if (peer == comm->rank) {
    *ptr = (void *)((uintptr_t)window->localBase + offset);
    return flagcxSuccess;
  }

  if (window->isVMM && window->flatBase != nullptr) {
    *ptr = (void *)((uintptr_t)window->flatBase +
                    (size_t)peerLocalRank * window->allocSize + offset);
    return flagcxSuccess;
  }

  int slot = window->ipcSlot;
  if (slot < 0 || comm->ipcTable == nullptr || slot >= comm->ipcTableSize)
    return flagcxNotSupported;
  struct flagcxIpcTableEntry *entry = &comm->ipcTable[slot];
  if (!entry->inUse || entry->hostPeerPtrs == nullptr ||
      peerLocalRank >= entry->nPeers ||
      entry->hostPeerPtrs[peerLocalRank] == nullptr)
    return flagcxNotSupported;

  *ptr = (void *)((uintptr_t)entry->hostPeerPtrs[peerLocalRank] + offset);
  return flagcxSuccess;
}
