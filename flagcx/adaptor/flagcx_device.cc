/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Host-side lifecycle management for flagcxDevComm_t and flagcxDevMem_t.
 *
 * Tier 1 (NCCL > 2.28): calls pncclDevCommCreate/Destroy via dlsym.
 *   DevMem supports both window mode and IPC mode at runtime.
 * Tier 2 (fallback):    IPC-based barrier + peer pointer exchange.
 *
 * IPC peer pointer exchange is shared across both tiers so that
 * -R 1 (IPC mode) works even on NCCL > 2.28.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "flagcx_kernel.h"
#include "p2p.h" // flagcxP2pAllocateShareableBuffer, flagcxP2pIpcDesc

// ==========================================================================
// Shared: IPC peer pointer exchange (used by both tiers)
// ==========================================================================

// Forward declaration (defined below buildIpcPeerPointers).
static void cleanupIpcPeerPointers(void **hostPeerPtrs, void **devPeerPtrs,
                                   int nPeers, void *ownBuff);

// Build IPC peer pointer table for a user buffer.
// Allocates devPeerPtrs (device array) and hostPeerPtrs (host array).
// Caller must free both on cleanup.
static flagcxResult_t buildIpcPeerPointers(flagcxComm_t comm, void *buff,
                                           size_t size, void ***outDevPeerPtrs,
                                           void ***outHostPeerPtrs,
                                           int *outNPeers) {

  int myRank = comm->rank;
  int nRanks = comm->nranks;
  int localRanks = comm->localRanks;
  int *localRankToRank = comm->localRankToRank;

  flagcxResult_t res = flagcxSuccess;
  struct flagcxP2pIpcDesc *allDescs = nullptr;
  void **hostPeerPtrs = nullptr;
  void **devPeerPtrs = nullptr;

  // Step 1: Get IPC handle for existing user buffer
  struct flagcxP2pIpcDesc myIpcDesc;
  memset(&myIpcDesc, 0, sizeof(myIpcDesc));
  {
    flagcxIpcMemHandle_t handlePtr = nullptr;
    size_t ipcSize = 0;
    FLAGCXCHECKGOTO(deviceAdaptor->ipcMemHandleCreate(&handlePtr, &ipcSize),
                    res, fail);
    res = deviceAdaptor->ipcMemHandleGet(handlePtr, buff);
    if (res != flagcxSuccess) {
      deviceAdaptor->ipcMemHandleFree(handlePtr);
      goto fail;
    }
    if (ipcSize > sizeof(flagcxIpcHandleData)) {
      deviceAdaptor->ipcMemHandleFree(handlePtr);
      res = flagcxInternalError;
      goto fail;
    }
    memcpy(&myIpcDesc.handleData, handlePtr, ipcSize);
    myIpcDesc.size = size;
    deviceAdaptor->ipcMemHandleFree(handlePtr);
  }

  // Step 2: Exchange IPC handles with all ranks
  allDescs = (struct flagcxP2pIpcDesc *)calloc(nRanks,
                                               sizeof(struct flagcxP2pIpcDesc));
  if (allDescs == nullptr) {
    res = flagcxSystemError;
    goto fail;
  }
  memcpy(&allDescs[myRank], &myIpcDesc, sizeof(struct flagcxP2pIpcDesc));
  FLAGCXCHECKGOTO(bootstrapAllGather(comm->bootstrap, allDescs,
                                     sizeof(struct flagcxP2pIpcDesc)),
                  res, fail);

  // Step 3: Open intra-node peer IPC handles
  hostPeerPtrs = (void **)calloc(localRanks, sizeof(void *));
  if (hostPeerPtrs == nullptr) {
    res = flagcxSystemError;
    goto fail;
  }
  for (int lr = 0; lr < localRanks; lr++) {
    int gr = localRankToRank[lr];
    if (gr == myRank) {
      hostPeerPtrs[lr] = buff;
    } else {
      flagcxIpcMemHandle_t handlePtr =
          (flagcxIpcMemHandle_t)&allDescs[gr].handleData;
      FLAGCXCHECKGOTO(
          deviceAdaptor->ipcMemHandleOpen(handlePtr, &hostPeerPtrs[lr]), res,
          fail);
    }
  }
  free(allDescs);
  allDescs = nullptr;

  // Step 4: Build device peer pointer array
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc((void **)&devPeerPtrs,
                                              localRanks * sizeof(void *),
                                              flagcxMemDevice, NULL),
                  res, fail);
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMemcpy(
                      devPeerPtrs, hostPeerPtrs, localRanks * sizeof(void *),
                      flagcxMemcpyHostToDevice, NULL, NULL),
                  res, fail);

  *outDevPeerPtrs = devPeerPtrs;
  *outHostPeerPtrs = hostPeerPtrs;
  *outNPeers = localRanks;
  return flagcxSuccess;

fail:
  free(allDescs);
  cleanupIpcPeerPointers(hostPeerPtrs, devPeerPtrs, localRanks, buff);
  return res;
}

// Close IPC peer handles and free arrays.
static void cleanupIpcPeerPointers(void **hostPeerPtrs, void **devPeerPtrs,
                                   int nPeers, void *ownBuff) {
  if (hostPeerPtrs) {
    for (int i = 0; i < nPeers; i++) {
      if (hostPeerPtrs[i] && hostPeerPtrs[i] != ownBuff) {
        deviceAdaptor->ipcMemHandleClose(hostPeerPtrs[i]);
      }
    }
    free(hostPeerPtrs);
  }
  if (devPeerPtrs) {
    deviceAdaptor->deviceFree(devPeerPtrs, flagcxMemDevice, NULL);
  }
}

#ifdef FLAGCX_DEVICE_API_NCCL

// ==========================================================================
// Tier 1: NCCL > 2.28
// ==========================================================================

#include "nvidia_adaptor.h"

flagcxResult_t flagcxDevCommCreate(flagcxComm_t comm,
                                   const flagcxDevCommRequirements *reqs,
                                   flagcxDevComm_t *devComm) {
  if (comm == nullptr || reqs == nullptr || devComm == nullptr) {
    return flagcxInvalidArgument;
  }

  flagcxInnerComm_t innerComm = comm->homoComm;
  if (innerComm == nullptr) {
    return flagcxInternalError;
  }

  // Allocate the opaque handle
  flagcxDevComm_t handle =
      (flagcxDevComm_t)malloc(sizeof(struct flagcxDevCommInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevCommInternal));

  // Map opaque FlagCX requirements to NCCL requirements
  ncclDevCommRequirements ncclReqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ncclReqs.lsaBarrierCount = reqs->fields[0];
  ncclReqs.lsaMultimem = reqs->fields[1];
  ncclReqs.railGinBarrierCount = reqs->fields[2];
  ncclReqs.ginSignalCount = reqs->fields[3];

  flagcxResult_t ret =
      ncclAdaptorDevCommCreate(innerComm->base, &ncclReqs, &handle->ncclDev);
  if (ret != flagcxSuccess) {
    free(handle);
    return ret;
  }

  handle->barrierEpoch = 0;
  *devComm = handle;
  return flagcxSuccess;
}

flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm,
                                    flagcxDevComm_t devComm) {
  if (devComm == nullptr) {
    return flagcxSuccess;
  }
  if (comm == nullptr) {
    return flagcxInvalidArgument;
  }

  flagcxInnerComm_t innerComm = comm->homoComm;
  if (innerComm == nullptr) {
    free(devComm);
    return flagcxInternalError;
  }

  ncclAdaptorDevCommDestroy(innerComm->base, &devComm->ncclDev);

  free(devComm);
  return flagcxSuccess;
}

// ---------- DevMem: Tier 1 (window + IPC dual mode) ----------

flagcxResult_t flagcxDevMemCreate(flagcxComm_t comm, void *buff, size_t size,
                                  flagcxWindow_t win, flagcxDevMem_t *devMem) {
  if (comm == nullptr || buff == nullptr || size == 0 || devMem == nullptr) {
    return flagcxInvalidArgument;
  }

  flagcxDevMem_t handle =
      (flagcxDevMem_t)malloc(sizeof(struct flagcxDevMemInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevMemInternal));

  // Store local rank index for IPC pointer access
  handle->intraRank = comm->localRank;

  if (win != nullptr) {
    // Window mode: store ncclWindow_t, no IPC needed (NCCL handles pointers)
    handle->mode = flagcxDevMemWindow;
    handle->ncclWin = win->base;
    handle->winHandle = (void *)win;
  } else {
    // IPC mode: build peer pointer table via IPC exchange
    handle->mode = flagcxDevMemIpc;
    flagcxResult_t res =
        buildIpcPeerPointers(comm, buff, size, &handle->devPeerPtrs,
                             &handle->hostPeerPtrs, &handle->nPeers);
    if (res != flagcxSuccess) {
      free(handle);
      return res;
    }
  }

  *devMem = handle;
  return flagcxSuccess;
}

flagcxResult_t flagcxDevMemDestroy(flagcxComm_t comm, flagcxDevMem_t devMem) {
  if (devMem == nullptr) {
    return flagcxSuccess;
  }
  if (comm == nullptr) {
    return flagcxInvalidArgument;
  }

  // Clean up IPC peer pointers (only present in IPC mode)
  if (devMem->hostPeerPtrs) {
    void *ownBuff = devMem->hostPeerPtrs[comm->localRank];
    cleanupIpcPeerPointers(devMem->hostPeerPtrs, devMem->devPeerPtrs,
                           devMem->nPeers, ownBuff);
  }

  free(devMem);
  return flagcxSuccess;
}

#else // !FLAGCX_DEVICE_API_NCCL

// ==========================================================================
// Tier 2: Fallback — IPC-based barrier + peer pointer exchange
// ==========================================================================

flagcxResult_t flagcxDevCommCreate(flagcxComm_t comm,
                                   const flagcxDevCommRequirements *reqs,
                                   flagcxDevComm_t *devComm) {
  if (comm == nullptr || devComm == nullptr) {
    return flagcxInvalidArgument;
  }

  int myRank = comm->rank;
  int nRanks = comm->nranks;
  int localRank = comm->localRank;
  int localRanks = comm->localRanks;
  int *lrToR = comm->localRankToRank;

  // Allocate the opaque handle
  flagcxDevComm_t handle =
      (flagcxDevComm_t)malloc(sizeof(struct flagcxDevCommInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevCommInternal));

  // Populate rank info
  handle->rank = myRank;
  handle->nRanks = nRanks;
  handle->intraRank = localRank;
  handle->intraSize = localRanks;
  handle->nLocalRanks = localRanks;

  // Copy localRankToRank for cleanup
  handle->localRankToRank = (int *)malloc(localRanks * sizeof(int));
  if (handle->localRankToRank == nullptr) {
    free(handle);
    return flagcxSystemError;
  }
  memcpy(handle->localRankToRank, lrToR, localRanks * sizeof(int));

  // Step 1: Allocate local barrier flags (IPC-shareable device memory)
  struct flagcxP2pIpcDesc barrierIpcDesc;
  memset(&barrierIpcDesc, 0, sizeof(barrierIpcDesc));
  size_t barrierSize = FLAGCX_DEVICE_CTA_COUNT * sizeof(uint32_t);
  FLAGCXCHECK(flagcxP2pAllocateShareableBuffer(
      barrierSize, 0, &barrierIpcDesc, (void **)&handle->localBarrierFlags));

  // Zero barrier flags
  deviceAdaptor->deviceMemset(handle->localBarrierFlags, 0, barrierSize,
                              flagcxMemDevice, NULL);

  // Step 2: Exchange barrier IPC handles with all ranks
  struct flagcxP2pIpcDesc *allBarrierDescs = (struct flagcxP2pIpcDesc *)calloc(
      nRanks, sizeof(struct flagcxP2pIpcDesc));
  if (allBarrierDescs == nullptr) {
    deviceAdaptor->deviceFree(handle->localBarrierFlags, flagcxMemDevice, NULL);
    free(handle->localRankToRank);
    free(handle);
    return flagcxSystemError;
  }
  memcpy(&allBarrierDescs[myRank], &barrierIpcDesc,
         sizeof(struct flagcxP2pIpcDesc));
  FLAGCXCHECK(bootstrapAllGather(comm->bootstrap, allBarrierDescs,
                                 sizeof(struct flagcxP2pIpcDesc)));

  // Step 3: Open intra-node peer barrier IPC handles
  handle->peerBarrierPtrs = (void **)calloc(localRanks, sizeof(void *));
  if (handle->peerBarrierPtrs == nullptr) {
    free(allBarrierDescs);
    deviceAdaptor->deviceFree(handle->localBarrierFlags, flagcxMemDevice, NULL);
    free(handle->localRankToRank);
    free(handle);
    return flagcxSystemError;
  }
  for (int lr = 0; lr < localRanks; lr++) {
    int gr = lrToR[lr];
    if (gr == myRank) {
      handle->peerBarrierPtrs[lr] = handle->localBarrierFlags;
    } else {
      flagcxIpcMemHandle_t handlePtr =
          (flagcxIpcMemHandle_t)&allBarrierDescs[gr].handleData;
      FLAGCXCHECK(deviceAdaptor->ipcMemHandleOpen(
          handlePtr, &handle->peerBarrierPtrs[lr]));
    }
  }
  free(allBarrierDescs);

  // Step 4: Build device barrier pointer array
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&handle->barrierPeers,
                                          localRanks * sizeof(uint32_t *),
                                          flagcxMemDevice, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      handle->barrierPeers, handle->peerBarrierPtrs,
      localRanks * sizeof(uint32_t *), flagcxMemcpyHostToDevice, NULL, NULL));

  handle->barrierEpoch = 0;

  *devComm = handle;
  return flagcxSuccess;
}

flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm,
                                    flagcxDevComm_t devComm) {
  if (devComm == nullptr) {
    return flagcxSuccess;
  }

  // Close peer barrier IPC handles
  if (devComm->peerBarrierPtrs) {
    for (int i = 0; i < devComm->nLocalRanks; i++) {
      if (devComm->peerBarrierPtrs[i] &&
          devComm->peerBarrierPtrs[i] != devComm->localBarrierFlags) {
        deviceAdaptor->ipcMemHandleClose(devComm->peerBarrierPtrs[i]);
      }
    }
    free(devComm->peerBarrierPtrs);
  }

  // Free device barrier pointer array
  if (devComm->barrierPeers) {
    deviceAdaptor->deviceFree(devComm->barrierPeers, flagcxMemDevice, NULL);
  }

  // Free local barrier flags
  if (devComm->localBarrierFlags) {
    deviceAdaptor->deviceFree(devComm->localBarrierFlags, flagcxMemDevice,
                              NULL);
  }

  free(devComm->localRankToRank);
  free(devComm);
  return flagcxSuccess;
}

// ---------- DevMem: Tier 2 (IPC-only, win param ignored) ----------

flagcxResult_t flagcxDevMemCreate(flagcxComm_t comm, void *buff, size_t size,
                                  flagcxWindow_t win, flagcxDevMem_t *devMem) {
  (void)win; // Tier 2: window mode not available, always IPC
  if (comm == nullptr || buff == nullptr || size == 0 || devMem == nullptr) {
    return flagcxInvalidArgument;
  }
  if (win != nullptr) {
    // Window mode requires NCCL > 2.28 (Tier 1)
    return flagcxInvalidArgument;
  }

  flagcxDevMem_t handle =
      (flagcxDevMem_t)malloc(sizeof(struct flagcxDevMemInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevMemInternal));

  handle->mode = flagcxDevMemIpc;
  handle->basePtr = buff;

  // Store local rank index for IPC pointer access
  handle->intraRank = comm->localRank;

  flagcxResult_t res =
      buildIpcPeerPointers(comm, buff, size, &handle->devPeerPtrs,
                           &handle->hostPeerPtrs, &handle->nPeers);
  if (res != flagcxSuccess) {
    free(handle);
    return res;
  }

  *devMem = handle;
  return flagcxSuccess;
}

flagcxResult_t flagcxDevMemDestroy(flagcxComm_t comm, flagcxDevMem_t devMem) {
  if (devMem == nullptr) {
    return flagcxSuccess;
  }

  cleanupIpcPeerPointers(devMem->hostPeerPtrs, devMem->devPeerPtrs,
                         devMem->nPeers, devMem->basePtr);

  free(devMem);
  return flagcxSuccess;
}

#endif // FLAGCX_DEVICE_API_NCCL