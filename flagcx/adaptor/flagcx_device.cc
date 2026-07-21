/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-side lifecycle management for flagcxDevComm_t and flagcxDevMem_t.
 * Thin dispatcher: delegates all backend-specific logic via devApiBackend.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "comm.h"
#include "flagcx_kernel.h"
#include "proxy.h"
#include <cstring>
#include <pthread.h>
#include <sched.h>

#include "dev_api_backend.h"

// ==========================================================================
// DevComm lifecycle
// ==========================================================================

extern "C" flagcxResult_t
flagcxDevCommCreate(flagcxComm_t comm, const flagcxDevCommRequirements *reqs,
                    flagcxDevComm_t *devComm) {
  if (comm == nullptr || reqs == nullptr || devComm == nullptr) {
    return flagcxInvalidArgument;
  }

  flagcxDevComm_t handle =
      (flagcxDevComm_t)malloc(sizeof(struct flagcxDevCommInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevCommInternal));
  pthread_mutex_init(&handle->cachedPtrMutex, NULL);
  handle->barrierIpcIndex = -1;

  // Baseline: always
  handle->rank = comm->rank;
  handle->nRanks = comm->nranks;
  handle->intraRank = comm->localRank;
  handle->intraSize = comm->localRanks;
  {
    int ctxCount = (reqs->interContextCount > 0) ? reqs->interContextCount : 1;
    if (comm->heteroComm != nullptr &&
        comm->heteroComm->proxyState != nullptr) {
      int available = comm->heteroComm->proxyState->kernelState.contextCount;
      if (available > 0 && ctxCount > available)
        ctxCount = available;
    }
    if (ctxCount > FLAGCX_DEVICE_CTA_COUNT)
      ctxCount = FLAGCX_DEVICE_CTA_COUNT;
    handle->contextCount = ctxCount;
    for (int i = 0; i < ctxCount; i++) {
      handle->fifoBuffers[i] = (comm->heteroComm != nullptr)
                                   ? comm->heteroComm->fifoBuffers[i]
                                   : nullptr;
    }
  }

  // Backend-specific creation
  {
    flagcxResult_t ret = devApiBackend->devCommCreate(comm, reqs, handle);
    if (ret != flagcxSuccess) {
      WARN("flagcxDevCommCreate: %s backend failed (%d)", devApiBackend->name,
           ret);
      pthread_mutex_destroy(&handle->cachedPtrMutex);
      free(handle);
      return ret;
    }
  }

  *devComm = handle;

  // Publish to heteroComm so proxy thread can access this DevComm
  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero != nullptr) {
    hetero->devCommHandle = handle;
  }

  return flagcxSuccess;
}

extern "C" flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm,
                                               flagcxDevComm_t devComm) {
  if (devComm == nullptr) {
    return flagcxSuccess;
  }

  devApiBackend->devCommDestroy(comm, devComm);

  // Free cached device pointers (thin-layer responsibility)
  if (devComm->cachedNetContextsPtr) {
    flagcxCommDeferFree(comm, devComm->cachedNetContextsPtr, flagcxMemDevice);
  }
  if (devComm->cachedDevicePtr) {
    flagcxCommDeferFree(comm, devComm->cachedDevicePtr, flagcxMemDevice);
  }

  pthread_mutex_destroy(&devComm->cachedPtrMutex);
  free(devComm);
  return flagcxSuccess;
}

// ==========================================================================
// DevMem lifecycle
// ==========================================================================

extern "C" flagcxResult_t flagcxDevMemCreate(flagcxComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t win,
                                             flagcxDevMem_t *devMem) {
  if (comm == nullptr || devMem == nullptr) {
    return flagcxInvalidArgument;
  }

  flagcxDevMem_t handle =
      (flagcxDevMem_t)malloc(sizeof(struct flagcxDevMemInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevMemInternal));
  pthread_mutex_init(&handle->cachedPtrMutex, NULL);

  // Baseline: always
  handle->rawPtr = buff;
  handle->ipcIndex = -1;

  // Backend-specific creation
  {
    flagcxResult_t ret =
        devApiBackend->devMemCreate(comm, buff, size, win, handle);
    if (ret != flagcxSuccess) {
      WARN("flagcxDevMemCreate: %s backend failed (%d)", devApiBackend->name,
           ret);
      pthread_mutex_destroy(&handle->cachedPtrMutex);
      free(handle);
      return ret;
    }
  }

  *devMem = handle;
  return flagcxSuccess;
}

extern "C" flagcxResult_t flagcxDevMemDestroy(flagcxComm_t comm,
                                              flagcxDevMem_t devMem) {
  if (devMem == nullptr) {
    return flagcxSuccess;
  }

  // Release IPC table slot (resources moved to deferred queue)
  if (devMem->ipcIndex >= 0) {
    releaseIpcTableSlot(comm, devMem->ipcIndex);
  }

  devApiBackend->devMemDestroy(comm, devMem);

  // Free cached device pointer
  if (devMem->cachedDevicePtr) {
    flagcxCommDeferFree(comm, devMem->cachedDevicePtr, flagcxMemDevice);
  }

  pthread_mutex_destroy(&devMem->cachedPtrMutex);
  free(devMem);
  return flagcxSuccess;
}

// ==========================================================================
// Device Pointer API
// ==========================================================================

extern "C" flagcxResult_t flagcxDevCommGetDevicePtr(flagcxDevComm_t devComm,
                                                    void **devPtr) {
  return devApiBackend->devCommGetDevicePtr(devComm, devPtr);
}

extern "C" flagcxResult_t flagcxDevCommFreeDevicePtr(flagcxDevComm_t devComm) {
  return devApiBackend->devCommFreeDevicePtr(devComm);
}

extern "C" flagcxResult_t flagcxDevMemGetDevicePtr(flagcxDevMem_t devMem,
                                                   void **devPtr) {
  return devApiBackend->devMemGetDevicePtr(devMem, devPtr);
}

extern "C" flagcxResult_t flagcxDevMemFreeDevicePtr(flagcxDevMem_t devMem) {
  return devApiBackend->devMemFreeDevicePtr(devMem);
}

// ==========================================================================
// Comm-level cleanup — called from flagcxCommDestroy in flagcx.cc
// ==========================================================================

extern "C" flagcxResult_t flagcxCommCleanup(flagcxComm_t comm) {
  return devApiBackend->commCleanup(comm);
}

// ==========================================================================
// IPC table cleanup — called from flagcxCommDestroy after heteroComm destroy
// ==========================================================================

flagcxResult_t flagcxCommCleanupIpcTable(flagcxComm_t comm) {
  if (comm == nullptr) {
    return flagcxSuccess;
  }

  for (int k = 0; k < FLAGCX_MAX_IPC_ENTRIES; k++) {
    struct flagcxIpcTableEntry *e = &comm->ipcTable[k];
    if (e->hostPeerPtrs == nullptr && e->devPeerPtrs == nullptr) {
      continue; // empty slot
    }

    if (e->inUse) {
      WARN("flagcxCommCleanupIpcTable: entry %d still in use — "
           "flagcxDevMemDestroy should be called before flagcxCommDestroy",
           k);
    }

    // Close IPC handles
    if (e->hostPeerPtrs) {
      for (int i = 0; i < e->nPeers; i++) {
        if (e->hostPeerPtrs[i] && e->hostPeerPtrs[i] != e->basePtr) {
          deviceAdaptor->ipcMemHandleClose(e->hostPeerPtrs[i]);
        }
      }
      free(e->hostPeerPtrs);
      e->hostPeerPtrs = nullptr;
    }

    // Free device memory safely
    if (e->devPeerPtrs) {
      deviceAdaptor->deviceFree(e->devPeerPtrs, flagcxMemDevice, NULL);
      e->devPeerPtrs = nullptr;
    }

    e->inUse = false;
  }

  return flagcxSuccess;
}

// ==========================================================================
// Deferred IPC table slot release.
// ==========================================================================
void releaseIpcTableSlot(flagcxComm_t comm, int slot) {
  if (comm == nullptr || slot < 0 || slot >= FLAGCX_MAX_IPC_ENTRIES) {
    return;
  }
  struct flagcxIpcTableEntry *e = &comm->ipcTable[slot];
  if (e->hostPeerPtrs == nullptr && e->devPeerPtrs == nullptr) {
    e->inUse = false;
    return;
  }

  // Move resources to deferred linked list for cleanup at comm destroy
  struct flagcxDeferredIpcEntry *d =
      (struct flagcxDeferredIpcEntry *)malloc(sizeof(*d));
  if (d == nullptr) {
    // OOM: leave slot occupied so flagcxCommCleanupIpcTable handles it at
    // destroy. The slot won't be reusable, but resources are still safe.
    WARN(
        "releaseIpcTableSlot: OOM, keeping slot %d occupied until comm destroy",
        slot);
    e->inUse = false;
    return;
  }
  d->hostPeerPtrs = e->hostPeerPtrs;
  d->devPeerPtrs = e->devPeerPtrs;
  d->nPeers = e->nPeers;
  d->basePtr = e->basePtr;
  d->next = nullptr;
  flagcxIntruQueueEnqueue(&comm->deferredIpcQueue, d);

  // Clear slot — now reusable by buildIpcPeerPointers
  e->hostPeerPtrs = nullptr;
  e->devPeerPtrs = nullptr;
  e->nPeers = 0;
  e->basePtr = nullptr;
  e->inUse = false;
}

flagcxResult_t flagcxCommDrainDeferredIpc(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  while (!flagcxIntruQueueEmpty(&comm->deferredIpcQueue)) {
    struct flagcxDeferredIpcEntry *d =
        flagcxIntruQueueDequeue(&comm->deferredIpcQueue);
    if (d->hostPeerPtrs) {
      for (int j = 0; j < d->nPeers; j++) {
        if (d->hostPeerPtrs[j] && d->hostPeerPtrs[j] != d->basePtr)
          deviceAdaptor->ipcMemHandleClose(d->hostPeerPtrs[j]);
      }
      free(d->hostPeerPtrs);
    }
    if (d->devPeerPtrs)
      deviceAdaptor->deviceFree(d->devPeerPtrs, flagcxMemDevice, NULL);
    free(d);
  }
  return flagcxSuccess;
}

// ==========================================================================
// Deferred device/host-pinned memory free.
// ==========================================================================
void flagcxCommDeferFree(flagcxComm_t comm, void *ptr, int memType) {
  if (comm == nullptr || ptr == nullptr)
    return;
  if (comm->deferredFreeCount >= FLAGCX_MAX_DEFERRED_FREES) {
    WARN("flagcxCommDeferFree: deferred free list full (%d), freeing now",
         FLAGCX_MAX_DEFERRED_FREES);
    deviceAdaptor->deviceFree(ptr, (flagcxMemType_t)memType, NULL);
    return;
  }
  comm->deferredFrees[comm->deferredFreeCount].ptr = ptr;
  comm->deferredFrees[comm->deferredFreeCount].memType = memType;
  comm->deferredFreeCount++;
}

flagcxResult_t flagcxCommDrainDeferredFrees(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  for (int i = 0; i < comm->deferredFreeCount; i++) {
    struct flagcxDeferredFree *d = &comm->deferredFrees[i];
    if (d->ptr) {
      deviceAdaptor->deviceFree(d->ptr, (flagcxMemType_t)d->memType, NULL);
      d->ptr = nullptr;
    }
  }
  comm->deferredFreeCount = 0;
  return flagcxSuccess;
}

flagcxResult_t flagcxCommDrainDeferredBuffers(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  while (!flagcxIntruQueueEmpty(&comm->deferredBufferQueue)) {
    struct flagcxDevCommBufferHandle *h =
        flagcxIntruQueueDequeue(&comm->deferredBufferQueue);
    if (h->localBarrierFlags)
      deviceAdaptor->deviceFree(h->localBarrierFlags, flagcxMemDevice, NULL);
    if (h->epochBuffer)
      deviceAdaptor->deviceFree(h->epochBuffer, flagcxMemDevice, NULL);
    if (h->signalBuffer) {
      if (h->signalHostEnable)
        deviceAdaptor->deviceFree(h->signalBuffer, flagcxMemHost, NULL);
      else
        deviceAdaptor->gdrMemFree(h->signalBuffer, NULL);
    }
    if (h->shadowBuffer)
      deviceAdaptor->deviceFree(h->shadowBuffer, flagcxMemDevice, NULL);
    if (h->counterBuffer)
      deviceAdaptor->deviceFree(h->counterBuffer, flagcxMemHost, NULL);
    if (h->putValueStagingBuffer)
      deviceAdaptor->deviceFree(h->putValueStagingBuffer, flagcxMemHost, NULL);
    free(h);
  }
  comm->deferredBufferCount = 0;
  return flagcxSuccess;
}

// ==========================================================================
// Communicator property query
// ==========================================================================

flagcxResult_t flagcxCommQueryProperties(flagcxComm_t comm,
                                         flagcxCommProperties_t *props) {
  if (comm == nullptr || props == nullptr) {
    return flagcxInvalidArgument;
  }
  memset(props, 0, sizeof(*props));

  // Baseline fields (always available)
  props->rank = comm->rank;
  props->nRanks = comm->nranks;
  props->deviceId = comm->heteroComm ? comm->heteroComm->cudaDev : -1;

  // Query multicast support via adaptor
#ifdef FLAGCX_DEVICE_API_VENDOR
  props->vendorDeviceApiSupport = true;
#else
  props->vendorDeviceApiSupport = false;
#endif
  int mcSupported = 0;
  if (deviceAdaptor->symMulticastSupported)
    deviceAdaptor->symMulticastSupported(&mcSupported);
  props->multicastSupport = (mcSupported != 0);
  props->netType = flagcxNetTypeNone;

  return flagcxSuccess;
}

// ==========================================================================
// Barrier requirement stubs (resource-handle model not yet implemented)
// ==========================================================================

flagcxResult_t
flagcxIntraBarrierCreateRequirement(flagcxTeam_t team, int nBarriers,
                                    flagcxIntraBarrierHandle_t *outHandle,
                                    flagcxDevCommRequirements *outReq) {
  (void)team;
  (void)nBarriers;
  (void)outHandle;
  (void)outReq;
  return flagcxNotSupported;
}

flagcxResult_t flagcxInterBarrierCreateRequirement(
    flagcxComm_t comm, flagcxTeam_t team, int nBarriers,
    flagcxInterBarrierHandle_t *outHandle, flagcxDevCommRequirements *outReq) {
  (void)comm;
  (void)team;
  (void)nBarriers;
  (void)outHandle;
  (void)outReq;
  return flagcxNotSupported;
}
