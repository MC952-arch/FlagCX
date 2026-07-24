/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM Device API backend for flagcxDevComm lifecycle.
 * Linked when USE_SHMEM=1.
 ************************************************************************/

#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "device_api/nvshmem_comm_traits.h"
#include "nvshmem_adaptor.h"
#include "shmem_adaptor.h"

#include <cstddef>

// Verify that flagcxShmemCommInternal and CommTraits<NvshmemBackend>::Comm have
// compatible layout — the constructor in flagcx_device_core.h does a raw cast.
static_assert(
    sizeof(flagcxShmemCommInternal) == sizeof(CommTraits<NvshmemBackend>::Comm),
    "ShmemCommInternal and CommTraits<NvshmemBackend>::Comm size mismatch");
static_assert(
    offsetof(flagcxShmemCommInternal, worldBarrierCount) ==
        offsetof(CommTraits<NvshmemBackend>::Comm, worldBarrierCount),
    "ShmemCommInternal and CommTraits::Comm last-field offset mismatch");

static flagcxResult_t
nvshmemDevApiCommCreate(flagcxComm_t comm,
                        const struct flagcxDevCommRequirements *reqs,
                        flagcxDevComm_t handle) {
  if (shmemAdaptor == nullptr) {
    return flagcxInternalError;
  }

  // Initialize NVSHMEM (reference-counted, safe to call multiple times)
  flagcxResult_t ret = shmemAdaptor->init(comm->rank, comm->nranks);
  if (ret != flagcxSuccess) {
    return ret;
  }

  flagcxShmemComm_t shmemComm = nullptr;
  ret = shmemAdaptor->devCommCreate(comm, reqs, &shmemComm);
  if (ret != flagcxSuccess) {
    shmemAdaptor->finalize();
    return ret;
  }

  handle->devComm = (flagcxInnerDevComm_t)shmemComm;
  handle->signalBuffer = shmemComm->signalBuffer;
  handle->shadowBuffer = shmemComm->shadowBuffer;
  handle->counterBuffer = shmemComm->counterBuffer;
  handle->signalCount = shmemComm->signalCount;
  handle->counterCount = shmemComm->counterCount;
  // NVSHMEM does not use FIFO contexts — leave contextCount at 0.
  handle->contextCount = 0;
  // NVSHMEM handles inter-node transparently — no relay peers needed.
  handle->nInterPeers = 0;

  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiCommDestroy(flagcxComm_t comm,
                                               flagcxDevComm_t devComm) {
  (void)comm;
  if (shmemAdaptor != nullptr && devComm->devComm != nullptr) {
    shmemAdaptor->devCommDestroy((flagcxShmemComm_t)devComm->devComm);
    devComm->devComm = nullptr;
    shmemAdaptor->finalize();
  }
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiMemCreate(flagcxComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t win,
                                             flagcxDevMem_t handle) {
  (void)comm;
  (void)win;
  using Window = CommTraits<NvshmemBackend>::Window;
  auto *w = new Window();
  w->symBase = buff;
  w->allocSize = size;
  w->rawPtr = buff;
  handle->window = (void *)w;
  handle->hasWindow = true;
  handle->isSymmetric = true;
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiMemDestroy(flagcxComm_t comm,
                                              flagcxDevMem_t devMem) {
  (void)comm;
  if (devMem->window) {
    delete (CommTraits<NvshmemBackend>::Window *)devMem->window;
    devMem->window = nullptr;
  }
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
                                                    void **devPtr) {
  (void)devComm;
  (void)devPtr;
  return flagcxNotSupported;
}

static flagcxResult_t nvshmemDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
  (void)devComm;
  return flagcxNotSupported;
}

static flagcxResult_t nvshmemDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
                                                   void **devPtr) {
  (void)devMem;
  (void)devPtr;
  return flagcxNotSupported;
}

static flagcxResult_t nvshmemDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
  (void)devMem;
  return flagcxNotSupported;
}

static flagcxResult_t nvshmemCommCleanup(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

static struct flagcxDevApiBackend nvshmemBackend = {
    .name = "nvshmem",
    .devCommCreate = nvshmemDevApiCommCreate,
    .devCommDestroy = nvshmemDevApiCommDestroy,
    .devMemCreate = nvshmemDevApiMemCreate,
    .devMemDestroy = nvshmemDevApiMemDestroy,
    .devCommGetDevicePtr = nvshmemDevApiCommGetDevicePtr,
    .devCommFreeDevicePtr = nvshmemDevApiCommFreeDevicePtr,
    .devMemGetDevicePtr = nvshmemDevApiMemGetDevicePtr,
    .devMemFreeDevicePtr = nvshmemDevApiMemFreeDevicePtr,
    .commCleanup = nvshmemCommCleanup,
};

struct flagcxDevApiBackend *devApiBackend = &nvshmemBackend;
