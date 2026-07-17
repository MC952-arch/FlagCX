/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM Device API backend for flagcxDevComm lifecycle.
 * Linked when USE_SHMEM=1.
 ************************************************************************/

#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "nvshmem_adaptor.h"
#include "shmem_adaptor.h"

static flagcxResult_t
nvshmemDevApiCommCreate(flagcxComm_t comm,
                        const struct flagcxDevCommRequirements *reqs,
                        flagcxDevComm_t handle) {
  if (shmemAdaptor == nullptr) {
    return flagcxInternalError;
  }

  flagcxShmemComm_t shmemComm = nullptr;
  flagcxResult_t ret = shmemAdaptor->devCommCreate(comm, reqs, &shmemComm);
  if (ret != flagcxSuccess) {
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

  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiCommDestroy(flagcxComm_t comm,
                                               flagcxDevComm_t devComm) {
  (void)comm;
  if (shmemAdaptor != nullptr && devComm->devComm != nullptr) {
    shmemAdaptor->devCommDestroy((flagcxShmemComm_t)devComm->devComm);
    devComm->devComm = nullptr;
  }
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiMemCreate(flagcxComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t win,
                                             flagcxDevMem_t handle) {
  (void)comm;
  (void)buff;
  (void)size;
  (void)win;
  (void)handle;
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiMemDestroy(flagcxComm_t comm,
                                              flagcxDevMem_t devMem) {
  (void)comm;
  (void)devMem;
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
