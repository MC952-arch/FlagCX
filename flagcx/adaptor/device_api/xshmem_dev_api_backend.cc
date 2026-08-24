/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * XSHMEM Device API backend for flagcxDevComm lifecycle.
 * Linked when USE_KUNLUNXIN_ADAPTOR + USE_SHMEM=1.
 ************************************************************************/

#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "shmem_adaptor.h"
#include "xshmem_adaptor.h" // flagcxShmemCommInternal (sc->signalBuffer, ...)

// ==========================================================================
// DevComm lifecycle
// ==========================================================================
static flagcxResult_t
xshmemDevApiCommCreate(flagcxComm_t comm,
                       const struct flagcxDevCommRequirements *reqs,
                       flagcxDevComm_t devComm) {
  if (shmemAdaptor == nullptr) {
    return flagcxInternalError;
  }

  // Initialize xshmem against the BKCL context (reference-counted). This is
  // what makes xshmem_get_xshmemi_device_state_h() valid on the device side.
  flagcxResult_t ret = shmemAdaptor->init(comm);
  if (ret != flagcxSuccess) {
    return ret;
  }

  flagcxShmemComm_t shmemComm = nullptr;
  ret = shmemAdaptor->devCommCreate(comm, reqs, &shmemComm);
  if (ret != flagcxSuccess) {
    shmemAdaptor->finalize();
    return ret;
  }

  // Baseline rank/nRanks/intra* are already filled by flagcxDevCommCreate.
  // Attach the shmem state and expose its buffers (unused by the current
  // native kernel, but kept consistent for future device-API kernels).
  devComm->devComm = (flagcxInnerDevComm_t)shmemComm;
  devComm->signalBuffer = shmemComm->signalBuffer;
  devComm->shadowBuffer = shmemComm->shadowBuffer;
  devComm->counterBuffer = shmemComm->counterBuffer;
  devComm->signalCount = shmemComm->signalCount;
  devComm->counterCount = shmemComm->counterCount;
  devComm->contextCount = 1;
  // Single-node intra path only (no inter-node relay); see xshmem_adaptor.cc.
  devComm->nInterPeers = 0;

  return flagcxSuccess;
}

static flagcxResult_t xshmemDevApiCommDestroy(flagcxComm_t comm,
                                              flagcxDevComm_t devComm) {
  (void)comm;
  if (shmemAdaptor != nullptr && devComm->devComm != nullptr) {
    shmemAdaptor->devCommDestroy((flagcxShmemComm_t)devComm->devComm);
    devComm->devComm = nullptr;
    shmemAdaptor->finalize();
  }
  return flagcxSuccess;
}

// ==========================================================================
// DevMem lifecycle
// ==========================================================================
static flagcxResult_t xshmemDevApiMemCreate(flagcxComm_t comm, void *buff,
                                            size_t size, flagcxWindow_t win,
                                            flagcxDevMem_t devMem) {
  (void)comm;
  (void)size;
  (void)win;
  // rawPtr is already set by flagcxDevMemCreate. The native xshmem kernel
  // reads rawPtr directly and does its own symmetric-heap scratch management,
  // so no window / peer-pointer layer is needed here.
  devMem->window = nullptr;
  devMem->hasWindow = false;
  devMem->isSymmetric = false;
  if (buff == nullptr)
    return flagcxInvalidArgument;
  return flagcxSuccess;
}

static flagcxResult_t xshmemDevApiMemDestroy(flagcxComm_t comm,
                                             flagcxDevMem_t devMem) {
  (void)comm;
  (void)devMem;
  return flagcxSuccess;
}

// ==========================================================================
// Device pointer materialization
// ==========================================================================
static flagcxResult_t xshmemDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
                                                   void **devPtr) {
  (void)devComm;
  (void)devPtr;
  return flagcxNotSupported;
}

static flagcxResult_t xshmemDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
  (void)devComm;
  return flagcxSuccess;
}

static flagcxResult_t xshmemDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
                                                  void **devPtr) {
  (void)devMem;
  (void)devPtr;
  return flagcxNotSupported;
}

static flagcxResult_t xshmemDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
  (void)devMem;
  return flagcxSuccess;
}

static flagcxResult_t xshmemDevApiCommCleanup(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

static struct flagcxDevApiBackend xshmemBackend = {
    .name = "xshmem",
    .devCommCreate = xshmemDevApiCommCreate,
    .devCommDestroy = xshmemDevApiCommDestroy,
    .devMemCreate = xshmemDevApiMemCreate,
    .devMemDestroy = xshmemDevApiMemDestroy,
    .devCommGetDevicePtr = xshmemDevApiCommGetDevicePtr,
    .devCommFreeDevicePtr = xshmemDevApiCommFreeDevicePtr,
    .devMemGetDevicePtr = xshmemDevApiMemGetDevicePtr,
    .devMemFreeDevicePtr = xshmemDevApiMemFreeDevicePtr,
    .commCleanup = xshmemDevApiCommCleanup,
};

struct flagcxDevApiBackend *devApiBackend = &xshmemBackend;
