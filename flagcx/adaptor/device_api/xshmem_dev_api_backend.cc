/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * XSHMEM Device API backend for flagcxDevComm / flagcxDevMem lifecycle
 * on Kunlunxin (P800). Linked when USE_KUNLUNXIN_ADAPTOR + USE_SHMEM=1.
 *
 * This backend deliberately does NOT use the device-side comm_traits
 * machinery (CommTraits<...>::Comm / ::Window). The native xshmem test
 * kernels (test/kernel/klx/device_api.xpu) read only the always-populated
 * baseline fields of the handles:
 *   - flagcxDevMemInternal::rawPtr        (set by flagcxDevMemCreate)
 *   - flagcxDevCommInternal::{rank,nRanks,intraRank,intraSize}
 *                                         (set by flagcxDevCommCreate)
 * So all this backend must do is:
 *   1. initialize xshmem (via the shmem adaptor) so the device state handle
 *      returned by xshmem_get_xshmemi_device_state_h() is valid, and
 *   2. keep the shmem comm state alive for the DevComm's lifetime.
 *
 * The trait-based device-pointer materialization (GetDevicePtr) is left
 * unsupported on purpose — the native kernels launch with raw pointers and
 * never ask FlagCX to build a device-resident handle.
 ************************************************************************/

#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "shmem_adaptor.h"
#include "xshmem_adaptor.h" // flagcxShmemCommInternal (sc->signalBuffer, ...)

#include <cstdio>
#include <pthread.h>
#include <xpu/runtime.h> // XPUStream/XPUEvent, xpu_event_*, xpu_stream_wait_event
#include <xshmem/xshmem.h> // xshmem_calloc / xshmem_free

// ==========================================================================
// Kernel scratch owned by the DevComm
// ==========================================================================
// The native kernels need symmetric buffers the registered FlagCX buffer does
// not provide: a remotely-written C2C put target plus a signal (AMO) array.
// Those, the monotonic tile stamp and the completion event are
// per-(DevComm, stream) state with the DevComm's lifetime, so they belong HERE
// rather than in the kernel translation unit: this is the layer that has a
// destroy hook (xshmemDevApiCommDestroy).
//
// Only the registry is shared, so only the registry is locked. A context body
// needs no lock: one (DevComm, stream) pair is driven by its own caller, and
// reuse of its scratch is ordered on the DEVICE via `event`.
namespace {

struct XshmemArScratch {
  float *workPing = nullptr;  // symmetric C2C put targets, workElems * npes
  float *workPong = nullptr;  // floats EACH (pingpong across tiles)
  uint64_t *signal = nullptr; // 2 * clusternum * npes slots
  long workElems = 0;
};

struct XshmemA2AScratch {
  float *stage = nullptr;     // 2 halves x stageElems x npes floats
  uint64_t *signal = nullptr; // 2 * clusternum * npes slots
  long stageElems = 0;
};

struct XshmemKernelCtx {
  flagcxDevComm_t devComm = nullptr;
  XPUStream stream = nullptr;

  XshmemArScratch ar;
  long arSeq = 0;
  int arNpes = 0;
  int arClusternum = 0;
  XPUEvent arEvent = nullptr;

  XshmemA2AScratch a2a;
  long a2aSeq = 0;
  int a2aNpes = 0;
  int a2aClusternum = 0;
  XPUEvent a2aEvent = nullptr;
};

// One entry per (DevComm, stream). Entries are released on DevComm destroy, so
// this only has to cover the streams alive at the same time.
const int XSHMEM_MAX_KERNEL_CTX = 64;
XshmemKernelCtx g_kctx[XSHMEM_MAX_KERNEL_CTX];
int g_kctxCount = 0;
pthread_mutex_t g_kctxLock = PTHREAD_MUTEX_INITIALIZER;

// Reverse allocation order: the symmetric heap is a bump allocator, so freeing
// last-allocated-first is what lets it actually reclaim.
void freeArScratch(XshmemArScratch *s) {
  if (s->signal)
    xshmem_free(s->signal);
  if (s->workPong)
    xshmem_free(s->workPong);
  if (s->workPing)
    xshmem_free(s->workPing);
  *s = XshmemArScratch();
}

void freeA2AScratch(XshmemA2AScratch *s) {
  if (s->signal)
    xshmem_free(s->signal);
  if (s->stage)
    xshmem_free(s->stage);
  *s = XshmemA2AScratch();
}

// Drain before freeing: the last kernel using these buffers may still be
// running, and ordering (xpu_stream_wait_event) is not enough for a free.
void drainEvent(XPUEvent *ev) {
  if (*ev == nullptr)
    return;
  xpu_event_wait(*ev);
  xpu_event_destroy(*ev);
  *ev = nullptr;
}

void releaseCtx(XshmemKernelCtx *c) {
  drainEvent(&c->arEvent);
  drainEvent(&c->a2aEvent);
  freeA2AScratch(&c->a2a);
  freeArScratch(&c->ar);
  *c = XshmemKernelCtx();
}

// nullptr only if the table is exhausted.
XshmemKernelCtx *getCtx(flagcxDevComm_t devComm, XPUStream stream) {
  pthread_mutex_lock(&g_kctxLock);
  XshmemKernelCtx *ctx = nullptr;
  for (int i = 0; i < g_kctxCount; ++i) {
    if (g_kctx[i].devComm == devComm && g_kctx[i].stream == stream) {
      ctx = &g_kctx[i];
      break;
    }
  }
  if (ctx == nullptr && g_kctxCount < XSHMEM_MAX_KERNEL_CTX) {
    ctx = &g_kctx[g_kctxCount++];
    ctx->devComm = devComm;
    ctx->stream = stream;
  }
  pthread_mutex_unlock(&g_kctxLock);
  return ctx;
}

// Release every context of this DevComm, from xshmemDevApiCommDestroy, i.e.
// while the device runtime is still alive.
void releaseCtxsFor(flagcxDevComm_t devComm) {
  pthread_mutex_lock(&g_kctxLock);
  for (int i = 0; i < g_kctxCount;) {
    if (g_kctx[i].devComm == devComm) {
      releaseCtx(&g_kctx[i]);
      g_kctx[i] = g_kctx[--g_kctxCount];
      g_kctx[g_kctxCount] = XshmemKernelCtx();
    } else {
      ++i;
    }
  }
  pthread_mutex_unlock(&g_kctxLock);
}

} // namespace

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
  // Release the kernel scratch FIRST: xshmem_free must run before the shmem
  // comm (and with it the symmetric heap) is torn down below.
  releaseCtxsFor(devComm);
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
// Device pointer materialization — unsupported on the native path.
// ==========================================================================
static flagcxResult_t xshmemDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
                                                   void **devPtr) {
  (void)devComm;
  (void)devPtr;
  // Native xshmem kernels launch with raw pointers; they never consume a
  // device-resident flagcxDevComm value. Trait-based materialization would
  // require CommTraits, which this backend intentionally avoids.
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

// ==========================================================================
// Kernel-facing ABI (consumed by test/kernel/klx/device_api.xpu)
// ==========================================================================
// The launchers live in a separate translation unit compiled by the XTDK
// clang, so the scratch is handed over through a plain C interface instead of
// the C++ structs above. `acquire` returns the buffers to launch with, tells
// the caller whether a cross-PE rendezvous is required (only a FRESH
// allocation can race a peer still zeroing its signal array), and orders a
// reuse after the previous launch on the same buffers. `commit` advances the
// stamp and records the completion event that the NEXT acquire waits on.
extern "C" {

int flagcxXshmemAcquireArScratch(flagcxDevComm_t devComm, void *stream,
                                 int npes, int clusternum, long slotElems,
                                 float **workPing, float **workPong,
                                 uint64_t **signal, long *seqBase,
                                 int *needRendezvous) {
  XPUStream cs = (XPUStream)stream;
  XshmemKernelCtx *ctx = getCtx(devComm, cs);
  if (ctx == nullptr)
    return 1;

  bool needAlloc = (ctx->ar.workPing == nullptr) || (npes != ctx->arNpes) ||
                   (clusternum != ctx->arClusternum) ||
                   (slotElems > ctx->ar.workElems);
  if (needAlloc) {
    drainEvent(&ctx->arEvent); // previous kernel may still read these buffers
    freeArScratch(&ctx->ar);
    ctx->ar.workPing =
        (float *)xshmem_calloc((size_t)slotElems * npes, sizeof(float), 1024);
    ctx->ar.workPong =
        (float *)xshmem_calloc((size_t)slotElems * npes, sizeof(float), 1024);
    ctx->ar.signal = (uint64_t *)xshmem_calloc(
        (size_t)2 * clusternum * npes, sizeof(uint64_t), sizeof(uint64_t));
    if (!ctx->ar.workPing || !ctx->ar.workPong || !ctx->ar.signal) {
      freeArScratch(&ctx->ar);
      return 1;
    }
    ctx->ar.workElems = slotElems;
    ctx->arNpes = npes;
    ctx->arClusternum = clusternum;
    // Fresh, zeroed signal array => stamps restart. Every rank reallocs at the
    // same points (same count sequence), so the reset is collective.
    ctx->arSeq = 0;
  } else if (ctx->arEvent != nullptr) {
    // Reuse: device-side dependency on the previous launch, host not blocked.
    xpu_stream_wait_event(cs, ctx->arEvent);
  }

  *workPing = ctx->ar.workPing;
  *workPong = ctx->ar.workPong;
  *signal = ctx->ar.signal;
  *seqBase = ctx->arSeq;
  *needRendezvous = needAlloc ? 1 : 0;
  return 0;
}

int flagcxXshmemCommitArScratch(flagcxDevComm_t devComm, void *stream,
                                long ntiles) {
  XPUStream cs = (XPUStream)stream;
  XshmemKernelCtx *ctx = getCtx(devComm, cs);
  if (ctx == nullptr)
    return 1;
  ctx->arSeq += ntiles;
  if (ctx->arEvent == nullptr && xpu_event_create(&ctx->arEvent) != XPU_SUCCESS)
    return 1;
  return xpu_event_record(ctx->arEvent, cs) == XPU_SUCCESS ? 0 : 1;
}

int flagcxXshmemAcquireA2AScratch(flagcxDevComm_t devComm, void *stream,
                                  int npes, int clusternum, long tileElems,
                                  float **stage, uint64_t **signal,
                                  long *seqBase, int *needRendezvous) {
  XPUStream cs = (XPUStream)stream;
  XshmemKernelCtx *ctx = getCtx(devComm, cs);
  if (ctx == nullptr)
    return 1;

  bool needAlloc = (ctx->a2a.stage == nullptr) || (npes != ctx->a2aNpes) ||
                   (clusternum != ctx->a2aClusternum) ||
                   (tileElems > ctx->a2a.stageElems);
  if (needAlloc) {
    drainEvent(&ctx->a2aEvent);
    freeA2AScratch(&ctx->a2a);
    ctx->a2a.stage = (float *)xshmem_calloc((size_t)2 * tileElems * npes,
                                            sizeof(float), 1024);
    ctx->a2a.signal = (uint64_t *)xshmem_calloc(
        (size_t)2 * clusternum * npes, sizeof(uint64_t), sizeof(uint64_t));
    if (!ctx->a2a.stage || !ctx->a2a.signal) {
      freeA2AScratch(&ctx->a2a);
      return 1;
    }
    ctx->a2a.stageElems = tileElems;
    ctx->a2aNpes = npes;
    ctx->a2aClusternum = clusternum;
    ctx->a2aSeq = 0;
  } else if (ctx->a2aEvent != nullptr) {
    xpu_stream_wait_event(cs, ctx->a2aEvent);
  }

  *stage = ctx->a2a.stage;
  *signal = ctx->a2a.signal;
  *seqBase = ctx->a2aSeq;
  *needRendezvous = needAlloc ? 1 : 0;
  return 0;
}

int flagcxXshmemCommitA2AScratch(flagcxDevComm_t devComm, void *stream,
                                 long ntiles) {
  XPUStream cs = (XPUStream)stream;
  XshmemKernelCtx *ctx = getCtx(devComm, cs);
  if (ctx == nullptr)
    return 1;
  ctx->a2aSeq += ntiles;
  if (ctx->a2aEvent == nullptr &&
      xpu_event_create(&ctx->a2aEvent) != XPU_SUCCESS)
    return 1;
  return xpu_event_record(ctx->a2aEvent, cs) == XPU_SUCCESS ? 0 : 1;
}

} // extern "C"
