/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device API Traits - Unified interface for Vendor and Default device APIs.
 *
 * Provides compile-time dispatch via template specialization:
 * - DeviceTraits<ncclDeviceImpl>: NCCL LSA barriers + GIN one-sided
 * - DeviceTraits<defaultDeviceImpl>: IPC barriers + FIFO one-sided (full impl)
 *
 * Kernel code uses DeviceAPI::* exclusively, no #ifdef branches.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_TRAITS_H_
#define FLAGCX_DEVICE_TRAITS_H_

#include "device_utils.h"
#include <cstddef>
#include <cstdint>

// Forward declarations
struct flagcxDevComm;
struct flagcxDevMem;

template <typename Impl>
struct DeviceTraits;

// ============================================================
// NCCL Implementation
// ============================================================
#ifdef USE_NVIDIA_ADAPTOR
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0) &&                              \
    !defined(FLAGCX_FORCE_FALLBACK)

#include "nccl_device.h"

struct ncclDeviceImpl {
  using DevComm = ncclDevComm;
  using Window = ncclWindow_t;
  using Team = ncclTeam_t;
  using Multimem = ncclMultimemHandle_t;
  using Barrier = ncclLsaBarrier;
  using RemoteAction = ncclGinRemoteAction;
  using LocalAction = ncclGinLocalAction;
  using FenceLevel = ncclGinFenceLevel;
};

template <>
struct DeviceTraits<ncclDeviceImpl> {
  using DevComm = ncclDevComm;
  using Window = ncclWindow_t;
  using Team = ncclTeam_t;
  using Multimem = ncclMultimemHandle_t;
  using Barrier = ncclLsaBarrier;
  using RemoteAction = ncclGinRemoteAction;
  using LocalAction = ncclGinLocalAction;
  using FenceLevel = ncclGinFenceLevel;

  // DevComm accessors (map to NCCL field names)
  static FLAGCX_DEVICE_INLINE int getIntraRank(const DevComm *dc) {
    return dc->lsaRank;
  }
  static FLAGCX_DEVICE_INLINE int getIntraSize(const DevComm *dc) {
    return dc->lsaSize;
  }
  static FLAGCX_DEVICE_INLINE int getRank(const DevComm *dc) {
    return dc->rank;
  }
  static FLAGCX_DEVICE_INLINE int getSize(const DevComm *dc) {
    return dc->nRanks;
  }
  static FLAGCX_DEVICE_INLINE void *getFifoBuffer(const DevComm *dc) {
    return nullptr;
  }

  // Pointer access operations
  static FLAGCX_DEVICE_INLINE void *getPeerPointer(Window *win, size_t offset,
                                                   Team team, int peer) {
    return ncclGetPeerPointer(*win, offset, team, peer);
  }

  static FLAGCX_DEVICE_INLINE void *getLocalPointer(Window *win,
                                                    size_t offset) {
    return ncclGetLocalPointer(*win, offset);
  }

  static FLAGCX_DEVICE_INLINE void *getIntraPointer(Window *win, size_t offset,
                                                    int peer) {
    return ncclGetLsaPointer(*win, offset, peer);
  }

  static FLAGCX_DEVICE_INLINE void *
  getMulticastPointer(Window *win, size_t offset, Multimem mmHandle) {
    return ncclGetMultimemPointer(*win, offset, mmHandle);
  }

  // Window comparison (NCCL: compare entire window structure)
  static FLAGCX_DEVICE_INLINE bool windowEqual(const Window *a,
                                               const Window *b) {
    // For ncclWindow_t, compare the entire structure
    // (ncclWindow_t is a struct, not a pointer)
    return a->base == b->base && a->size == b->size;
  }

  // Barrier operations
  static FLAGCX_DEVICE_INLINE void barrierArrive(Barrier *bar, int count) {
    ncclLsaBarrierArrive(bar, count);
  }

  static FLAGCX_DEVICE_INLINE void barrierWait(Barrier *bar, uint64_t phase) {
    ncclLsaBarrierWait(bar, phase);
  }

  // GIN operations
  static FLAGCX_DEVICE_INLINE void ginPut(Window *win, void *src, size_t size,
                                          int peer, RemoteAction remoteAction,
                                          LocalAction localAction,
                                          FenceLevel fence) {
    ncclGinPut(win, src, size, peer, remoteAction, localAction, fence);
  }

  static FLAGCX_DEVICE_INLINE void ginSignal(Window *win, int signalId,
                                             uint64_t value, FenceLevel fence) {
    ncclGinSignal(win, signalId, value, fence);
  }

  static FLAGCX_DEVICE_INLINE void ginWaitSignal(Window *win, int signalId,
                                                 uint64_t value) {
    ncclGinWaitSignal(win, signalId, value);
  }

  static FLAGCX_DEVICE_INLINE void ginFlush(Window *win, FenceLevel fence) {
    ncclGinFlush(win, fence);
  }
};

#define FLAGCX_DEVICE_API_VENDOR 1
using DeviceAPI = DeviceTraits<ncclDeviceImpl>;

#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
#endif // USE_NVIDIA_ADAPTOR

// ============================================================
// Cambricon Implementation (future)
// ============================================================
// When Cambricon support is added, implement:
// #ifdef USE_CAMBRICON_ADAPTOR
// #ifdef CNCL_HAS_DEVICE_API
// template<> struct DeviceTraits<cnclDeviceImpl> { ... };
// #endif
// #endif

// ============================================================
// Default Implementation (IPC barriers + FIFO one-sided)
// ============================================================
#ifndef FLAGCX_DEVICE_API_VENDOR

struct defaultDeviceImpl {};

template <>
struct DeviceTraits<defaultDeviceImpl> {
  // ---- Window: IPC + MR + rawPtr ----
  struct Window {
    void *rawPtr;     // Raw memory pointer (always valid)
    void **peerPtrs;  // IPC peer pointers (nullptr if no IPC)
    int intraRank;    // Local rank index
    uintptr_t mrBase; // MR base VA
    int mrIndex;      // MR table index
  };

  // ---- DevComm: All fallback layers ----
  struct DevComm {
    // Baseline
    int rank, nRanks;
    int intraRank, intraSize;
    void *fifoBuffer;

    // IPC barriers
    uint32_t **barrierPeers;
    uint32_t intraBarrierEpoch;
    int nBarriers;

    // Inter-node signal relay
    uint64_t *interSignalFlags;
    int nInterPeers;
    bool isInterLeader;
    uint64_t interBarrierEpoch;

    // One-sided fallback
    uint64_t *signalBuffer;
    uint64_t *shadowBuffer;
    uint64_t *counterBuffer;
    int signalCount;
    int counterCount;
    int contextCount;
  };

  // ---- Team: Pure arithmetic ----
  struct Team {
    int nRanks, rank, stride;
  };

  // ---- Multimem: Placeholder ----
  struct Multimem {
    void *mcBasePtr;
  };

  // ---- DevComm accessors ----
  static FLAGCX_DEVICE_INLINE int getIntraRank(const DevComm *dc) {
    return dc->intraRank;
  }
  static FLAGCX_DEVICE_INLINE int getIntraSize(const DevComm *dc) {
    return dc->intraSize;
  }
  static FLAGCX_DEVICE_INLINE int getRank(const DevComm *dc) {
    return dc->rank;
  }
  static FLAGCX_DEVICE_INLINE int getSize(const DevComm *dc) {
    return dc->nRanks;
  }
  static FLAGCX_DEVICE_INLINE void *getFifoBuffer(const DevComm *dc) {
    return dc->fifoBuffer;
  }

  // ---- Pointer access operations ----
  static FLAGCX_DEVICE_INLINE void *getPeerPointer(Window *win, size_t offset,
                                                   Team team, int peer) {
    if (win->peerPtrs) {
      int index = team.rank + (peer - team.rank) * team.stride;
      return (char *)win->peerPtrs[index] + offset;
    }
    return nullptr;
  }

  static FLAGCX_DEVICE_INLINE void *getLocalPointer(Window *win,
                                                    size_t offset) {
    if (win->peerPtrs)
      return (char *)win->peerPtrs[win->intraRank] + offset;
    return (char *)win->rawPtr + offset;
  }

  static FLAGCX_DEVICE_INLINE void *getIntraPointer(Window *win, size_t offset,
                                                    int peer) {
    if (win->peerPtrs)
      return (char *)win->peerPtrs[peer] + offset;
    return nullptr;
  }

  static FLAGCX_DEVICE_INLINE void *
  getMulticastPointer(Window *win, size_t offset, Multimem mmHandle) {
    (void)win;
    (void)offset;
    (void)mmHandle;
    return nullptr; // Multicast not available in fallback
  }

  // Window comparison (default: compare rawPtr)
  static FLAGCX_DEVICE_INLINE bool windowEqual(const Window *a,
                                               const Window *b) {
    return a->rawPtr == b->rawPtr;
  }
};

using DeviceAPI = DeviceTraits<defaultDeviceImpl>;

#endif // !FLAGCX_DEVICE_API_VENDOR

#endif // FLAGCX_DEVICE_TRAITS_H_
