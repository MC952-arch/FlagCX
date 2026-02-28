/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * FlagCX Device API - Template wrappers and inline functions for
 * platform-agnostic device-side communication primitives.
 *
 * On NVIDIA (NCCL 2.28+): wraps NCCL device API types and functions.
 * On other platforms: provides fallback implementations.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_API_H_
#define FLAGCX_DEVICE_API_H_

#include "atomic_device.h"
#include "device_utils.h"

// ============================================================
// NVIDIA backend: include NCCL device headers
// ============================================================
#ifdef USE_NVIDIA_ADAPTOR
#include "nccl.h"
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"
#define FLAGCX_DEVICE_API_NCCL 1
#endif
#endif

// ============================================================
// Section 1: flagcxDeviceComm — Device Communicator
//
// Template wrapper around ncclDevComm on NVIDIA.
// Provides platform-agnostic accessors for rank/size info.
// Passed by VALUE to kernels (matching NCCL pattern).
// ============================================================
struct flagcxDeviceComm {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclDevComm _base;

  FLAGCX_HOST_DEVICE_INLINE flagcxDeviceComm() : _base() {}
  FLAGCX_HOST_DEVICE_INLINE flagcxDeviceComm(const ncclDevComm &base)
      : _base(base) {}

  // Intra-node (LSA) accessors
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
    return _base.lsaRank;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
    return _base.lsaSize;
  }

  // Global accessors
  FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return _base.rank; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return _base.nRanks; }
#else
  int _rank, _nRanks;
  int _intraRank, _intraSize;
  void *_barrierMem;
  void **_peerPtrs;

  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const { return _intraRank; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const { return _intraSize; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return _rank; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return _nRanks; }
#endif
};

// ============================================================
// Section 2: flagcxDeviceWindow — Device-Side Window
//
// Value type passed to kernels (like ncclWindow_t).
// Distinct from host-side flagcxWindow_t (= flagcxWindow*).
// On NVIDIA: wraps ncclWindow_t with implicit conversion.
// On fallback: stores peer pointer arrays.
// ============================================================
struct flagcxDeviceWindow {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclWindow_t _base;

  FLAGCX_HOST_DEVICE_INLINE flagcxDeviceWindow() : _base() {}
  FLAGCX_HOST_DEVICE_INLINE flagcxDeviceWindow(ncclWindow_t base)
      : _base(base) {}
  FLAGCX_HOST_DEVICE_INLINE operator ncclWindow_t() const { return _base; }
#else
  void *basePtr;
  size_t size;
  void **peerPtrs;
#endif
};

// ============================================================
// Section 3: flagcxTeam_t — Team Descriptor
//
// Represents a subset of ranks (intra-node, inter-node, etc.).
// On NVIDIA: wraps ncclTeam_t.
// ============================================================
struct flagcxTeam {
  int nRanks;
  int rank;
  int stride;

#ifdef FLAGCX_DEVICE_API_NCCL
  ncclTeam_t _base;

  FLAGCX_HOST_DEVICE_INLINE flagcxTeam()
      : nRanks(0), rank(0), stride(0), _base() {}
  FLAGCX_HOST_DEVICE_INLINE flagcxTeam(ncclTeam_t base)
      : nRanks(base.nRanks), rank(base.rank), stride(base.stride), _base(base) {
  }
  FLAGCX_HOST_DEVICE_INLINE operator ncclTeam_t() const { return _base; }
#else
  FLAGCX_HOST_DEVICE_INLINE flagcxTeam() : nRanks(0), rank(0), stride(0) {}
#endif
};
typedef struct flagcxTeam flagcxTeam_t;

// ============================================================
// Section 4: Team Accessor Functions (Inline Wrappers)
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
FLAGCX_DEVICE_INLINE_DECORATOR flagcxTeam_t
flagcxTeamIntra(const flagcxDeviceComm &devComm) {
  return flagcxTeam_t(ncclTeamLsa(devComm._base));
}
#else
FLAGCX_DEVICE_INLINE_DECORATOR flagcxTeam_t
flagcxTeamIntra(const flagcxDeviceComm &devComm) {
  flagcxTeam_t team;
  team.nRanks = devComm.getIntraSize();
  team.rank = devComm.getIntraRank();
  team.stride = 1;
  return team;
}
#endif

// ============================================================
// Section 5: flagcxCoopBlock — Block-Level Cooperative Group
//
// On NVIDIA: wraps ncclCoopCta.
// ============================================================
struct flagcxCoopBlock {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclCoopCta _impl;

  FLAGCX_HOST_DEVICE_INLINE flagcxCoopBlock() : _impl() {}

  FLAGCX_DEVICE_INLINE_DECORATOR int thread_rank() const {
    return _impl.thread_rank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }

  // Implicit conversion for passthrough to NCCL APIs
  FLAGCX_HOST_DEVICE_INLINE operator ncclCoopCta() const { return _impl; }
#else
  FLAGCX_DEVICE_INLINE_DECORATOR int thread_rank() const {
    return 0; // placeholder for fallback
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
    return 0; // placeholder for fallback
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { FLAGCX_DEVICE_SYNC_THREADS(); }
#endif
};

// ============================================================
// Section 6: flagcxIntraBarrierSession — Intra-Node Barrier
//
// On NVIDIA: wraps ncclLsaBarrierSession with simplified constructor.
// NCCL requires 6 params: {coop, devComm, team, barrierHandle, index, true}
// FlagCX requires 4 params: {coop, devComm, team, index}
// The barrier handle is extracted internally from devComm._base.
// ============================================================
template <typename Coop>
struct flagcxIntraBarrierSession {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclLsaBarrierSession<ncclCoopCta> _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession(Coop coop, const flagcxDeviceComm &devComm,
                            flagcxTeam_t team, uint32_t index)
      : _impl(ncclCoopCta(), devComm._base, ncclTeamLsa(devComm._base),
              devComm._base.lsaBarrier, index, false) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.arrive(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.wait(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.sync(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order]);
  }
#else
  // Fallback: placeholder barrier using atomics
  volatile uint32_t *_counter;
  int _nRanks;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession(Coop coop, const flagcxDeviceComm &devComm,
                            flagcxTeam_t team, uint32_t index)
      : _counter(nullptr), _nRanks(team.nRanks) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    // Fallback: use flagcxDeviceAtomic* from atomic_device.h
  }
#endif
};

// ============================================================
// Section 7: Pointer Access Functions (Inline Wrappers)
//
// These unwrap _base from template wrappers and dispatch to
// vendor-specific functions. On fallback, they index into
// peer pointer arrays stored in flagcxDeviceWindow.
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDeviceWindow &win, size_t offset,
                          const flagcxDeviceComm &devComm) {
  return ncclGetLsaMultimemPointer(win._base, offset, devComm._base);
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDeviceWindow &win, size_t offset, int peer) {
  return ncclGetLsaPointer(win._base, offset, peer);
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointer(const flagcxDeviceWindow &win, size_t offset) {
  return ncclGetLocalPointer(win._base, offset);
}
#else
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDeviceWindow &win, size_t offset,
                          const flagcxDeviceComm &devComm) {
  return (char *)win.basePtr + offset;
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDeviceWindow &win, size_t offset, int peer) {
  return (char *)win.peerPtrs[peer] + offset;
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointer(const flagcxDeviceWindow &win, size_t offset) {
  return (char *)win.basePtr + offset;
}
#endif

// ============================================================
// Section 8: Constants
// ============================================================
#ifndef FLAGCX_DEVICE_CTA_COUNT
#define FLAGCX_DEVICE_CTA_COUNT 36
#endif
#ifndef FLAGCX_DEVICE_THREADS_PER_CTA
#define FLAGCX_DEVICE_THREADS_PER_CTA 512
#endif

#endif // FLAGCX_DEVICE_API_H_
