/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * FlagCX Device API - Template wrappers and inline functions for
 * platform-agnostic device-side communication primitives.
 *
 * On NVIDIA (NCCL > 2.28): wraps NCCL device API types and functions.
 * On other platforms: provides fallback implementations using IPC.
 *
 * This header is safe to include from both .cu files (nvcc) and
 * .cc files (g++).  Device-only functions (Sections 5-8) are guarded
 * by FLAGCX_DEVICE_COMPILE so they are invisible to host compilers
 * on all platforms.
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
// Section 1: flagcxDevCommInternal — Host-Side Opaque Handle
//
// Backing struct for flagcxDevComm_t (declared in flagcx_kernel.h).
// Populated by flagcxDevCommCreate, freed by flagcxDevCommDestroy.
// Defined BEFORE flagcxDevComm so the unified constructor can
// access its members inline.
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
struct flagcxDevCommInternal {
  ncclDevComm ncclDev;   // Populated by pncclDevCommCreate
  uint32_t barrierEpoch; // Unused in NCCL path, present for unified host code
};
#else
struct flagcxDevCommInternal {
  int rank, nRanks;
  int intraRank, intraSize;
  // Barrier: device array of pointers to each peer's barrier flags
  // Layout: barrierPeers[localRank][ctaIndex] = uint32_t counter
  uint32_t *
      *barrierPeers; // device pointer to array of nLocalRanks device pointers
  uint32_t *localBarrierFlags; // this rank's barrier memory (CTA_COUNT entries)
  uint32_t barrierEpoch; // monotonically increasing, set by host before launch
  // Host-side cleanup bookkeeping (not passed to kernel)
  void **peerBarrierPtrs; // host array of IPC-mapped pointers (for close)
  int *localRankToRank;   // intra-node rank mapping (for IPC exchange)
  int nLocalRanks;
};
#endif

// ============================================================
// Section 1b: flagcxDevMemType — Memory Mode Enum
//
// Defined here (with include guard) because structs below need it.
// Also defined in flagcx_kernel.h for the public API signature.
// ============================================================
#ifndef FLAGCX_DEV_MEM_TYPE_DEFINED
#define FLAGCX_DEV_MEM_TYPE_DEFINED
typedef enum {
  flagcxDevMemIpc = 0,   // IPC peer pointer mode (all NCCL versions)
  flagcxDevMemWindow = 1 // NCCL window mode (NCCL > 2.28 only)
} flagcxDevMemType;
#endif

// ============================================================
// Section 2: flagcxDevMemInternal — Host-Side Memory Handle
//
// Backing struct for flagcxDevMem_t.
// Created by flagcxDevMemCreate, freed by flagcxDevMemDestroy.
// Both modes (IPC and window) always populate devPeerPtrs so
// the kernel can use a single unified pointer access path.
// Defined BEFORE flagcxDevMem so the unified constructor can
// access its members inline.
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
struct flagcxDevMemInternal {
  flagcxDevMemType mode; // flagcxDevMemIpc or flagcxDevMemWindow
  void **devPeerPtrs;    // device array: [localRank] -> peer buffer (always)
  int nPeers;
  int intraRank;        // this rank's local rank index (for IPC local pointer)
  void **hostPeerPtrs;  // host array: for ipcMemHandleClose cleanup
  ncclWindow_t ncclWin; // from win->base (window mode only)
  void *winHandle;      // flagcxWindow_t stored as void* (window mode only)
};
#else
struct flagcxDevMemInternal {
  flagcxDevMemType mode; // always flagcxDevMemIpc on Tier 2
  void **devPeerPtrs;    // device array: [localRank] -> peer buffer ptr
  int nPeers;
  int intraRank;       // this rank's local rank index (for IPC local pointer)
  void **hostPeerPtrs; // host array: for ipcMemHandleClose cleanup
  void *basePtr;       // this rank's buffer pointer (for IPC close check)
};
#endif
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

// ============================================================
// Section 3: flagcxDevComm — Device Communicator (kernel-facing)
//
// Value type passed to kernels by value.
// On NVIDIA (NCCL > 2.28): wraps ncclDevComm.
// On fallback: carries rank info + barrier peer pointers.
// Unified constructor from flagcxDevCommInternal enables
// tier-agnostic host code: flagcxDevComm dc(*devCommHandle);
// ============================================================
struct flagcxDevComm {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclDevComm _base;

  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm() : _base() {}
  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm(const ncclDevComm &base)
      : _base(base) {}
  // Unified constructor from host handle
  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm(const flagcxDevCommInternal &di)
      : _base(di.ncclDev) {}

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
  uint32_t **_barrierPeers;
  uint32_t _barrierEpoch;

  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm()
      : _rank(0), _nRanks(0), _intraRank(0), _intraSize(0),
        _barrierPeers(nullptr), _barrierEpoch(0) {}
  // Unified constructor from host handle
  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm(const flagcxDevCommInternal &di)
      : _rank(di.rank), _nRanks(di.nRanks), _intraRank(di.intraRank),
        _intraSize(di.intraSize), _barrierPeers(di.barrierPeers),
        _barrierEpoch(di.barrierEpoch) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const { return _intraRank; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const { return _intraSize; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return _rank; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return _nRanks; }
#endif
};

// ============================================================
// Section 4: flagcxDevMem — Device-Side Memory Handle
//
// Value type passed to kernels by value.
// Both modes (IPC and window) carry mode + peerPtrs.
// On NCCL > 2.28, window mode additionally carries ncclWindow_t.
// Runtime dispatch in flagcxGetPeerPointer checks mode (Decision 7.19).
// Unified constructor from flagcxDevMemInternal enables
// tier-agnostic host code: flagcxDevMem dm(*devMemHandle);
// ============================================================
struct flagcxDevMem {
  flagcxDevMemType mode; // flagcxDevMemIpc or flagcxDevMemWindow
  void **peerPtrs;       // IPC mode: device array [localRank] -> peer buffer
  int intraRank; // local rank index (for flagcxGetLocalPointer in IPC mode)

#ifdef FLAGCX_DEVICE_API_NCCL
  ncclWindow_t _base; // used when mode == flagcxDevMemWindow

  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem()
      : mode(flagcxDevMemIpc), peerPtrs(nullptr), intraRank(0), _base() {}
  // Unified constructor from host handle
  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem(const flagcxDevMemInternal &di)
      : mode(di.mode), peerPtrs(di.devPeerPtrs), intraRank(di.intraRank),
        _base(di.ncclWin) {}
#else
  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem()
      : mode(flagcxDevMemIpc), peerPtrs(nullptr), intraRank(0) {}
  // Unified constructor from host handle
  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem(const flagcxDevMemInternal &di)
      : mode(di.mode), peerPtrs(di.devPeerPtrs), intraRank(di.intraRank) {}
#endif
};

// ============================================================
// Section 4b: flagcxTeam_t — Team Descriptor
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
// Sections 5-8: Device-only functions
//
// These sections use device builtins (threadIdx, __syncthreads, atomics)
// and are only safe under a device compiler (nvcc, hipcc, etc.).
// FLAGCX_DEVICE_COMPILE is defined in device_utils.h.
// ============================================================
#ifdef FLAGCX_DEVICE_COMPILE

// ============================================================
// Section 5: Team Accessor Functions (Inline Wrappers)
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamIntra(const flagcxDevComm &devComm) {
  return flagcxTeam_t(ncclTeamLsa(devComm._base));
}
#else
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamIntra(const flagcxDevComm &devComm) {
  flagcxTeam_t team;
  team.nRanks = devComm.getIntraSize();
  team.rank = devComm.getIntraRank();
  team.stride = 1;
  return team;
}
#endif

// ============================================================
// Section 6: flagcxCoopBlock — Block-Level Cooperative Group
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
// Section 7: flagcxIntraBarrierSession — Intra-Node Barrier
//
// On NVIDIA (NCCL > 2.28): wraps ncclLsaBarrierSession.
// On fallback: flag-based barrier using IPC-mapped peer memory + atomics.
// ============================================================
template <typename Coop>
struct flagcxIntraBarrierSession {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclLsaBarrierSession<ncclCoopCta> _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession(Coop coop, const flagcxDevComm &devComm,
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
  // Fallback: flag-based barrier using IPC-mapped peer memory + atomics
  uint32_t **_peerBarriers;
  int _nRanks, _myRank;
  uint32_t _ctaIndex;
  uint32_t _phase;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession(Coop coop, const flagcxDevComm &devComm,
                            flagcxTeam_t team, uint32_t index)
      : _peerBarriers(devComm._barrierPeers), _nRanks(team.nRanks),
        _myRank(team.rank), _ctaIndex(index), _phase(devComm._barrierEpoch) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    sync(coop, order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    sync(coop, order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _phase++;
    FLAGCX_DEVICE_SYNC_THREADS();
    if (threadIdx.x == 0) {
      // Signal: write my counter with release ordering
      flagcxDeviceAtomicStore(&_peerBarriers[_myRank][_ctaIndex], _phase,
                              flagcxDeviceMemoryOrderRelease);
      // Wait: spin until all peers reach this phase
      for (int p = 0; p < _nRanks; p++) {
        if (p == _myRank)
          continue;
        int iter = 0;
        while (flagcxDeviceAtomicLoad(&_peerBarriers[p][_ctaIndex],
                                      flagcxDeviceMemoryOrderAcquire) <
               _phase) {
          spinBackoff(iter++);
        }
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();
  }
#endif
};

// ============================================================
// Section 8: Pointer Access Functions (Inline Wrappers)
//
// 3 functions total (see plan Decision 7.8 / 7.9 / 7.19):
//   flagcxGetPeerPointer(mem, off, team, peer)  — canonical unicast
//   flagcxGetLocalPointer(mem, off)              — convenience (own buffer)
//   flagcxGetMulticastPointer(mem, off, devComm) — intra-node multicast
//
// On Tier 1 (NCCL > 2.28): runtime dispatch via mem.mode (Decision 7.19).
//   Window mode -> ncclGetPeerPointer;  IPC mode -> peerPtrs[index].
// On Tier 2: always IPC peerPtrs.
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDevMem &mem, size_t offset, flagcxTeam_t team,
                     int peer) {
  if (mem.mode == flagcxDevMemWindow) {
    return ncclGetPeerPointer(mem._base, offset, team._base, peer);
  } else {
    int index = team.rank + (peer - team.rank) * team.stride;
    return (char *)mem.peerPtrs[index] + offset;
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointer(const flagcxDevMem &mem, size_t offset) {
  if (mem.mode == flagcxDevMemWindow) {
    return ncclGetLocalPointer(mem._base, offset);
  } else {
    return (char *)mem.peerPtrs[mem.intraRank] + offset;
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDevMem &mem, size_t offset,
                          const flagcxDevComm &devComm) {
  if (mem.mode == flagcxDevMemWindow) {
    return ncclGetLsaMultimemPointer(mem._base, offset, devComm._base);
  } else {
    // IPC mode: multicast not available, return nullptr
    return nullptr;
  }
}
#else
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDevMem &mem, size_t offset, flagcxTeam_t team,
                     int peer) {
  // Tier 2: always IPC — team maps peer rank to flat index
  int index = team.rank + (peer - team.rank) * team.stride;
  return (char *)mem.peerPtrs[index] + offset;
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointer(const flagcxDevMem &mem, size_t offset) {
  return (char *)mem.peerPtrs[mem.intraRank] + offset;
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDevMem &mem, size_t offset,
                          const flagcxDevComm &devComm) {
  // Tier 2: multicast not available, return nullptr
  return nullptr;
}
#endif

#endif // FLAGCX_DEVICE_COMPILE

// ============================================================
// Section 9: Constants
// ============================================================
#ifndef FLAGCX_DEVICE_CTA_COUNT
#define FLAGCX_DEVICE_CTA_COUNT 36
#endif
#ifndef FLAGCX_DEVICE_THREADS_PER_CTA
#define FLAGCX_DEVICE_THREADS_PER_CTA 512
#endif

#endif // FLAGCX_DEVICE_API_H_
