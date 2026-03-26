/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API - Template wrappers and inline functions for
 * platform-agnostic device-side communication primitives.
 *
 * On Vendor: wraps vendor device API types and functions.
 * On other platforms: provides fallback implementations using IPC.
 *
 * This header is safe to include from both .cu files (nvcc) and
 * .cc files (g++).  Device-only functions (Sections 5-8) are guarded
 * by FLAGCX_DEVICE_COMPILE so they are invisible to host compilers
 * on all platforms.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_API_H_
#define FLAGCX_DEVICE_API_H_

#include <cstddef> // ptrdiff_t, size_t

#include "device_utils.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

// Device traits — provides DeviceAPI with all type/function dispatch.
// Also defines FLAGCX_DEVICE_API_VENDOR when vendor backend is active.
// Action types (flagcxDevNet_*, flagcxDescriptorSmem, etc.) are defined
// inside device_traits.h before the backend-specific trait includes.
#include "device_traits.h"

// Forward declaration for typed vendor device comm handle
struct flagcxInnerDevComm;
typedef struct flagcxInnerDevComm *flagcxInnerDevComm_t;

// ============================================================
// Section 1: flagcxDevCommInternal — Host-Side Opaque Handle
//
#define FLAGCX_MAX_INTER_PEERS 256

// Backing struct for flagcxDevComm_t (declared in flagcx_kernel.h).
// Populated by flagcxDevCommCreate, freed by flagcxDevCommDestroy.
// Unified capability-based design: baseline always populated,
// IPC and Vendor layers added when available.
// ============================================================
struct flagcxDevCommInternal {
  // ---- Baseline (always set) ----
  int rank, nRanks;
  int intraRank, intraSize;
  void *fifoBuffer; // Device-accessible FIFO (from heteroComm, may be null)
  // ---- IPC barrier layer (set if IPC barrier setup succeeds, else nullptr)
  // ----
  uint32_t *
      *barrierPeers; // device pointer to array of nLocalRanks device pointers
  uint32_t
      *localBarrierFlags; // this rank's inbox buffer (nLocalRanks × CTA_COUNT)
  uint32_t
      intraBarrierEpoch; // monotonically increasing, set by host before launch
  int nBarriers;         // = FLAGCX_DEVICE_CTA_COUNT (needed in kernel)
  // Host-side cleanup bookkeeping (not passed to kernel)
  int barrierIpcIndex;  // index into comm->ipcTable (-1 if no IPC barrier)
  int *localRankToRank; // intra-node rank mapping (for IPC exchange)
  int nLocalRanks;

  // ---- Inter-node signal relay (set if nInterPeers > 0, else nullptr) ----
  uint64_t *interSignalFlags;     // device pointer (from hostGetDevicePointer)
  uint64_t *interSignalFlagsHost; // host pointer (for recv thread + dealloc)
  uint64_t
      interBarrierEpoch; // inter-node epoch (separate from intraBarrierEpoch)
  int nInterPeers;       // number of inter-node peers (set on ALL ranks)
  bool isInterLeader;    // true only on localRank 0 (manages connections)
  int *interPeerRanks;   // global ranks of inter-node peers
  // netAdaptor connections for signal relay (one-sided RDMA atomic)
  void **signalSendComms;  // [nInterPeers] sendComm (for iputSignal)
  void **barrierRecvComms; // [nInterPeers] recvComm (kept alive for QP)
  void *barrierHandleInfo; // flagcxOneSideHandleInfo* with rkeys/baseVas
  // netAdaptor pointer (cached for proxy)
  void *netAdaptorPtr;

  // ---- One-sided Fallback layer (set if interSignalCount/interCounterCount >
  // 0)
  // ----
  uint64_t *signalBuffer; // GPU memory (flagcxMemAlloc), [signalCount] entries
  uint64_t
      *shadowBuffer; // GPU memory (local only, no MR), [signalCount] entries
  uint64_t
      *counterBuffer; // GPU memory (flagcxMemAlloc), [counterCount] entries
  int signalCount;
  int counterCount;
  int contextCount; // = reqs.interContextCount (default 4)
  // Host-only: MR handles + staging for cleanup
  void *signalBufferMr;        // MR handle for signalBuffer
  void *counterBufferMr;       // MR handle for counterBuffer
  void *putValueStagingBuffer; // 8 bytes host-pinned, MR registered
  void *putValueStagingMr;     // MR handle for staging buffer

  // ---- Vendor device comm (set if adaptor->devCommCreate succeeds, else NULL)
  // ----
  flagcxInnerDevComm_t devComm; // Typed vendor handle (per-adaptor struct)
};

// ============================================================
// Section 2: flagcxDevMemInternal — Host-Side Memory Handle
//
// Backing struct for flagcxDevMem_t.
// Created by flagcxDevMemCreate, freed by flagcxDevMemDestroy.
// Unified capability-based design: rawPtr always populated,
// IPC and Window layers added when available.
// Capabilities detected by null-checks:
//   devPeerPtrs != nullptr → IPC available
//   window != nullptr       → Window available (Vendor or default)
// ============================================================
struct flagcxDevMemInternal {
  // ---- Baseline (always set) ----
  void *rawPtr;   // = buff parameter
  bool hasWindow; // true if any window layer is available (basic or symmetric)
  bool isSymmetric; // true only for FLAGCX_WIN_COLL_SYMMETRIC (enables
                    // one-sided)

  // ---- Per-window MR layer (set by flagcxDevMemCreate from handle table) ----
  int mrIndex; // index into globalOneSideHandleTable (-1 if not registered)
  uintptr_t mrBase; // handles[mrIndex]->baseVas[myRank] (cached for device)

  // ---- IPC layer (set if IPC exchange succeeds, else nullptr) ----
  void **devPeerPtrs; // cached from comm->ipcTable[ipcIndex].devPeerPtrs
  int ipcIndex;       // index into comm->ipcTable (-1 if no IPC)
  int intraRank;      // this rank's local rank index (for IPC local pointer)

  // ---- Window layer (opaque pointer to DeviceAPI::Window) ----
  void *window;    // Points to vendor Window or defaultDeviceImpl::Window
                   // (fallback)
  void *winHandle; // Host-side handle for cleanup
};
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

// ============================================================
// Section 3: flagcxDevComm — Device Communicator (kernel-facing)
//
// Value type passed to kernels by value.
// Pure wrapper around DeviceAPI::DevComm which contains all fields.
// On Vendor: DevComm = vendor DevComm
// On default: DevComm = {rank, nRanks, fifoBuffer, barrierPeers, ...}
// ============================================================
struct flagcxDevComm {
  typename DeviceAPI::DevComm _commBase;

  // Wrapper-level fields needed by FIFO encoding on all paths.
  // Populated from flagcxDevCommInternal; safe to be 0 when unused.
  int _signalCount;
  int _counterCount;
  int _contextCount;
  int _nInterPeers;

  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm()
      : _commBase(), _signalCount(0), _counterCount(0), _contextCount(0),
        _nInterPeers(0) {}

  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm(const flagcxDevCommInternal &di)
      : _signalCount(di.signalCount), _counterCount(di.counterCount),
        _contextCount(di.contextCount), _nInterPeers(di.nInterPeers) {
    if (di.devComm) {
      _commBase = *(typename DeviceAPI::DevComm *)di.devComm;
    }
#ifndef FLAGCX_DEVICE_API_VENDOR
    else {
      // Fallback: populate _commBase directly from handle fields.
      // This ensures mutable fields (epochs) are always up-to-date.
      _commBase.rank = di.rank;
      _commBase.nRanks = di.nRanks;
      _commBase.intraRank = di.intraRank;
      _commBase.intraSize = di.intraSize;
      _commBase.fifoBuffer = di.fifoBuffer;
      _commBase.barrierPeers = di.barrierPeers;
      _commBase.intraBarrierEpoch = di.intraBarrierEpoch;
      _commBase.nBarriers = di.nBarriers;
      _commBase.interSignalFlags = di.interSignalFlags;
      _commBase.nInterPeers = di.nInterPeers;
      _commBase.isInterLeader = di.isInterLeader;
      _commBase.interBarrierEpoch = di.interBarrierEpoch;
      _commBase.signalBuffer = di.signalBuffer;
      _commBase.shadowBuffer = di.shadowBuffer;
      _commBase.counterBuffer = di.counterBuffer;
      _commBase.signalCount = di.signalCount;
      _commBase.counterCount = di.counterCount;
      _commBase.contextCount = di.contextCount;
    }
#endif
  }

  // Accessors delegate to _commBase member functions
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
    return _commBase.getIntraRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
    return _commBase.getIntraSize();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const {
    return _commBase.getRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const {
    return _commBase.getSize();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer() const {
    return _commBase.getFifoBuffer();
  }
};

// ============================================================
// Section 4: flagcxDevMem — Device-Side Memory Handle
//
// Value type passed to kernels by value.
// Pure wrapper around DeviceAPI::Window which contains all fields.
// On Vendor: Window = vendor Window
// On default: Window = {rawPtr, peerPtrs, intraRank, mrBase, mrIndex}
// ============================================================
struct flagcxDevMem {
  typename DeviceAPI::Window _winBase;

  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem() : _winBase() {}

  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem(const flagcxDevMemInternal &di) {
    if (di.window)
      _winBase = *(typename DeviceAPI::Window *)di.window;
  }

  FLAGCX_HOST_DEVICE_INLINE bool hasWindow() const {
    return _winBase.hasAccess();
  }
  FLAGCX_HOST_DEVICE_INLINE void *getRawPtr() const {
    return _winBase.getRawPtr();
  }
  FLAGCX_HOST_DEVICE_INLINE void **getDevPeerPtrs() const {
    return _winBase.getDevPeerPtrs();
  }
  FLAGCX_HOST_DEVICE_INLINE int getMrIndex() const {
    return _winBase.getMrIndex();
  }
};

// ============================================================
// Section 4b: flagcxTeam_t — Team Descriptor
//
// Represents a subset of ranks (intra-node, inter-node, etc.).
// Pure wrapper around DeviceAPI::Team.
// ============================================================
struct flagcxTeam {
  typename DeviceAPI::Team _teamBase;

  FLAGCX_HOST_DEVICE_INLINE flagcxTeam() : _teamBase() {}
  FLAGCX_HOST_DEVICE_INLINE flagcxTeam(int nr, int r, int s) {
    _teamBase.nRanks = nr;
    _teamBase.rank = r;
    _teamBase.stride = s;
  }
};
typedef struct flagcxTeam flagcxTeam_t;

// ============================================================
// Section 4c: flagcxMulticastHandle — Multicast Memory Handle
//
// Pure wrapper around DeviceAPI::Multimem.
// On Vendor: Multimem = vendor MultimemHandle
// On default: Multimem = {mcBasePtr}
// ============================================================
struct flagcxMulticastHandle {
  typename DeviceAPI::Multimem _multimemBase;

  FLAGCX_HOST_DEVICE_INLINE flagcxMulticastHandle() : _multimemBase() {}
};
typedef struct flagcxMulticastHandle flagcxMulticastHandle_t;

// ============================================================
// Section 4d: Barrier Handle Types
//
// flagcxIntraBarrierHandle → vendor intra-barrier handle (Vendor)
// flagcxInterBarrierHandle → vendor inter-barrier handle (Vendor)
// Fallback: placeholder structs (no resource-handle model yet).
// ============================================================
struct flagcxIntraBarrierHandle {
  typename DeviceAPI::IntraBarrierHandle _base;
};
typedef struct flagcxIntraBarrierHandle flagcxIntraBarrierHandle_t;

struct flagcxInterBarrierHandle {
  typename DeviceAPI::InterBarrierHandle _base;
};
typedef struct flagcxInterBarrierHandle flagcxInterBarrierHandle_t;

// Team tag types for barrier session constructors
struct flagcxTeamTagWorld {};
struct flagcxTeamTagIntra {};
struct flagcxTeamTagInter {};

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
//
// On Vendor: forwards to vendor team functions via _commBase.
// On default: computes from baseline fields in _commBase.
// No #ifdef — DeviceAPI resolves at compile time.
// ============================================================
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamIntra(const flagcxDevComm &devComm) {
  flagcxTeam_t team;
  team._teamBase.nRanks = devComm.getIntraSize();
  team._teamBase.rank = devComm.getIntraRank();
  team._teamBase.stride = 1;
  return team;
}
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamWorld(const flagcxDevComm &devComm) {
  flagcxTeam_t team;
  team._teamBase.nRanks = devComm.getSize();
  team._teamBase.rank = devComm.getRank();
  team._teamBase.stride = 1;
  return team;
}
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamInter(const flagcxDevComm &devComm) {
  flagcxTeam_t team;
  team._teamBase.nRanks = devComm.getSize() / devComm.getIntraSize();
  team._teamBase.rank = devComm.getRank() / devComm.getIntraSize();
  team._teamBase.stride = devComm.getIntraSize();
  return team;
}

// ---- Team Algebra (pure arithmetic on {nRanks, rank, stride}) ----
// These 5 functions are identical on all tiers — no vendor delegation needed.

// Is team b's bPeer also a member of team a?
FLAGCX_HOST_DEVICE_INLINE bool
flagcxTeamRankIsMember(flagcxTeam_t a, flagcxTeam_t b, int bPeer) {
  int wrank = (bPeer - b._teamBase.rank) * b._teamBase.stride;
  int adelta = wrank / a._teamBase.stride;
  int amod = wrank % a._teamBase.stride;
  int arank = a._teamBase.rank + adelta;
  return 0 <= arank && arank < a._teamBase.nRanks && amod == 0;
}

// Convert team b's bPeer to team a's rank.
FLAGCX_HOST_DEVICE_INLINE int flagcxTeamRankToTeam(flagcxTeam_t a,
                                                   flagcxTeam_t b, int bPeer) {
  int wrank = (bPeer - b._teamBase.rank) * b._teamBase.stride;
  int adelta = wrank / a._teamBase.stride;
  int arank = a._teamBase.rank + adelta;
  return arank;
}

// Extract inner sub-team (first innerSize ranks per stride group).
FLAGCX_HOST_DEVICE_INLINE flagcxTeam_t
flagcxTeamInnerFactor(flagcxTeam_t parent, int innerSize) {
  flagcxTeam_t ans;
  ans._teamBase.nRanks = innerSize;
  ans._teamBase.rank = parent._teamBase.rank % innerSize;
  ans._teamBase.stride = parent._teamBase.stride;
  return ans;
}

// Extract outer sub-team (stride groups).
FLAGCX_HOST_DEVICE_INLINE flagcxTeam_t
flagcxTeamOuterFactor(flagcxTeam_t parent, int innerSize) {
  flagcxTeam_t ans;
  ans._teamBase.nRanks = parent._teamBase.nRanks / innerSize;
  ans._teamBase.rank = parent._teamBase.rank / innerSize;
  ans._teamBase.stride = parent._teamBase.stride * innerSize;
  return ans;
}

// Return the index'th element of parent minus subset (set difference).
FLAGCX_HOST_DEVICE_INLINE int flagcxTeamRankInDifference(flagcxTeam_t parent,
                                                         flagcxTeam_t subset,
                                                         int index) {
  int stride = subset._teamBase.stride / parent._teamBase.stride;
  int below = parent._teamBase.rank - subset._teamBase.rank * stride;
  if (stride < 0) {
    stride = -stride;
    below -= (subset._teamBase.nRanks - 1) * stride;
  }
  if (index < below) {
    return index;
  } else if (index - below < (subset._teamBase.nRanks - 1) * (stride - 1)) {
    return below + 1 + ((index - below) / (stride - 1)) * stride +
           (index - below) % (stride - 1);
  } else {
    return below + 1 + (subset._teamBase.nRanks - 1) * stride +
           (index - below - (subset._teamBase.nRanks - 1) * (stride - 1));
  }
}

// ---- DevComm-dependent team conversions ----

// Convert team rank to world rank.
FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToWorld(const flagcxDevComm &devComm, flagcxTeam_t team,
                      int rank) {
  return devComm.getRank() +
         (rank - team._teamBase.rank) * team._teamBase.stride;
}

// Convert team rank to intra-node rank.
FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToIntra(const flagcxDevComm &devComm, flagcxTeam_t team,
                      int rank) {
  return devComm.getIntraRank() +
         (rank - team._teamBase.rank) * team._teamBase.stride;
}

// ============================================================
// Section 6: Cooperative Group Types
//
// Platform-neutral cooperative groups for device-side synchronization.
// Naming: "Tile" = N PEs cooperating (avoids vendor-specific
//         Warp/Wave/Subgroup terms).
//
// All implementations live in DeviceTraits; these are thin wrappers.
// ============================================================

// ---- 6a. flagcxCoopBlock — CTA-level cooperative group ----
struct flagcxCoopBlock {
  typename DeviceAPI::CoopBlock _base;

  FLAGCX_HOST_DEVICE_INLINE flagcxCoopBlock() : _base() {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6b. flagcxCoopTile<N> — Tile of N threads within a warp ----
template <int N>
struct flagcxCoopTile {
  typename DeviceAPI::template CoopTile<N> _base;

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
#ifdef FLAGCX_SIMT_WIDTH
  FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
    return _base.laneMask();
  }
#endif
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6c. flagcxCoopThread — single-thread alias ----
typedef flagcxCoopTile<1> flagcxCoopThread;

// ---- 6d. flagcxCoopWarp — full-warp alias (SIMT only) ----
#ifdef FLAGCX_SIMT_WIDTH
typedef flagcxCoopTile<FLAGCX_SIMT_WIDTH> flagcxCoopWarp;
#endif

// ---- 6e. flagcxCoopTileSpan — consecutive tiles with named barrier ----
#ifdef FLAGCX_SIMT_WIDTH
struct flagcxCoopTileSpan {
  typename DeviceAPI::CoopTileSpan _base;

  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopTileSpan(int t0, int nTiles, int id)
      : _base(t0, nTiles, id) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};
#endif // FLAGCX_SIMT_WIDTH

// ---- 6f. flagcxCoopLanes — arbitrary lane bitmask ----
#ifdef FLAGCX_SIMT_WIDTH
struct flagcxCoopLanes {
  typename DeviceAPI::CoopLanes _base;

  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopLanes(uint32_t lmask = 0xffffffffu)
      : _base(lmask) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
  FLAGCX_DEVICE_INLINE_DECORATOR uint32_t getLmask() const {
    return _base.getLmask();
  }
};
#endif // FLAGCX_SIMT_WIDTH

// ---- 6g. flagcxCoopAny — type-erased cooperative group ----
struct flagcxCoopAny {
  typename DeviceAPI::CoopAny _base;

  flagcxCoopAny() = default;
  flagcxCoopAny(flagcxCoopAny const &) = default;

  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopBlock b)
      : _base(b._base) {}
  template <int N>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopTile<N> t)
      : _base(t._base) {}
#ifdef FLAGCX_SIMT_WIDTH
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopTileSpan s)
      : _base(s._base) {}
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopLanes l)
      : _base(l._base) {}
#endif

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6h. Free functions ----

// flagcxCoopGetLaneMask: get the active lane bitmask for a cooperative group
#ifdef FLAGCX_SIMT_WIDTH
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
flagcxCoopGetLaneMask(flagcxCoopTile<N> coop) {
  return coop.laneMask();
}
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t flagcxCoopGetLaneMask(flagcxCoopBlock) {
  return 0xffffffffu;
}
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
flagcxCoopGetLaneMask(flagcxCoopLanes coop) {
  return coop.getLmask();
}
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
flagcxCoopGetLaneMask(flagcxCoopTileSpan) {
  return 0xffffffffu;
}
#endif // FLAGCX_SIMT_WIDTH

// flagcxCoopIsThread: compile-time check if group is a single thread
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopIsThread(flagcxCoopTile<N>) {
  return N == 1;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopIsThread(flagcxCoopBlock) {
  return false;
}
#ifdef FLAGCX_SIMT_WIDTH
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopIsThread(flagcxCoopLanes) {
  return false;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopIsThread(flagcxCoopTileSpan) {
  return false;
}
#endif // FLAGCX_SIMT_WIDTH

// flagcxCoopWithinTile: compile-time check if group fits within a single tile
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopWithinTile(flagcxCoopTile<N>) {
  return true;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopWithinTile(flagcxCoopBlock) {
  return false;
}
#ifdef FLAGCX_SIMT_WIDTH
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopWithinTile(flagcxCoopLanes) {
  return true;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopWithinTile(flagcxCoopTileSpan) {
  return false;
}
#endif // FLAGCX_SIMT_WIDTH

// flagcxCoopCoalesced: get a cooperative group of active/safe threads
#ifdef FLAGCX_SIMT_WIDTH
FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopLanes flagcxCoopCoalesced() {
  return flagcxCoopLanes{DeviceAPI::Intrin::activemask()};
}
template <typename Coop>
FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopWarp flagcxCoopCoalesced(Coop) {
  return flagcxCoopWarp();
}
FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopLanes
flagcxCoopCoalesced(flagcxCoopLanes coop) {
  return coop;
}
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopTile<N>
flagcxCoopCoalesced(flagcxCoopTile<N> coop) {
  return coop;
}
#endif // FLAGCX_SIMT_WIDTH

// ============================================================
// Section 7: flagcxIntraBarrierSession — Intra-Node Barrier
//
// On NVIDIA (Vendor > 2.28): wraps vendor barrier session.
// On fallback: flag-based barrier using IPC-mapped peer memory + atomics.
// ============================================================
template <typename Coop>
struct flagcxIntraBarrierSession {
#ifdef FLAGCX_DEVICE_API_VENDOR
  ncclLsaBarrierSession<ncclCoopCta> _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession(Coop coop, const flagcxDevComm &devComm,
                            flagcxTeam_t team, uint32_t index,
                            bool multimem = false,
                            flagcxMulticastHandle mcHandle = {})
      : _impl(ncclCoopCta(), devComm._commBase, ncclTeamLsa(devComm._commBase),
              devComm._commBase._impl.lsaBarrier, index, multimem,
              mcHandle._multimemBase) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.arrive(ncclCoopCta(), DeviceAPI::Atomic::toNativeOrder(order));
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.wait(ncclCoopCta(), DeviceAPI::Atomic::toNativeOrder(order));
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.sync(ncclCoopCta(), DeviceAPI::Atomic::toNativeOrder(order));
  }
#else
  // Fallback: thread-striped per-peer inbox barrier (aligned with
  // vendor barrier session). Each rank has an inbox buffer: inbox[senderRank *
  // nBarriers + ctaIndex]. arrive: thread-striped store(epoch+1) to each peer's
  // inbox slot for me. wait:   thread-striped spin on own inbox slots from each
  // peer.
  uint32_t **_peerBuffers; // IPC-mapped pointers to each peer's inbox buffer
  int _nRanks, _myRank;
  int _nBarriers; // = CTA_COUNT
  uint32_t _ctaIndex;
  uint32_t _epoch;

  // Default constructor (no-op, for inter-only barrier composition)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession()
      : _peerBuffers(nullptr), _nRanks(0), _myRank(0), _nBarriers(0),
        _ctaIndex(0), _epoch(0) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession(Coop coop, const flagcxDevComm &devComm,
                            flagcxTeam_t team, uint32_t index,
                            bool multimem = false,
                            flagcxMulticastHandle mcHandle = {})
      : _peerBuffers(devComm._commBase.barrierPeers),
        _nRanks(team._teamBase.nRanks), _myRank(team._teamBase.rank),
        _nBarriers(devComm._commBase.nBarriers), _ctaIndex(index),
        _epoch(devComm._commBase.intraBarrierEpoch) {}

  // arrive: thread-striped store epoch+1 to each peer's inbox slot for me
  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    coop.sync();
    for (int i = FLAGCX_THREAD_IDX_X; i < _nRanks - 1;
         i += FLAGCX_BLOCK_DIM_X) {
      int peer = 1 + _myRank + i;
      if (peer >= _nRanks)
        peer -= _nRanks;
      // Write to peer's buffer at inbox[myRank * nBarriers + ctaIndex]
      DeviceAPI::Atomic::store(
          &_peerBuffers[peer][_myRank * _nBarriers + _ctaIndex], _epoch + 1,
          flagcxDeviceMemoryOrderRelease);
    }
  }

  // wait: thread-striped spin on own inbox slots from each peer
  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    for (int i = FLAGCX_THREAD_IDX_X; i < _nRanks - 1;
         i += FLAGCX_BLOCK_DIM_X) {
      int peer = 1 + _myRank + i;
      if (peer >= _nRanks)
        peer -= _nRanks;
      // Read from my buffer at inbox[peer * nBarriers + ctaIndex]
      int iter = 0;
      while (DeviceAPI::Atomic::load(
                 &_peerBuffers[_myRank][peer * _nBarriers + _ctaIndex],
                 flagcxDeviceMemoryOrderAcquire) < _epoch + 1) {
        DeviceAPI::Intrin::spinBackoff(iter++);
      }
    }
    _epoch += 1;
    coop.sync();
  }

  // sync = arrive + wait (same as vendor)
  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(coop, order);
    wait(coop, order);
  }
#endif
};

// ============================================================
// Section 8: Pointer Access Functions (Inline Wrappers)
//
// All functions delegate to _winBase member functions — no #ifdef branches.
// On Vendor: forwards to vendor pointer functions via _winBase.
// On default: uses IPC peerPtrs / rawPtr fallback.
// ============================================================
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDevMem &mem, size_t offset, flagcxTeam_t team,
                     int peer) {
  return mem._winBase.getPeerPointer(offset, team._teamBase, peer);
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointer(const flagcxDevMem &mem, size_t offset) {
  return mem._winBase.getLocalPointer(offset);
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDevMem &mem, size_t offset,
                          const flagcxDevComm &devComm) {
  (void)devComm;
  flagcxMulticastHandle_t mmHandle;
  return mem._winBase.getMulticastPointer(offset, mmHandle._multimemBase);
}

// ---- Additional pointer functions ----

// Peer pointer without team parameter.
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDevMem &mem, size_t offset, int peer) {
  // Without team, treat as intra-node access
  return mem._winBase.getIntraPointer(offset, peer);
}

// Intra-node rank pointer.
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetIntraPointer(const flagcxDevMem &mem, size_t offset, int peer) {
  return mem._winBase.getIntraPointer(offset, peer);
}

// Multicast pointer with explicit MulticastHandle.
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDevMem &mem, size_t offset,
                          flagcxMulticastHandle_t mmHandle) {
  return mem._winBase.getMulticastPointer(offset, mmHandle._multimemBase);
}

// Reverse lookup: raw pointer → flagcxDevMem.
// Vendor: cooperative search through vendor window table.
// Fallback: not supported (returns empty flagcxDevMem).
template <typename Coop>
FLAGCX_DEVICE_INLINE_DECORATOR flagcxDevMem
flagcxFindMem(Coop coop, const flagcxDevComm &devComm, void const *ptr) {
  flagcxDevMem result;
  (void)coop;
  (void)devComm;
  (void)ptr;
  return result;
}

// ============================================================
// Section 8b: flagcxSymPtr<T> — Typed Symmetric Pointer
//
// Value type storing {flagcxDevMem, offset}. Provides typed
// pointer methods and type-aware arithmetic.
// Mirrors vendor's SymPtr<T>.
// ============================================================
template <typename T>
struct flagcxSymPtr {
  flagcxDevMem mem;
  size_t offset;

  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr() : mem(), offset(0) {}
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr(flagcxDevMem m, size_t off)
      : mem(m), offset(off) {}

  // Type conversion (e.g. flagcxSymPtr<float> → flagcxSymPtr<char>)
  template <typename U>
  FLAGCX_HOST_DEVICE_INLINE operator flagcxSymPtr<U>() const {
    return {mem, offset};
  }

  // Typed pointer methods (delegate to free functions)
  FLAGCX_DEVICE_INLINE_DECORATOR T *localPtr() const {
    return (T *)flagcxGetLocalPointer(mem, offset);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *peerPtr(flagcxTeam_t team, int peer) const {
    return (T *)flagcxGetPeerPointer(mem, offset, team, peer);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *peerPtr(int peer) const {
    return (T *)flagcxGetPeerPointer(mem, offset, peer);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *intraPtr(int peer) const {
    return (T *)flagcxGetIntraPointer(mem, offset, peer);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *
  multicastPtr(const flagcxDevComm &devComm) const {
    return (T *)flagcxGetMulticastPointer(mem, offset, devComm);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *
  multicastPtr(flagcxMulticastHandle_t mmHandle) const {
    return (T *)flagcxGetMulticastPointer(mem, offset, mmHandle);
  }

  // Type-aware pointer arithmetic (integer math, no UB)
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(int d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(unsigned int d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(long d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(unsigned long d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(long long d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(unsigned long long d) {
    offset += d * sizeof(T);
    return *this;
  }

  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(int d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(unsigned int d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(long d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(unsigned long d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(long long d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(unsigned long long d) {
    offset -= d * sizeof(T);
    return *this;
  }
};

// Free operators for flagcxSymPtr<T>
template <typename T, typename Int>
FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> operator+(flagcxSymPtr<T> p, Int d) {
  return p += d;
}
template <typename T, typename Int>
FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> operator-(flagcxSymPtr<T> p, Int d) {
  return p -= d;
}
template <typename T>
FLAGCX_HOST_DEVICE_INLINE ptrdiff_t operator-(flagcxSymPtr<T> a,
                                              flagcxSymPtr<T> b) {
  return ((ptrdiff_t)a.offset - (ptrdiff_t)b.offset) / (ptrdiff_t)sizeof(T);
}
template <typename T>
FLAGCX_HOST_DEVICE_INLINE bool operator==(flagcxSymPtr<T> a,
                                          flagcxSymPtr<T> b) {
  return a.mem._winBase == b.mem._winBase && a.offset == b.offset;
}
template <typename T>
FLAGCX_HOST_DEVICE_INLINE bool operator!=(flagcxSymPtr<T> a,
                                          flagcxSymPtr<T> b) {
  return !(a == b);
}

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

// ============================================================
// Sections 9b-12: flagcxDevNet + Barriers (device-only)
// ============================================================
#ifdef FLAGCX_DEVICE_COMPILE

// ============================================================
// Section 10: flagcxDevNet — Device Network (thin wrapper)
//
// Delegates all operations to DeviceAPI::Net which contains
// backend-specific logic (vendor ncclGin or fallback FIFO).
// No #ifdef FLAGCX_DEVICE_API_VENDOR in this struct.
// ============================================================
struct flagcxDevNet {
  const flagcxDevComm &_devComm; // for barrier sessions
  typename DeviceAPI::Net _netBase;
  int _contextId;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevNet(const flagcxDevComm &dc, int contextIndex = 0)
      : _devComm(dc), _netBase(dc._commBase, contextIndex) {
    int cnt = (dc._contextCount > 0) ? dc._contextCount : 1;
    _contextId = contextIndex % cnt;
  }

  // ---- Two-sided operations ----
  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
  send(Coop coop, const flagcxDevMem &mem, size_t offset, size_t count,
       flagcxDataType_t datatype, int peer) const {
    return _netBase.send(coop._base, mem._winBase, offset, count, datatype,
                         peer);
  }
  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
  recv(Coop coop, const flagcxDevMem &mem, size_t offset, size_t count,
       flagcxDataType_t datatype, int peer) const {
    return _netBase.recv(coop._base, mem._winBase, offset, count, datatype,
                         peer);
  }
  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t term(Coop coop) const {
    return _netBase.term(coop._base);
  }
  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t wait(Coop coop) const {
    return _netBase.wait(coop._base);
  }

  // ---- One-sided: put (raw ptr) ----
  template <typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t team, int peer, const flagcxDevMem &dstMem, size_t dstOffset,
      const flagcxDevMem &srcMem, size_t srcOffset, size_t bytes,
      RemoteAction remoteAction = flagcxDevNet_None{},
      LocalAction localAction = flagcxDevNet_None{},
      Coop coop = flagcxCoopBlock{},
      DescriptorSmem descriptor = flagcxDevNet_None{},
      flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
      flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    _netBase.put(team._teamBase, peer, dstMem._winBase, dstOffset,
                 srcMem._winBase, srcOffset, bytes, remoteAction, localAction,
                 coop._base, descriptor, alreadyReleased, expected_scope);
  }

  // ---- One-sided: put (SymPtr) ----
  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t team, int peer, flagcxSymPtr<T> dst, flagcxSymPtr<T> src,
      size_t nElts, RemoteAction remoteAction = flagcxDevNet_None{},
      LocalAction localAction = flagcxDevNet_None{},
      Coop coop = flagcxCoopBlock{},
      DescriptorSmem descriptor = flagcxDevNet_None{},
      flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
      flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    this->put(team, peer, dst.mem, dst.offset, src.mem, src.offset,
              nElts * sizeof(T), remoteAction, localAction, coop, descriptor,
              alreadyReleased, expected_scope);
  }

  // ---- One-sided: putValue (raw ptr) ----
  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  putValue(flagcxTeam_t team, int peer, const flagcxDevMem &dstMem,
           size_t dstOffset, T value,
           RemoteAction remoteAction = flagcxDevNet_None{},
           Coop coop = flagcxCoopBlock{},
           DescriptorSmem descriptor = flagcxDevNet_None{},
           flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
           flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    _netBase.putValue(team._teamBase, peer, dstMem._winBase, dstOffset, value,
                      remoteAction, coop._base, descriptor, alreadyReleased,
                      expected_scope);
  }

  // ---- One-sided: putValue (SymPtr) ----
  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  putValue(flagcxTeam_t team, int peer, flagcxSymPtr<T> dst, T value,
           RemoteAction remoteAction = flagcxDevNet_None{},
           Coop coop = flagcxCoopBlock{},
           DescriptorSmem descriptor = flagcxDevNet_None{},
           flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
           flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    this->putValue(team, peer, dst.mem, dst.offset, value, remoteAction, coop,
                   descriptor, alreadyReleased, expected_scope);
  }

  // ---- One-sided: signal ----
  template <typename RemoteAction, typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  signal(flagcxTeam_t team, int peer, RemoteAction remoteAction,
         Coop coop = flagcxCoopBlock{},
         DescriptorSmem descriptor = flagcxDevNet_None{},
         flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
         flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    _netBase.signal(team._teamBase, peer, remoteAction, coop._base, descriptor,
                    alreadyReleased, expected_scope);
  }

  // ---- One-sided: flush ----
  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void flush(
      Coop coop,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _netBase.flush(coop._base, order);
  }

  // ---- One-sided: get (Fallback only, vendor stub exists for compilation)
  // ----
  template <typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  get(flagcxTeam_t team, int peer, const flagcxDevMem &srcMem, size_t srcOffset,
      const flagcxDevMem &dstMem, size_t dstOffset, size_t bytes,
      Coop coop = flagcxCoopBlock{}) const {
    _netBase.get(team._teamBase, peer, srcMem._winBase, srcOffset,
                 dstMem._winBase, dstOffset, bytes, coop._base);
  }

  // ---- Signal operations ----
  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignal(
      Coop coop, flagcxDevNetSignal_t signal, uint64_t least, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _netBase.waitSignal(coop._base, signal, least, bits, order);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignalMeetShadow(
      Coop coop, flagcxDevNetSignal_t signal, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _netBase.waitSignalMeetShadow(coop._base, signal, bits, order);
  }

  template <typename Coop, typename Uint>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignalFollowShadow(
      Coop coop, flagcxDevNetSignal_t signal, Uint leastDelta, Uint *before,
      Uint *delta, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _netBase.waitSignalFollowShadow(coop._base, signal, leastDelta, before,
                                    delta, bits, order);
  }

  // ---- Shadow manipulation ----
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
  getSignalShadowPtr(flagcxDevNetSignal_t signal) const {
    return _netBase.getSignalShadowPtr(signal);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  increaseSignalShadow(flagcxDevNetSignal_t signal, uint64_t delta) const {
    _netBase.increaseSignalShadow(signal, delta);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readSignal(
      flagcxDevNetSignal_t signal, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    return _netBase.readSignal(signal, bits, order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  resetSignal(flagcxDevNetSignal_t signal) const {
    _netBase.resetSignal(signal);
  }

  // ---- Counter operations ----
  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitCounter(
      Coop coop, flagcxDevNetCounter_t counter, uint64_t least, int bits = 56,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _netBase.waitCounter(coop._base, counter, least, bits, order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readCounter(
      flagcxDevNetCounter_t counter, int bits = 56,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    return _netBase.readCounter(counter, bits, order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  resetCounter(flagcxDevNetCounter_t counter) const {
    _netBase.resetCounter(counter);
  }
};

// ============================================================
// Section 11: flagcxInterBarrierSession — Inter-Node Barrier (Vendor only)
// ============================================================
#ifdef FLAGCX_DEVICE_API_VENDOR
// NOTE: On the vendor path, _nInterPeers is always 0 (set only by fallback's
// setupInterNodeSignalRelay). sync() is therefore a no-op. Multi-node kernels
// should use flagcxBarrierSession(flagcxTeamTagWorld) which wraps
// ncclBarrierSession for combined intra+inter sync.
template <typename Coop>
struct flagcxInterBarrierSession {
  alignas(ncclGinBarrierSession<ncclCoopCta>) char _implStorage[sizeof(
      ncclGinBarrierSession<ncclCoopCta>)];
  static_assert(sizeof(_implStorage) >=
                    sizeof(ncclGinBarrierSession<ncclCoopCta>),
                "implStorage too small for ncclGinBarrierSession");
  int _nInterPeers;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession(Coop coop, const flagcxDevNet &net,
                            flagcxTeam_t team, uint32_t index)
      : _nInterPeers(net._devComm._nInterPeers) {
    new (_implStorage) ncclGinBarrierSession<ncclCoopCta>(
        ncclCoopCta(), net._netBase._gin, team._teamBase,
        net._netBase._gin.comm.railGinBarrier, index);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      reinterpret_cast<ncclGinBarrierSession<ncclCoopCta> *>(_implStorage)
          ->sync(ncclCoopCta(), DeviceAPI::Atomic::toNativeOrder(order),
                 flagcxGinFenceLevelMap[static_cast<int>(fence)]);
    }
  }
};
#else
// Fallback: Inter-node barrier via FIFO Signal + netAdaptor isend/irecv.
// Sends signals to inter-node peers, waits on host-mapped interSignalFlags.
// Only the inter leader (localRank 0) actually sends/waits; non-leaders are
// no-ops. All ranks know nInterPeers for two-phase logic in BarrierSession.
template <typename Coop>
struct flagcxInterBarrierSession {
  uint64_t *_interSignals; // host-mapped inter signal array [CTA_COUNT]
  void *_fifoBuffer;       // for FIFO Signal entries
  int _nInterPeers;
  bool _isLeader;
  uint32_t _ctaIndex;
  uint64_t _epoch;

  // Active constructor (world barrier with inter-node peers)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession(Coop coop, const flagcxDevComm &devComm,
                            uint32_t index)
      : _interSignals(devComm._commBase.interSignalFlags),
        _fifoBuffer(devComm.getFifoBuffer()),
        _nInterPeers(devComm._nInterPeers),
        _isLeader(devComm._commBase.isInterLeader), _ctaIndex(index),
        _epoch(devComm._commBase.interBarrierEpoch) {}

  // Overload matching Vendor signature (coop, net, team, index)
  // Used by kernels that want inter-only barrier directly.
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession(Coop coop, const flagcxDevNet &net,
                            flagcxTeam_t team, uint32_t index)
      : _interSignals(net._devComm._commBase.interSignalFlags),
        _fifoBuffer(net._devComm.getFifoBuffer()),
        _nInterPeers(net._devComm._nInterPeers),
        _isLeader(net._devComm._commBase.isInterLeader), _ctaIndex(index),
        _epoch(net._devComm._commBase.interBarrierEpoch) {}

  // Default constructor (intra-only, all operations are no-ops)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession()
      : _interSignals(nullptr), _fifoBuffer(nullptr), _nInterPeers(0),
        _isLeader(false), _ctaIndex(0), _epoch(0) {}

  // Arrive: write one FIFO Signal entry (proxy fans out to all inter peers)
  // Only the leader sends; non-leaders skip.  Coop-scope: all threads
  // participate in sync, only threadRank==0 touches FIFO.
  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _epoch += _nInterPeers;
    coop.sync();
    if (coop.threadRank() == 0 && _isLeader) {
      DeviceAPI::fifoEnqueue(
          _fifoBuffer, (uint64_t)_ctaIndex, 0,
          DeviceAPI::buildTrd(flagcxDevicePrimBarrierSignal, 0, 0));
    }
    coop.sync();
  }

  // Wait: spin on host-mapped inter signal array
  // Only the leader waits; non-leaders skip.  Coop-scope.
  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    coop.sync();
    if (coop.threadRank() == 0 && _isLeader) {
      int iter = 0;
      while (DeviceAPI::Atomic::load(&_interSignals[_ctaIndex],
                                     flagcxDeviceMemoryOrderAcquire) < _epoch) {
        DeviceAPI::Intrin::spinBackoff(iter++);
      }
    }
    coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(coop, order);
    wait(coop, order);
  }
};
#endif

// ============================================================
// Section 12: flagcxBarrierSession — Unified Barrier (both tiers)
// ============================================================
#ifdef FLAGCX_DEVICE_API_VENDOR
template <typename Coop>
struct flagcxBarrierSession {
  // Placement-new storage: large enough for vendor barrier session (World) or
  // vendor intra-barrier session (Intra-only). The world barrier wraps
  // intra+inter so it is always >= the intra-barrier session in size and
  // alignment.
  alignas(ncclBarrierSession<ncclCoopCta>) char _implStorage[sizeof(
      ncclBarrierSession<ncclCoopCta>)];
  bool _intraOnly;

  // World barrier (intra + inter) — construct vendor barrier session
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagWorld, const flagcxDevNet &net,
                       uint32_t index, bool multimem = false)
      : _intraOnly(false) {
    new (_implStorage) ncclBarrierSession<ncclCoopCta>(
        ncclCoopCta(), ncclTeamTagWorld(), net._netBase._gin, index, multimem);
  }

  // Intra-only barrier — construct vendor intra-barrier session directly.
  // Bypasses the world barrier constructor which triggers a deleted
  // copy constructor in vendor utility internals.
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagIntra,
                       const flagcxDevComm &devComm, uint32_t index,
                       bool multimem = false)
      : _intraOnly(true) {
    new (_implStorage) ncclLsaBarrierSession<ncclCoopCta>(
        ncclCoopCta(), devComm._commBase, ncclTeamLsa(devComm._commBase),
        devComm._commBase._impl.lsaBarrier, index, multimem);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_intraOnly) {
      reinterpret_cast<ncclLsaBarrierSession<ncclCoopCta> *>(_implStorage)
          ->sync(ncclCoopCta(), DeviceAPI::Atomic::toNativeOrder(order));
    } else {
      reinterpret_cast<ncclBarrierSession<ncclCoopCta> *>(_implStorage)
          ->sync(ncclCoopCta(), DeviceAPI::Atomic::toNativeOrder(order),
                 flagcxGinFenceLevelMap[static_cast<int>(fence)]);
    }
  }
};
#else
// Fallback: Composes intra (IPC atomicAdd) + inter (FIFO Signal relay).
//         Three-phase pattern for multi-node:
//           Phase 1: intra sync (all local ranks ensure data visible)
//           Phase 2: leader inter signal+wait (non-leaders skip)
//           Phase 3: intra sync (broadcasts inter completion)
//         Single-node: just one intra sync (no phase 2/3).
template <typename Coop>
struct flagcxBarrierSession {
  flagcxIntraBarrierSession<Coop> _intra;
  flagcxInterBarrierSession<Coop> _inter;
  int _nInterPeers;

  // World barrier: intra (IPC) + inter (FIFO Signal → isend)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagWorld, const flagcxDevNet &net,
                       uint32_t index, bool multimem = false)
      : _intra(coop, net._devComm, flagcxTeamIntra(net._devComm), index),
        _inter(coop, net._devComm, index),
        _nInterPeers(net._devComm._nInterPeers) {}

  // Intra-only barrier: inter is default constructed (no-op)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagIntra,
                       const flagcxDevComm &devComm, uint32_t index,
                       bool multimem = false)
      : _intra(coop, devComm, flagcxTeamIntra(devComm), index), _inter(),
        _nInterPeers(0) {}

  // Accessors for sub-barrier sessions (Fallback only)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession<Coop> &intraBarrier() { return _intra; }

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession<Coop> &interBarrier() { return _inter; }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      // Phase 1: intra sync (arrive has coop.sync at start, wait has coop.sync
      // at end)
      _intra.arrive(coop, flagcxDeviceMemoryOrderRelease);
      _intra.wait(coop, flagcxDeviceMemoryOrderRelease);
      // Phase 2: inter signal+wait (leader only, non-leaders skip)
      _inter.arrive(coop, order);
      _inter.wait(coop, order);
      // Phase 3: intra sync (broadcast inter completion)
      _intra.arrive(coop, flagcxDeviceMemoryOrderAcquire);
      _intra.wait(coop, flagcxDeviceMemoryOrderAcquire);
    } else {
      // Single-node: one intra sync
      _intra.arrive(coop, order);
      _intra.wait(coop, order);
    }
  }
};
#endif

#endif // FLAGCX_DEVICE_COMPILE (Sections 9b-12)

#endif // FLAGCX_DEVICE_API_H_
