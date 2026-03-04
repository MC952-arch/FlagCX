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
#include "flagcx.h"

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
// Unified capability-based design: baseline always populated,
// IPC and NCCL layers added when available.
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
  uint32_t *localBarrierFlags; // this rank's barrier memory (CTA_COUNT entries)
  uint32_t barrierEpoch; // monotonically increasing, set by host before launch
  // Host-side cleanup bookkeeping (not passed to kernel)
  void **peerBarrierPtrs; // host array of IPC-mapped pointers (for close)
  int *localRankToRank;   // intra-node rank mapping (for IPC exchange)
  int nLocalRanks;

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- NCCL layer (set if ncclDevCommCreate succeeds) ----
  ncclDevComm ncclDev;
  bool hasNcclDev;
#endif
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
//   hasWindow == true       → Window available (Tier 1 only)
// ============================================================
struct flagcxDevMemInternal {
  // ---- Baseline (always set) ----
  void *rawPtr; // = buff parameter

  // ---- IPC layer (set if IPC exchange succeeds, else nullptr) ----
  void **devPeerPtrs; // device array: [localRank] -> peer buffer ptr
  int nPeers;
  int intraRank;       // this rank's local rank index (for IPC local pointer)
  void **hostPeerPtrs; // host array: for ipcMemHandleClose cleanup
  void *basePtr;       // this rank's buffer pointer (for IPC close check)

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- Window layer (set if window registration succeeds) ----
  ncclWindow_t ncclWin;
  void *winHandle;
  bool hasWindow;
#endif
};
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

// ============================================================
// Section 3: flagcxDevComm — Device Communicator (kernel-facing)
//
// Value type passed to kernels by value.
// Unified capability-based design: baseline fields always valid,
// IPC barrier pointers present (may be nullptr),
// NCCL device comm present only on Tier 1 (guarded by _hasBase).
// Constructor from flagcxDevCommInternal enables
// tier-agnostic host code: flagcxDevComm dc(*devCommHandle);
// ============================================================
struct flagcxDevComm {
  // ---- Baseline (always valid) ----
  int _rank, _nRanks;
  int _intraRank, _intraSize;
  void *_fifoBuffer; // FIFO for device Send/Recv (from heteroComm, may be null)

  // ---- IPC layer (may be nullptr if IPC barrier not set up) ----
  uint32_t **_barrierPeers;
  uint32_t _barrierEpoch;

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- NCCL layer (valid only if _hasBase is true) ----
  ncclDevComm _base;
  bool _hasBase;
#endif

  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm()
      : _rank(0), _nRanks(0), _intraRank(0), _intraSize(0),
        _fifoBuffer(nullptr), _barrierPeers(nullptr), _barrierEpoch(0)
#ifdef FLAGCX_DEVICE_API_NCCL
        ,
        _base(), _hasBase(false)
#endif
  {
  }

  // Unified constructor from host handle
  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm(const flagcxDevCommInternal &di)
      : _rank(di.rank), _nRanks(di.nRanks), _intraRank(di.intraRank),
        _intraSize(di.intraSize), _fifoBuffer(di.fifoBuffer),
        _barrierPeers(di.barrierPeers), _barrierEpoch(di.barrierEpoch)
#ifdef FLAGCX_DEVICE_API_NCCL
        ,
        _base(di.ncclDev), _hasBase(di.hasNcclDev)
#endif
  {
  }

  // Accessors (unified — always use baseline fields)
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const { return _intraRank; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const { return _intraSize; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return _rank; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return _nRanks; }
  FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer() const {
    return _fifoBuffer;
  }
};

// ============================================================
// Section 4: flagcxDevMem — Device-Side Memory Handle
//
// Value type passed to kernels by value.
// Unified capability-based design: rawPtr always valid,
// IPC peerPtrs present (may be nullptr),
// Window _base present only on Tier 1 (guarded by _hasWindow).
// Runtime dispatch in flagcxGetPeerPointer uses priority:
//   Window > IPC > Raw.
// Constructor from flagcxDevMemInternal enables
// tier-agnostic host code: flagcxDevMem dm(*devMemHandle);
// ============================================================
struct flagcxDevMem {
  // ---- Baseline (always valid) ----
  void *rawPtr;

  // ---- IPC layer (may be nullptr) ----
  void **peerPtrs; // device array [localRank] -> peer buffer
  int intraRank;   // local rank index (for flagcxGetLocalPointer in IPC mode)

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- Window layer (valid only if _hasWindow is true) ----
  ncclWindow_t _base;
  bool _hasWindow;
#endif

  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem()
      : rawPtr(nullptr), peerPtrs(nullptr), intraRank(0)
#ifdef FLAGCX_DEVICE_API_NCCL
        ,
        _base(), _hasWindow(false)
#endif
  {
  }

  // Unified constructor from host handle
  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem(const flagcxDevMemInternal &di)
      : rawPtr(di.rawPtr), peerPtrs(di.devPeerPtrs), intraRank(di.intraRank)
#ifdef FLAGCX_DEVICE_API_NCCL
        ,
        _base(di.ncclWin), _hasWindow(di.hasWindow)
#endif
  {
  }
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
// On Tier 1 with _hasBase: uses NCCL team functions.
// Otherwise: computes from baseline fields.
// ============================================================
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamIntra(const flagcxDevComm &devComm) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm._hasBase)
    return flagcxTeam_t(ncclTeamLsa(devComm._base));
#endif
  flagcxTeam_t team;
  team.nRanks = devComm.getIntraSize();
  team.rank = devComm.getIntraRank();
  team.stride = 1;
  return team;
}
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamWorld(const flagcxDevComm &devComm) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm._hasBase)
    return flagcxTeam_t(ncclTeamWorld(devComm._base));
#endif
  flagcxTeam_t team;
  team.nRanks = devComm.getSize();
  team.rank = devComm.getRank();
  team.stride = 1;
  return team;
}
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamInter(const flagcxDevComm &devComm) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm._hasBase)
    return flagcxTeam_t(ncclTeamRail(devComm._base));
#endif
  flagcxTeam_t team;
  team.nRanks = devComm.getSize() / devComm.getIntraSize();
  team.rank = devComm.getRank() / devComm.getIntraSize();
  team.stride = devComm.getIntraSize();
  return team;
}

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
// Priority dispatch: Window > IPC > Raw.
// Capabilities detected by _hasWindow flag and peerPtrs null-check.
// ============================================================
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDevMem &mem, size_t offset, flagcxTeam_t team,
                     int peer) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (mem._hasWindow)
    return ncclGetPeerPointer(mem._base, offset, team._base, peer);
#endif
  if (mem.peerPtrs != nullptr) {
    int index = team.rank + (peer - team.rank) * team.stride;
    return (char *)mem.peerPtrs[index] + offset;
  }
  return nullptr; // Raw mode: no peer access
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointer(const flagcxDevMem &mem, size_t offset) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (mem._hasWindow)
    return ncclGetLocalPointer(mem._base, offset);
#endif
  if (mem.peerPtrs != nullptr)
    return (char *)mem.peerPtrs[mem.intraRank] + offset;
  return (char *)mem.rawPtr + offset;
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDevMem &mem, size_t offset,
                          const flagcxDevComm &devComm) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (mem._hasWindow)
    return ncclGetLsaMultimemPointer(mem._base, offset, devComm._base);
#endif
  return nullptr; // IPC/raw mode: multicast not available
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

// Forward declarations of device-side FIFO functions (defined in
// flagcx_kernel_device.cu, declared in flagcx_kernel.h). Needed by
// flagcxDevNet::send/recv/term/wait and Tier 2 flagcxBarrierSession::sync.
FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceSend(const void *sendbuff, size_t count, flagcxDataType_t datatype,
                 int peer, const flagcxDevComm &devComm);
FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceRecv(void *recvbuff, size_t count, flagcxDataType_t datatype,
                 int peer, const flagcxDevComm &devComm);
FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceTerm(const flagcxDevComm &devComm);
FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceWait(const flagcxDevComm &devComm);

// ============================================================
// Section 9b: GIN Types (Tier 1 only)
// ============================================================
// Fence level enum — available on all tiers for unified barrier API
enum class flagcxGinFenceLevel { Relaxed };

#ifdef FLAGCX_DEVICE_API_NCCL
typedef uint32_t flagcxDevNetSignal_t;
typedef uint32_t flagcxDevNetCounter_t;

struct flagcxDevNet_None {};
struct flagcxDevNet_SignalInc {
  flagcxDevNetSignal_t signal;
};
struct flagcxDevNet_SignalAdd {
  flagcxDevNetSignal_t signal;
  uint64_t value;
};
struct flagcxDevNet_CounterInc {
  flagcxDevNetCounter_t counter;
};

// Action type mapping helpers (flagcx -> nccl)
FLAGCX_DEVICE_INLINE_DECORATOR ncclGin_None toNccl(flagcxDevNet_None) {
  return {};
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclGin_SignalInc
toNccl(flagcxDevNet_SignalInc a) {
  return {a.signal};
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclGin_SignalAdd
toNccl(flagcxDevNet_SignalAdd a) {
  return {a.signal, a.value};
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclGin_CounterInc
toNccl(flagcxDevNet_CounterInc a) {
  return {a.counter};
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclCoopCta toNccl(flagcxCoopBlock) {
  return {};
}
#endif // FLAGCX_DEVICE_API_NCCL

// ============================================================
// Section 10: flagcxDevNet — Device Network (all tiers)
// ============================================================
struct flagcxDevNet {
  const flagcxDevComm &_devComm; // for barrier + Send/Recv on all tiers

#ifdef FLAGCX_DEVICE_API_NCCL
  ncclGin _gin; // GIN backend (Tier 1 only)

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevNet(const flagcxDevComm &dc, int contextIndex = 0)
      : _devComm(dc), _gin(dc._base, contextIndex) {}
#else
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevNet(const flagcxDevComm &dc, int contextIndex = 0) : _devComm(dc) {}
#endif

  // ---- Two-sided operations (all tiers, via FIFO) ----
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t send(const void *buff,
                                                     size_t count,
                                                     flagcxDataType_t datatype,
                                                     int peer) const {
    return flagcxDeviceSend(buff, count, datatype, peer, _devComm);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t recv(void *buff, size_t count,
                                                     flagcxDataType_t datatype,
                                                     int peer) const {
    return flagcxDeviceRecv(buff, count, datatype, peer, _devComm);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t term() const {
    return flagcxDeviceTerm(_devComm);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t wait() const {
    return flagcxDeviceWait(_devComm);
  }

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- GIN one-sided operations (Tier 1 only) ----

  template <typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t team, int peer, const flagcxDevMem &dstMem, size_t dstOffset,
      const flagcxDevMem &srcMem, size_t srcOffset, size_t bytes,
      RemoteAction remoteAction = flagcxDevNet_None{},
      Coop coop = flagcxCoopBlock{}) const {
    _gin.put(team._base, peer, dstMem._base, dstOffset, srcMem._base, srcOffset,
             bytes, toNccl(remoteAction), ncclGin_None{}, toNccl(coop));
  }

  template <typename RemoteAction, typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  signal(flagcxTeam_t team, int peer, RemoteAction remoteAction,
         Coop coop = flagcxCoopBlock{}) const {
    _gin.signal(team._base, peer, toNccl(remoteAction), toNccl(coop));
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void flush(
      Coop coop,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _gin.flush(toNccl(coop), flagcxDeviceMemoryOrderMap[order]);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignal(
      Coop coop, flagcxDevNetSignal_t signal, uint64_t least,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _gin.waitSignal(toNccl(coop), signal, least, 64,
                    flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readSignal(
      flagcxDevNetSignal_t signal,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    return _gin.readSignal(signal, 64, flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  resetSignal(flagcxDevNetSignal_t signal) const {
    _gin.resetSignal(signal);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitCounter(
      Coop coop, flagcxDevNetCounter_t counter, uint64_t least,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _gin.waitCounter(toNccl(coop), counter, least, 56,
                     flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readCounter(
      flagcxDevNetCounter_t counter,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    return _gin.readCounter(counter, 56, flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  resetCounter(flagcxDevNetCounter_t counter) const {
    _gin.resetCounter(counter);
  }
#endif // FLAGCX_DEVICE_API_NCCL
};

// ============================================================
// Section 11: flagcxInterBarrierSession — GIN Barrier (Tier 1 only)
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
template <typename Coop>
struct flagcxInterBarrierSession {
  ncclGinBarrierSession<ncclCoopCta> _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession(Coop coop, const flagcxDevNet &net,
                            flagcxTeam_t team, uint32_t index)
      : _impl(ncclCoopCta(), net._gin, team._base, net._gin.comm.railGinBarrier,
              index) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    _impl.sync(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order],
               ncclGinFenceLevel::Relaxed);
  }
};
#endif

// ============================================================
// Section 12: flagcxBarrierSession — Unified Barrier (both tiers)
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
template <typename Coop>
struct flagcxBarrierSession {
  ncclBarrierSession<ncclCoopCta> _impl;

  // World barrier (intra + inter)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagWorld, const flagcxDevNet &net,
                       uint32_t index)
      : _impl(ncclCoopCta(), ncclTeamTagWorld(), net._gin, index) {}

  // Intra-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagIntra,
                       const flagcxDevComm &devComm, uint32_t index)
      : _impl(ncclCoopCta(), ncclTeamTagLsa(), devComm._base, index) {}

  // Inter-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagInter, const flagcxDevNet &net,
                       uint32_t index)
      : _impl(ncclCoopCta(), ncclTeamTagRail(), net._gin, index) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    _impl.sync(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order],
               ncclGinFenceLevel::Relaxed);
  }
};
#else
// Tier 2/3: World barrier uses Term/Wait (FIFO-based, covers both intra +
// inter)
//           Intra barrier uses flagcxIntraBarrierSession (IPC-based, multi-CTA)
//           Inter barrier not available standalone (use World instead)
template <typename Coop>
struct flagcxBarrierSession {
  bool _useTermWait;                      // true=World (Term/Wait), false=Intra
  flagcxDevComm _devCommCopy;             // copy for Term/Wait in World mode
  flagcxIntraBarrierSession<Coop> _intra; // used in Intra mode

  // World barrier: Term + Wait via FIFO (single-CTA constraint)
  // flagcxDeviceTerm signals host proxy to execute all pending ops,
  // flagcxDeviceWait spins until all FIFO items consumed — full global barrier.
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagWorld, const flagcxDevNet &net,
                       uint32_t index)
      : _useTermWait(true), _devCommCopy(net._devComm),
        _intra(coop, net._devComm, flagcxTeamIntra(net._devComm), index) {}

  // Intra-only barrier: IPC-based (multi-CTA)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagIntra,
                       const flagcxDevComm &devComm, uint32_t index)
      : _useTermWait(false), _devCommCopy(),
        _intra(coop, devComm, flagcxTeamIntra(devComm), index) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_useTermWait) {
      // World barrier: Term + Wait (all threads must sync before thread 0 acts)
      FLAGCX_DEVICE_SYNC_THREADS();
      if (threadIdx.x == 0) {
        flagcxDeviceTerm(_devCommCopy);
        flagcxDeviceWait(_devCommCopy);
      }
      FLAGCX_DEVICE_SYNC_THREADS();
    } else {
      // Intra-only barrier: IPC-based
      _intra.sync(coop, order);
    }
  }
};
#endif

#endif // FLAGCX_DEVICE_COMPILE (Sections 9b-12)

#endif // FLAGCX_DEVICE_API_H_
