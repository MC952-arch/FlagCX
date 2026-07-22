/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM CommTraits — device-side backend using NVSHMEM PGAS APIs.
 * Provides CommTraits<NvshmemBackend> specialization with:
 *   - Comm, Team, Window, Multimem (data types)
 *   - Net (one-sided: put, putValue, signal, flush, wait)
 *   - Barrier specializations (intra/inter/world via nvshmemx_signal_op)
 ************************************************************************/

#ifndef FLAGCX_NVSHMEM_COMM_TRAITS_H_
#define FLAGCX_NVSHMEM_COMM_TRAITS_H_

#include "flagcx_kernel_core.h"
#include <nvshmem.h>
#include <nvshmemx.h>

struct NvshmemBackend {};

template <>
struct CommTraits<NvshmemBackend> {
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;

  // ---- Multimem ----
  struct Multimem {
    void *mcBasePtr;
  };

  // ---- Team ----
  struct Team {
    int nRanks, rank, stride;
  };

  // ---- Window ----
  struct Window {
    void *symBase;
    size_t allocSize;
    void *rawPtr;

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &, int) const {
      return (char *)symBase + offset;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      return (char *)rawPtr + offset;
    }
    FLAGCX_HOST_DEVICE_INLINE void *getRawPtr() const { return rawPtr; }
    FLAGCX_HOST_DEVICE_INLINE bool hasAccess() const {
      return symBase != nullptr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getMrIndex() const { return 0; }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator==(const Window &o) const {
      return symBase == o.symBase;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator!=(const Window &o) const {
      return !(*this == o);
    }
  };

  // ---- Comm ----
  struct Comm {
    int rank, nRanks;
    int intraRank, intraSize;
    nvshmem_team_t intraTeam;
    nvshmem_team_t interTeam;

    uint64_t *signalBuffer;
    int signalCount;
    uint64_t *counterBuffer;
    int counterCount;
    uint64_t *shadowBuffer;

    uint64_t *intraBarrierSignals;
    uint64_t *interBarrierSignals;
    uint64_t *worldBarrierSignals;
    uint64_t *barrierUsage;

    int intraBarrierCount;
    int interBarrierCount;
    int worldBarrierCount;

    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
      return intraRank;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
      return intraSize;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return nRanks; }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer(int) const {
      return nullptr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR Multimem getMulticastHandle() const {
      Multimem mm;
      mm.mcBasePtr = nullptr;
      return mm;
    }

    template <typename DI>
    static FLAGCX_HOST_DEVICE_INLINE void populateFromInternal(Comm &dc,
                                                               const DI &di) {
      dc.rank = di.rank;
      dc.nRanks = di.nRanks;
      dc.intraRank = di.intraRank;
      dc.intraSize = di.intraSize;
      dc.intraTeam = di.intraTeam;
      dc.interTeam = di.interTeam;
      dc.signalBuffer = di.signalBuffer;
      dc.signalCount = di.signalCount;
      dc.counterBuffer = di.counterBuffer;
      dc.counterCount = di.counterCount;
      dc.shadowBuffer = di.shadowBuffer;
      dc.intraBarrierSignals = di.intraBarrierSignals;
      dc.interBarrierSignals = di.interBarrierSignals;
      dc.worldBarrierSignals = di.worldBarrierSignals;
      dc.barrierUsage = di.barrierUsage;
      dc.intraBarrierCount = di.intraBarrierCount;
      dc.interBarrierCount = di.interBarrierCount;
      dc.worldBarrierCount = di.worldBarrierCount;
    }
  };

  // ---- Coop types: aliased from PlatformTraits ----
  using CoopBlock = typename PlatformTraits<NvidiaPlatform>::CoopBlock;
  template <int N>
  using CoopTile =
      typename PlatformTraits<NvidiaPlatform>::template CoopTile<N>;
  using CoopThread = typename PlatformTraits<NvidiaPlatform>::CoopThread;
  using CoopWarp = typename PlatformTraits<NvidiaPlatform>::CoopWarp;
  using CoopTileSpan = typename PlatformTraits<NvidiaPlatform>::CoopTileSpan;
  using CoopLanes = typename PlatformTraits<NvidiaPlatform>::CoopLanes;
  using CoopAny = typename PlatformTraits<NvidiaPlatform>::CoopAny;

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    int nBarriers;
  };
  struct InterBarrierHandle {
    int placeholder;
  };

  // ---- DescriptorSmem: empty for NVSHMEM ----
  struct DescriptorSmem {};

  // ---- Barrier alias ----
  template <typename Tag, typename Coop>
  using Barrier = ::Barrier<NvshmemBackend, Tag, Coop>;

  // ---- Net ----
  struct Net {
    Comm _dc;
    int _contextId;

    FLAGCX_DEVICE_INLINE_DECORATOR
    Net(const Comm &dc, int contextIndex) : _dc(dc), _contextId(contextIndex) {}

    FLAGCX_DEVICE_INLINE_DECORATOR bool isValid() const { return true; }

    // ---- Helper: resolve PE from team + peer index ----
    FLAGCX_DEVICE_INLINE_DECORATOR int resolvePE(Team team, int peer) const {
      // team.rank is my rank within team; peer is absolute rank
      // For NVSHMEM, peer IS the PE number (world-scope)
      return peer;
    }

    // ---- One-sided: put ----
    template <typename RA, typename LA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)team;
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        void *dstPtr = (char *)dst.symBase + dstOff;
        void *srcPtr = (char *)src.rawPtr + srcOff;
        int pe = peer;
        putImpl(dstPtr, srcPtr, bytes, pe, ra, la);
      }
      coop.sync();
    }

    // ---- One-sided: putValue ----
    template <typename T, typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putValue(Team team, int peer, Window dst, size_t dstOff, T value, RA ra,
             Coop coop, Desc desc, flagcxDeviceScope_t ar,
             flagcxDeviceScope_t es) const {
      (void)team;
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        void *dstPtr = (char *)dst.symBase + dstOff;
        nvshmem_putmem(dstPtr, (const void *)&value, sizeof(T), peer);
        signalImpl(peer, ra);
      }
      coop.sync();
    }

    // ---- One-sided: signal ----
    template <typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signal(Team team, int peer, RA ra, Coop coop, Desc desc,
           flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)team;
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        signalImpl(peer, ra);
      }
      coop.sync();
    }

    // ---- Ordering: flush ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    flush(Coop coop, flagcxDeviceMemoryOrder_t order) const {
      if (order == flagcxDeviceMemoryOrderAcqRel) {
        coop.sync();
        if (coop.threadRank() == 0)
          nvshmem_quiet();
        coop.sync();
      } else {
        nvshmem_fence();
      }
    }

    // ---- Wait: waitSignal ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, flagcxDevNetSignal_t signalId, uint64_t least,
               int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t *addr = _dc.signalBuffer + (int)signalId;
        nvshmem_uint64_wait_until(addr, NVSHMEM_CMP_GE, least);
      }
      coop.sync();
    }

    // ---- Wait: waitSignalMeetShadow ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, flagcxDevNetSignal_t signalId, int bits,
                         flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t target = _dc.shadowBuffer[(int)signalId];
        uint64_t *addr = _dc.signalBuffer + (int)signalId;
        nvshmem_uint64_wait_until(addr, NVSHMEM_CMP_GE, target);
      }
      coop.sync();
    }

    // ---- Wait: waitSignalFollowShadow ----
    template <typename Coop, typename Uint>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, flagcxDevNetSignal_t signalId,
                           Uint leastDelta, Uint *before, Uint *delta, int bits,
                           flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t shadow = _dc.shadowBuffer[(int)signalId];
        uint64_t target = shadow + (uint64_t)leastDelta;
        uint64_t *addr = _dc.signalBuffer + (int)signalId;
        nvshmem_uint64_wait_until(addr, NVSHMEM_CMP_GE, target);
        uint64_t cur = Atomic::load(addr, flagcxDeviceMemoryOrderAcquire);
        if (before)
          *before = (Uint)shadow;
        if (delta)
          *delta = (Uint)(cur - shadow);
      }
      coop.sync();
    }

    // ---- Shadow access ----
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalShadowPtr(flagcxDevNetSignal_t signalId) const {
      return &_dc.shadowBuffer[(int)signalId];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    increaseSignalShadow(flagcxDevNetSignal_t signalId, uint64_t delta) const {
      _dc.shadowBuffer[(int)signalId] += delta;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(flagcxDevNetSignal_t signalId, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      return Atomic::load(&_dc.signalBuffer[(int)signalId],
                          flagcxDeviceMemoryOrderAcquire);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetSignal(flagcxDevNetSignal_t signalId) const {
      Atomic::store(&_dc.signalBuffer[(int)signalId], (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }

    // ---- Counter: waitCounter ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, flagcxDevNetCounter_t counterId, uint64_t least,
                int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = (int)counterId;
        int iter = 0;
        while (Atomic::load(&_dc.counterBuffer[idx],
                            flagcxDeviceMemoryOrderAcquire) < least) {
          Intrin::spinBackoff(iter++);
        }
      }
      coop.sync();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(flagcxDevNetCounter_t counterId, int bits,
                flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      return Atomic::load(&_dc.counterBuffer[(int)counterId],
                          flagcxDeviceMemoryOrderAcquire);
    }

  private:
    // ---- put dispatch: select fused put+signal vs plain put ----
    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe,
            flagcxDevNet_SignalInc ra, LA la) const {
      uint64_t *sigAddr = _dc.signalBuffer + (int)ra.signal;
      nvshmem_putmem_signal(dst, src, bytes, sigAddr, 1, NVSHMEM_SIGNAL_ADD,
                            pe);
      counterImpl(la);
    }

    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe,
            flagcxDevNet_SignalAdd ra, LA la) const {
      uint64_t *sigAddr = _dc.signalBuffer + (int)ra.signal;
      nvshmem_putmem_signal(dst, src, bytes, sigAddr, ra.value,
                            NVSHMEM_SIGNAL_ADD, pe);
      counterImpl(la);
    }

    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe, flagcxDevNet_CounterInc,
            LA) const {
      nvshmem_putmem(dst, src, bytes, pe);
      Atomic::fetchAdd(&_dc.counterBuffer[0], (uint64_t)1,
                       flagcxDeviceMemoryOrderRelease);
    }

    template <typename RA, typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe, RA, LA la) const {
      nvshmem_putmem(dst, src, bytes, pe);
      counterImpl(la);
    }

    // ---- signal dispatch ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signalImpl(int pe, flagcxDevNet_SignalInc ra) const {
      uint64_t *sigAddr = _dc.signalBuffer + (int)ra.signal;
      nvshmemx_signal_op(sigAddr, 1, NVSHMEM_SIGNAL_ADD, pe);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signalImpl(int pe, flagcxDevNet_SignalAdd ra) const {
      uint64_t *sigAddr = _dc.signalBuffer + (int)ra.signal;
      nvshmemx_signal_op(sigAddr, ra.value, NVSHMEM_SIGNAL_ADD, pe);
    }
    template <typename RA>
    FLAGCX_DEVICE_INLINE_DECORATOR void signalImpl(int, RA) const {}

    // ---- counter helper ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    counterImpl(flagcxDevNet_CounterInc c) const {
      Atomic::fetchAdd(&_dc.counterBuffer[(int)c.counter], (uint64_t)1,
                       flagcxDeviceMemoryOrderRelease);
    }
    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void counterImpl(LA) const {}
  }; // struct Net
};   // struct CommTraits<NvshmemBackend>

// ============================================================
// Barrier specializations for NvshmemBackend
// Signal-based split-phase barriers using nvshmemx_signal_op.
// ============================================================

// ---- Barrier<NvshmemBackend, flagcxTeamTagIntra, Coop> ----
template <typename Coop>
struct Barrier<NvshmemBackend, flagcxTeamTagIntra, Coop> {
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;
  using Comm = CommTraits<NvshmemBackend>::Comm;
  using Team = CommTraits<NvshmemBackend>::Team;
  using Multimem = CommTraits<NvshmemBackend>::Multimem;

  Coop _coop;
  nvshmem_team_t _team;
  int _teamSize, _teamRank;
  uint64_t *_barrierSignals;
  uint64_t *_usageCount;

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _team(NVSHMEM_TEAM_INVALID), _teamSize(0), _teamRank(0),
        _barrierSignals(nullptr), _usageCount(nullptr) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Comm &dc, Team team, uint32_t index, bool = false,
          const Multimem & = {})
      : _coop(coop), _team(dc.intraTeam), _teamSize(dc.intraSize),
        _teamRank(dc.intraRank),
        _barrierSignals(dc.intraBarrierSignals + index * dc.intraSize),
        _usageCount(&dc.barrierUsage[index]) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _coop.sync();
    if (_coop.threadRank() == 0)
      nvshmem_fence();
    _coop.sync();
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      int peerPE = nvshmem_team_translate_pe(_team, peer, NVSHMEM_TEAM_WORLD);
      nvshmemx_signal_op(_barrierSignals + _teamRank, 1, NVSHMEM_SIGNAL_ADD,
                         peerPE);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    uint64_t target = *_usageCount + 1;
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      nvshmem_uint64_wait_until(_barrierSignals + peer, NVSHMEM_CMP_GE, target);
    }
    _coop.sync();
    if (_coop.threadRank() == 0)
      *_usageCount = target;
    _coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(order);
    wait(order);
  }
};

// ---- Barrier<NvshmemBackend, flagcxTeamTagInter, Coop> ----
template <typename Coop>
struct Barrier<NvshmemBackend, flagcxTeamTagInter, Coop> {
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;
  using Comm = CommTraits<NvshmemBackend>::Comm;
  using Team = CommTraits<NvshmemBackend>::Team;
  using Multimem = CommTraits<NvshmemBackend>::Multimem;

  Coop _coop;
  nvshmem_team_t _team;
  int _teamSize, _teamRank;
  uint64_t *_barrierSignals;
  uint64_t *_usageCount;

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _team(NVSHMEM_TEAM_INVALID), _teamSize(0), _teamRank(0),
        _barrierSignals(nullptr), _usageCount(nullptr) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Comm &dc, Team team, uint32_t index, bool = false,
          const Multimem & = {})
      : _coop(coop), _team(dc.interTeam),
        _teamSize((dc.intraSize > 0) ? dc.nRanks / dc.intraSize : 1),
        _teamRank((dc.intraSize > 0) ? dc.rank / dc.intraSize : 0),
        _barrierSignals(
            dc.interBarrierSignals +
            index * ((dc.intraSize > 0) ? dc.nRanks / dc.intraSize : 1)),
        _usageCount(&dc.barrierUsage[dc.intraBarrierCount + index]) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _coop.sync();
    if (_coop.threadRank() == 0)
      nvshmem_fence();
    _coop.sync();
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      int peerPE = nvshmem_team_translate_pe(_team, peer, NVSHMEM_TEAM_WORLD);
      nvshmemx_signal_op(_barrierSignals + _teamRank, 1, NVSHMEM_SIGNAL_ADD,
                         peerPE);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    uint64_t target = *_usageCount + 1;
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      nvshmem_uint64_wait_until(_barrierSignals + peer, NVSHMEM_CMP_GE, target);
    }
    _coop.sync();
    if (_coop.threadRank() == 0)
      *_usageCount = target;
    _coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(order);
    wait(order);
  }
};

// ---- Barrier<NvshmemBackend, flagcxTeamTagWorld, Coop> ----
template <typename Coop>
struct Barrier<NvshmemBackend, flagcxTeamTagWorld, Coop> {
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;
  using Comm = CommTraits<NvshmemBackend>::Comm;
  using Team = CommTraits<NvshmemBackend>::Team;
  using Multimem = CommTraits<NvshmemBackend>::Multimem;

  Coop _coop;
  int _teamSize, _teamRank;
  uint64_t *_barrierSignals;
  uint64_t *_usageCount;

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _teamSize(0), _teamRank(0), _barrierSignals(nullptr),
        _usageCount(nullptr) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Comm &dc, Team team, uint32_t index, bool = false,
          const Multimem & = {})
      : _coop(coop), _teamSize(dc.nRanks), _teamRank(dc.rank),
        _barrierSignals(dc.worldBarrierSignals + index * dc.nRanks),
        _usageCount(&dc.barrierUsage[dc.intraBarrierCount +
                                     dc.interBarrierCount + index]) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _coop.sync();
    if (_coop.threadRank() == 0)
      nvshmem_fence();
    _coop.sync();
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      // World barrier uses NVSHMEM_TEAM_WORLD: PE = peer directly
      nvshmemx_signal_op(_barrierSignals + _teamRank, 1, NVSHMEM_SIGNAL_ADD,
                         peer);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    uint64_t target = *_usageCount + 1;
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      nvshmem_uint64_wait_until(_barrierSignals + peer, NVSHMEM_CMP_GE, target);
    }
    _coop.sync();
    if (_coop.threadRank() == 0)
      *_usageCount = target;
    _coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(order);
    wait(order);
  }
};

#endif // FLAGCX_NVSHMEM_COMM_TRAITS_H_
