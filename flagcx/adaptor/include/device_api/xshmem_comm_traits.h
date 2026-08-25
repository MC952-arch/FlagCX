#ifndef FLAGCX_XSHMEM_COMM_TRAITS_H_
#define FLAGCX_XSHMEM_COMM_TRAITS_H_

#include "flagcx_kernel_core.h"
#include <cstddef>
#include <cstdint>

#ifndef __xpu__
#error                                                                         \
    "xshmem_comm_traits.h requires the XPU toolchain: compile with xpu-clang --xpu-arch=xpu3 (host pass included)"
#endif
#include "xpu/kernel/xtdk.h"
#include "xshmem/xshmem.h"
#include "xshmem/xshmemx.h"
#define XSHMEM_FGP __global_ptr__

#define XSHMEM_DEVICE_INLINE __device__ inline
#define XSHMEM_HOST_DEVICE_INLINE __device__ inline

struct XshmemBackend {};

template <>
struct CommTraits<XshmemBackend> {
  // ---- Multimem ----
  struct Multimem {
    void *mcBasePtr;
  };

  // ---- Team ----
  // XSHMEM has no team objects; PE ids equal global ranks.
  struct Team {
    int nRanks, rank, stride;
  };

  // ---- Local scratch (per-core LM staging buffer) ----
  static XSHMEM_DEVICE_INLINE float *localScratch() {
    return (float *)get_xshmemi_local_buf();
  }
  static XSHMEM_DEVICE_INLINE int localScratchBytes() {
    return XSHMEMI_LOCAL_BUF_LEN;
  }

  // ---- Window ----
  struct Window {
    XSHMEM_FGP void *symBase;
    size_t allocSize;
    XSHMEM_FGP void *rawPtr;

    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *getPeerPointer(size_t, const Team &,
                                                         int) const {
      return nullptr;
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *getLocalPointer(size_t) const {
      return nullptr;
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *getIntraPointer(size_t, int) const {
      return nullptr;
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *
    getMulticastPointer(size_t, const void *) const {
      return nullptr;
    }
    XSHMEM_HOST_DEVICE_INLINE XSHMEM_FGP void *getRawPtr() const {
      return nullptr;
    }
    XSHMEM_HOST_DEVICE_INLINE bool hasAccess() const { return false; }
    XSHMEM_HOST_DEVICE_INLINE void **getDevPeerPtrs() const { return nullptr; }
    XSHMEM_HOST_DEVICE_INLINE int getMrIndex() const { return 0; }
    XSHMEM_DEVICE_INLINE bool operator==(const Window &) const { return false; }
    XSHMEM_DEVICE_INLINE bool operator!=(const Window &) const { return true; }
  };

  // ---- Comm ----
  struct Comm {
    int rank, nRanks;
    int intraRank, intraSize;
    int intraTeam;
    int interTeam;
    int worldTeam;

    XSHMEM_FGP uint64_t *signalBuffer;
    int signalCount;
    XSHMEM_FGP uint64_t *counterBuffer;
    int counterCount;
    XSHMEM_FGP uint64_t *shadowBuffer;

    XSHMEM_FGP uint64_t *gridSyncState;

    XSHMEM_FGP void *devStateHandle;

    XSHMEM_DEVICE_INLINE int getIntraRank() const { return 0; }
    XSHMEM_DEVICE_INLINE int getIntraSize() const { return 0; }
    XSHMEM_DEVICE_INLINE int getRank() const { return 0; }
    XSHMEM_DEVICE_INLINE int getSize() const { return 0; }
    XSHMEM_DEVICE_INLINE void *getFifoBuffer(int) const { return nullptr; }
    XSHMEM_DEVICE_INLINE Multimem getMulticastHandle() const {
      Multimem mm;
      mm.mcBasePtr = nullptr;
      return mm;
    }

    template <typename DI>
    static XSHMEM_HOST_DEVICE_INLINE void populateFromInternal(Comm &,
                                                               const DI &) {}
  };

  // ---- Coop types: cluster-scoped (xshmem's cooperative unit is the
  // cluster) ----
  struct CoopBlock {
    XSHMEM_DEVICE_INLINE int threadRank() const { return core_id(); }
    XSHMEM_DEVICE_INLINE int size() const { return 64; }
    XSHMEM_DEVICE_INLINE void sync() const {
      xshmemi_threadgroup_sync<XSHMEMI_THREADGROUP_CLUSTER>();
    }
  };
  template <int N>
  struct CoopTile {
    XSHMEM_DEVICE_INLINE int threadRank() const { return core_id(); }
    XSHMEM_DEVICE_INLINE int size() const { return N; }
    XSHMEM_DEVICE_INLINE void sync() const {
      xshmemi_threadgroup_sync<XSHMEMI_THREADGROUP_CLUSTER>();
    }
  };
  using CoopThread = CoopTile<1>;
  using CoopWarp = CoopTile<64>;
  struct CoopTileSpan {
    CoopTileSpan(int, int, int) {}
    XSHMEM_DEVICE_INLINE int threadRank() const { return core_id(); }
    XSHMEM_DEVICE_INLINE int size() const { return 64; }
    XSHMEM_DEVICE_INLINE void sync() const {
      xshmemi_threadgroup_sync<XSHMEMI_THREADGROUP_CLUSTER>();
    }
  };
  struct CoopLanes {
    CoopLanes(uint32_t = 1u) {}
    XSHMEM_DEVICE_INLINE int threadRank() const { return core_id(); }
    XSHMEM_DEVICE_INLINE int size() const { return 64; }
    XSHMEM_DEVICE_INLINE void sync() const {
      xshmemi_threadgroup_sync<XSHMEMI_THREADGROUP_CLUSTER>();
    }
  };
  using CoopAny = PlatformCoop;

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    int nBarriers;
  };
  struct InterBarrierHandle {
    int placeholder;
  };

  // ---- DescriptorSmem: empty for XSHMEM ----
  struct DescriptorSmem {};

  // ---- Barrier alias ----
  template <typename Tag, typename Coop>
  using Barrier = ::Barrier<XshmemBackend, Tag, Coop>;

  // ---- Net ----
  struct Net {
    Comm _dc;

    XSHMEM_HOST_DEVICE_INLINE
    Net(const Comm &dc, int /*contextIndex*/) : _dc(dc) {}

    XSHMEM_DEVICE_INLINE bool isValid() const { return true; }

    // ---- Helper: resolve PE from team + peer index ----
    static XSHMEM_DEVICE_INLINE int resolvePE(const Comm &dc, Team team,
                                              int peer) {
      int base = dc.rank - team.rank * team.stride;
      return base + peer * team.stride;
    }

    // ---- One-sided: put ----
    template <typename RA, typename LA, typename Coop, typename Desc>
    XSHMEM_DEVICE_INLINE void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      (void)coop;
      int pe = resolvePE(_dc, team, peer);
      putImpl(_dc,
              (XSHMEM_FGP float *)((XSHMEM_FGP char *)dst.symBase + dstOff),
              (XSHMEM_FGP float *)((XSHMEM_FGP char *)src.rawPtr + srcOff),
              bytes, pe, ra, la);
    }

    // ---- One-sided: putValue ----
    template <typename T, typename RA, typename Coop, typename Desc>
    XSHMEM_DEVICE_INLINE void putValue(Team, int, Window, size_t, T, RA, Coop,
                                       Desc, flagcxDeviceScope_t,
                                       flagcxDeviceScope_t) const {}

    // ---- One-sided: signal ----
    template <typename RA, typename Coop, typename Desc>
    XSHMEM_DEVICE_INLINE void signal(Team team, int peer, RA ra, Coop coop,
                                     Desc desc, flagcxDeviceScope_t ar,
                                     flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      (void)coop;
      int pe = resolvePE(_dc, team, peer);
      signalImpl(_dc, pe, ra);
    }

    // ---- Ordering: flush ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void flush(Coop coop,
                                    flagcxDeviceMemoryOrder_t order) const {
      if (order == flagcxDeviceMemoryOrderAcqRel) {
        coop.sync();
        for (int pe = 0; pe < _dc.nRanks; ++pe)
          xshmemi_quiet<XSHMEMI_THREADGROUP_CLUSTER>(pe);
      } else {
        xshmem_fence();
      }
    }

    // ---- Wait: waitSignal ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    waitSignal(Coop coop, flagcxDevNetSignal_t signalId, uint64_t least,
               int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      if (coop.threadRank() == 0) {
        XSHMEM_FGP uint64_t *addr = _dc.signalBuffer + (int)signalId;
        xshmem_signal_wait_until(addr, XSHMEM_CMP_GE, least);
      }
    }

    // ---- Wait: waitSignalMeetShadow ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    waitSignalMeetShadow(Coop, flagcxDevNetSignal_t, int,
                         flagcxDeviceMemoryOrder_t) const {}

    // ---- Wait: waitSignalFollowShadow ----
    template <typename Coop, typename Uint>
    XSHMEM_DEVICE_INLINE void
    waitSignalFollowShadow(Coop, flagcxDevNetSignal_t, Uint, Uint *, Uint *,
                           int, flagcxDeviceMemoryOrder_t) const {}

    // ---- Shadow access ----
    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getSignalShadowPtr(flagcxDevNetSignal_t) const {
      return nullptr;
    }

    XSHMEM_DEVICE_INLINE void increaseSignalShadow(flagcxDevNetSignal_t,
                                                   uint64_t) const {}

    XSHMEM_DEVICE_INLINE uint64_t readSignal(flagcxDevNetSignal_t, int,
                                             flagcxDeviceMemoryOrder_t) const {
      return 0;
    }

    XSHMEM_DEVICE_INLINE void resetSignal(flagcxDevNetSignal_t) const {}

    // ---- Local signal write ----
    XSHMEM_DEVICE_INLINE void setSignal(flagcxDevNetSignal_t, uint64_t) const {}

    // ---- Counter interfaces ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void waitCounter(Coop, flagcxDevNetCounter_t, uint64_t,
                                          int,
                                          flagcxDeviceMemoryOrder_t) const {}

    XSHMEM_DEVICE_INLINE uint64_t readCounter(flagcxDevNetCounter_t, int,
                                              flagcxDeviceMemoryOrder_t) const {
      return 0;
    }

    XSHMEM_DEVICE_INLINE void resetCounter(flagcxDevNetCounter_t) const {}

    // ---- Collective: barrierAll ----
    XSHMEM_DEVICE_INLINE void barrierAll() const {}

    // ---- Two-sided: send/recv/term/wait ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t send(Coop, Window, size_t, size_t,
                                             flagcxDataType_t, int) const {
      return flagcxSuccess;
    }

    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t recv(Coop, Window, size_t, size_t,
                                             flagcxDataType_t, int) const {
      return flagcxSuccess;
    }

    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t term(Coop) const {
      return flagcxSuccess;
    }

    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t wait(Coop) const {
      return flagcxSuccess;
    }

    // ---- One-sided: get ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void get(Team, int, Window, size_t, Window, size_t,
                                  size_t, Coop) const {}

  private:
    // ---- Cooperative data put (all cores issue; single non-blocking call;
    static XSHMEM_DEVICE_INLINE void putData(XSHMEM_FGP float *dst,
                                             XSHMEM_FGP float *src,
                                             size_t bytes, int pe) {
      xshmemx_float_put_nbi_cluster(dst, src, bytes / sizeof(float), pe);
    }

    // ---- put dispatch ----
    template <typename RA, typename LA>
    static XSHMEM_DEVICE_INLINE void
    putImpl(const Comm &_dc, XSHMEM_FGP float *dst, XSHMEM_FGP float *src,
            size_t bytes, int pe, RA, LA) {
      putData(dst, src, bytes, pe);
    }

    // ---- signal dispatch ----
    static XSHMEM_DEVICE_INLINE void signalImpl(const Comm &_dc, int pe,
                                                flagcxDevNet_SignalAdd ra) {
      if (core_id() == 0) {
        XSHMEM_FGP uint64_t *slot = _dc.signalBuffer + (int)ra.signal;
        xshmemx_signal_op(slot, ra.value, XSHMEM_SIGNAL_SET, pe);
      }
    }
  };
};

#endif
