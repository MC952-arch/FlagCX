/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * XSHMEM CommTraits — device-side backend using KunlunXin xshmem PGAS APIs.
 * Provides CommTraits<XshmemBackend> specialization with:
 *   - Comm, Team, Window, Multimem (data types)
 *   - Coop types (cluster-scoped, from xtdk)
 *   - Net (real implementations only for what the KLX kernels use:
 *          put / flush / signal / waitSignal; all other methods are
 *          placeholders)
 *   - localScratch helpers (per-core LM staging buffer)
 *
 * This header is the ONLY place device code touches xshmem: test kernels
 * (test/kernel/klx/device_api.xpu) must stay entirely on the Net semantic
 * surface and never include xshmem headers or call xshmem_* directly.
 *
 * XSHMEM symmetric heap: every PE maps the symmetric heap at a DIFFERENT VA
 * but with the SAME in-heap offset (remote addr = (dest - local_heap_base) +
 * remote_heap_base). The xshmem device API (xshmemx_*_put) accepts the local
 * symmetric address and performs the translation internally — so Window
 * pointers are passed as-is, exactly like the official xshmem examples.
 *
 * Compile-time model: this header ALWAYS includes the real xtdk/xshmem
 * device headers. Every translation unit that reaches it must be compiled
 * by xpu-clang with --xpu-arch=xpu3 (which defines __xpu__ on both the
 * device pass and the internal host pass); XSHMEM_FGP = __global_ptr__.
 * There are no host shims/stubs: a non-XPU compiler (plain g++) fails
 * fast at the #error guard below.
 *
 * P800 platform notes baked into this backend (validated on this node by the
 * previously direct-xshmem test kernels):
 *   - cluster put/quiet primitives are COOPERATIVE: every launched core of
 *     the cluster must issue them (the library syncs internally);
 *   - xshmemx_float_put_nbi_cluster handles full-range transfers on this
 *     xccl build, so putData issues ONE non-blocking put per call (no
 *     chunking) and does NOT quiet — completion is ordered either by the
 *     fused-signal put path (quiet-before-signal) or by Net::flush;
 *   - C2C signal is SET-only (XSHMEM_SIGNAL_SET; no atomic ADD): a shared
 *     slot cannot count multiple sources. Callers use per-source slots with
 *     monotonic stamps and wait CMP_GE on ALL of them;
 *   - XSHMEM_DEVICE_INIT declares __local__/__shared__ state in the KERNEL
 *     scope, so it must remain a macro used at kernel entry (it cannot be
 *     wrapped in a function). It is part of this header's surface.
 *
 * NOTE: do NOT #include "comm_traits.h" here — comm_traits.h includes this
 * file (via kunlunxin_comm_traits.h). Callers include device_api/comm_traits.h.
 ************************************************************************/

#ifndef FLAGCX_XSHMEM_COMM_TRAITS_H_
#define FLAGCX_XSHMEM_COMM_TRAITS_H_

#include "flagcx_kernel_core.h"
#include <cstddef>
#include <cstdint>

// ============================================================
// XSHMEM device API availability
// ============================================================
// Real xtdk/xshmem headers, unconditionally: every pass that parses this
// header must be XPU-aware (xpu-clang --xpu-arch=xpu3 defines __xpu__ on
// both the device pass and the internal host pass). No g++ shim branch.
#ifndef __xpu__
#error "xshmem_comm_traits.h requires the XPU toolchain: compile with xpu-clang --xpu-arch=xpu3 (host pass included)"
#endif
#include "xpu/kernel/xtdk.h" // cluster_id/cluster_num/core_id
#include "xshmem/xshmem.h"
#include "xshmem/xshmemx.h"
// xshmem RMA/signal functions expect __global_ptr__-qualified pointers on
// xpu3; unqualified generic pointers can fault (kl3 status 700). Cast at
// call sites via XSHMEM_FGP.
#define XSHMEM_FGP __global_ptr__

// Device inline qualifier. The framework's FLAGCX_DEVICE_INLINE_DECORATOR is
// EMPTY on non-NVIDIA/DU platforms (device_utils.h), which would compile
// every Net/Window/Comm method as a host function under xpu-clang and fail
// with "call to __device__ function from __host__ function" when they touch
// xshmem device primitives. __device__ is always valid here: this header is
// only ever parsed by xpu-clang (see the #error guard above).
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
  // Wraps the xshmem-internal per-core LM buffer registered by
  // XSHMEM_DEVICE_INIT, so kernels can stage GM2LM/LM2GM tiles without
  // touching xshmem names directly.
  static XSHMEM_DEVICE_INLINE float *localScratch() {
    // Plain (LM-space) pointer: the buffer is per-core local memory, NOT a
    // global-space symmetric buffer -- do not qualify with XSHMEM_FGP.
    return (float *)get_xshmemi_local_buf();
  }
  static XSHMEM_DEVICE_INLINE int localScratchBytes() {
    return XSHMEMI_LOCAL_BUF_LEN;
  }

  // ---- Window ----
  // Pointer fields are XSHMEM_FGP (__global_ptr__ on device, plain void*
  // on host): these buffers live in the symmetric heap / device DRAM
  // (global space).
  struct Window {
    XSHMEM_FGP void *symBase; // symmetric buffer base (same in-heap offset on
                              // every PE)
    size_t allocSize;
    XSHMEM_FGP void *rawPtr; // local pointer (= symBase for symmetric buffers)

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
  // Field layout mirrors flagcxShmemCommInternal plus the device state
  // handle. Test kernels populate it field-by-field from launch params.
  struct Comm {
    int rank, nRanks;
    int intraRank, intraSize;
    int intraTeam; // placeholders (xshmem has no teams)
    int interTeam;
    int worldTeam;

    XSHMEM_FGP uint64_t *signalBuffer;
    int signalCount;
    XSHMEM_FGP uint64_t *counterBuffer;
    int counterCount;
    XSHMEM_FGP uint64_t *shadowBuffer;

    XSHMEM_FGP uint64_t *gridSyncState; // nullptr (P800: no device barrier)

    XSHMEM_FGP void *devStateHandle; // xshmem device state (XSHMEM_DEVICE_INIT
                                     // arg)

    XSHMEM_DEVICE_INLINE int getIntraRank() const { return 0; }
    XSHMEM_DEVICE_INLINE int getIntraSize() const { return 0; }
    XSHMEM_DEVICE_INLINE int getRank() const { return 0; }
    XSHMEM_DEVICE_INLINE int getSize() const { return 0; }
    XSHMEM_DEVICE_INLINE void *getFifoBuffer(int) const { return nullptr; }
    XSHMEM_DEVICE_INLINE Multimem getMulticastHandle() const {
      Multimem mm;
      mm.mcBasePtr = nullptr; // XSHMEM doesn't use multicast
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
  // Placeholder-only: P800 has no usable device barrier under
  // XSHMEM_DEVICE_INIT; kernel barriers must use per-source signals.
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
    // static (data passed in): on XPU, a __device__ non-static member
    // calling another non-static member trips a `this` address-space error;
    // private helpers are static and take data explicitly.
    static XSHMEM_DEVICE_INLINE int resolvePE(const Comm &dc, Team team,
                                              int peer) {
      int base = dc.rank - team.rank * team.stride;
      return base + peer * team.stride;
    }

    // ---- One-sided: put ----
    // XSHMEM cluster primitives are COOPERATIVE: every core of the cluster
    // must issue the call so the transfer is partitioned across cores. The
    // put is NON-BLOCKING; completion is enforced by quiet:
    //   - RA = SignalInc/SignalAdd: quiet(pe) THEN the signal (fused
    //     put+signal — the receiver's wait implies the data landed);
    //   - RA = None: no quiet — the caller overlaps several puts and calls
    //     flush() once (matches the proven multi-peer pipeline).
    template <typename RA, typename LA, typename Coop, typename Desc>
    XSHMEM_DEVICE_INLINE void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      (void)coop;
      // NO extra threadgroup_sync here: cluster put primitives are
      // cooperative with their own internal barriers, and inserting extra
      // syncs between consecutive puts / before signal_op perturbs the
      // xshmem flow-control window (observed on P800: the fused stamp of a
      // later signal_op is intermittently DROPPED and the target rank spins
      // forever in waitSignal). Mirror the proven direct-xshmem sequence
      // exactly: back-to-back puts, then flush(), then signal().
      int pe = resolvePE(_dc, team, peer);
      putImpl(_dc,
              (XSHMEM_FGP float *)((XSHMEM_FGP char *)dst.symBase + dstOff),
              (XSHMEM_FGP float *)((XSHMEM_FGP char *)src.rawPtr + srcOff),
              bytes, pe, ra, la);
    }

    // ---- One-sided: putValue ----
    template <typename T, typename RA, typename Coop, typename Desc>
    XSHMEM_DEVICE_INLINE void
    putValue(Team, int, Window, size_t, T, RA, Coop, Desc, flagcxDeviceScope_t,
             flagcxDeviceScope_t) const {}

    // ---- One-sided: signal ----
    template <typename RA, typename Coop, typename Desc>
    XSHMEM_DEVICE_INLINE void
    signal(Team team, int peer, RA ra, Coop coop, Desc desc,
           flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      (void)coop;
      // core0-gated signal_op, no surrounding syncs (see put() note); the
      // caller inserts one coop.sync() after its signal loop, mirroring the
      // proven direct-xshmem sequence.
      int pe = resolvePE(_dc, team, peer);
      signalImpl(_dc, pe, ra);
    }

    // ---- Ordering: flush ----
    // AcqRel: drain ALL my outstanding puts (per-peer cooperative quiet over
    // every PE — mirrors the proven kernels' post-put quiet loop). Anything
    // weaker: a fence.
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    flush(Coop coop, flagcxDeviceMemoryOrder_t order) const {
      if (order == flagcxDeviceMemoryOrderAcqRel) {
        // sync once after the caller's non-blocking put loop, then quiet
        // every PE (cooperative, own internal barriers). No trailing sync:
        // this mirrors the proven direct-xshmem sequence 1:1 (extra syncs
        // here perturb the flow-control window, see put()).
        coop.sync();
        for (int pe = 0; pe < _dc.nRanks; ++pe)
          xshmemi_quiet<XSHMEMI_THREADGROUP_CLUSTER>(pe);
      } else {
        xshmem_fence();
      }
    }

    // ---- Wait: waitSignal ----
    // CMP_GE: stamps are monotonic (iteration counters), so a stale smaller
    // value can never satisfy the wait and slots never need resetting.
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    waitSignal(Coop coop, flagcxDevNetSignal_t signalId, uint64_t least,
               int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      // core0-gated wait, no surrounding syncs (see put() note); the caller
      // inserts one coop.sync() after its wait loop, mirroring the proven
      // direct-xshmem sequence.
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

    XSHMEM_DEVICE_INLINE void
    increaseSignalShadow(flagcxDevNetSignal_t, uint64_t) const {}

    XSHMEM_DEVICE_INLINE uint64_t
    readSignal(flagcxDevNetSignal_t, int, flagcxDeviceMemoryOrder_t) const {
      return 0;
    }

    XSHMEM_DEVICE_INLINE void resetSignal(flagcxDevNetSignal_t) const {}

    // ---- Local signal write ----
    XSHMEM_DEVICE_INLINE void setSignal(flagcxDevNetSignal_t, uint64_t) const {}

    // ---- Counter interfaces ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    waitCounter(Coop, flagcxDevNetCounter_t, uint64_t, int,
                flagcxDeviceMemoryOrder_t) const {}

    XSHMEM_DEVICE_INLINE uint64_t
    readCounter(flagcxDevNetCounter_t, int, flagcxDeviceMemoryOrder_t) const {
      return 0;
    }

    XSHMEM_DEVICE_INLINE void resetCounter(flagcxDevNetCounter_t) const {}

    // ---- Collective: barrierAll ----
    XSHMEM_DEVICE_INLINE void barrierAll() const {}

    // ---- Two-sided: send/recv/term/wait ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t
    send(Coop, Window, size_t, size_t, flagcxDataType_t, int) const {
      return flagcxSuccess;
    }

    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t
    recv(Coop, Window, size_t, size_t, flagcxDataType_t, int) const {
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
    XSHMEM_DEVICE_INLINE void
    get(Team, int, Window, size_t, Window, size_t, size_t, Coop) const {}

  private:
    // ---- Cooperative data put (all cores issue; single non-blocking call;
    // NO quiet — see put() contract above) ----
    static XSHMEM_DEVICE_INLINE void
    putData(XSHMEM_FGP float *dst, XSHMEM_FGP float *src, size_t bytes,
            int pe) {
      // bytes is a multiple of sizeof(float) (Device API data puts carry
      // float payloads). This xccl build's put_nbi_cluster handles the full
      // range in one call (validated on this node up to multi-MB segments).
      xshmemx_float_put_nbi_cluster(dst, src, bytes / sizeof(float), pe);
    }

    // ---- put dispatch ----
    template <typename RA, typename LA>
    static XSHMEM_DEVICE_INLINE void
    putImpl(const Comm &_dc, XSHMEM_FGP float *dst, XSHMEM_FGP float *src,
            size_t bytes, int pe, RA, LA) {
      // RA = flagcxDevNet_None: non-blocking put only; the caller overlaps
      // several puts and completes them with one flush().
      putData(dst, src, bytes, pe);
    }

    // ---- signal dispatch ----
    // P800 C2C signal is SET-only (no atomic ADD): a shared slot cannot
    // count multiple sources — callers use per-source slots and wait on ALL
    // of them. SignalAdd therefore delivers "SET value" (the monotonic
    // stamp). xshmemx_signal_op is issued from core 0 only (it is not a
    // cooperative primitive), mirroring the proven direct-xshmem kernels on
    // this node.
    static XSHMEM_DEVICE_INLINE void
    signalImpl(const Comm &_dc, int pe, flagcxDevNet_SignalAdd ra) {
      if (core_id() == 0) {
        XSHMEM_FGP uint64_t *slot = _dc.signalBuffer + (int)ra.signal;
        xshmemx_signal_op(slot, ra.value, XSHMEM_SIGNAL_SET, pe);
      }
    }
  }; // struct Net
};   // struct CommTraits<XshmemBackend>

#endif // FLAGCX_XSHMEM_COMM_TRAITS_H_
