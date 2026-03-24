/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Fallback Device Traits — Common IPC-based implementation.
 *
 * DeviceTraits<Fallback<PlatformTag>> provides:
 *   - Intrin, Atomic: inherited from PlatformTraits<PlatformTag> via using
 *   - Window:   IPC peer pointers + raw pointer
 *   - DevComm:  rank/size + IPC barriers + signal buffers
 *   - Team:     pure arithmetic (nRanks, rank, stride)
 *   - Multimem: placeholder (no multicast)
 *
 * This partial specialization is written ONCE and works for any platform.
 * Adding a new platform (e.g. Cambricon) requires zero changes here.
 ************************************************************************/

#ifndef FLAGCX_FALLBACK_DEVICE_TRAITS_H_
#define FLAGCX_FALLBACK_DEVICE_TRAITS_H_

template <typename PlatformTag>
struct DeviceTraits<Fallback<PlatformTag>> {
  // Platform capabilities (resolved via PlatformTag)
  using Intrin = typename PlatformTraits<PlatformTag>::Intrin;
  using Atomic = typename PlatformTraits<PlatformTag>::Atomic;

  // ---- Team: Pure arithmetic ----
  struct Team {
    int nRanks, rank, stride;
  };

  // ---- Multimem: Placeholder ----
  struct Multimem {
    void *mcBasePtr;
  };

  // ---- Window: IPC + MR + rawPtr ----
  struct Window {
    void *rawPtr;     // Raw memory pointer (always valid)
    void **peerPtrs;  // IPC peer pointers (nullptr if no IPC)
    int intraRank;    // Local rank index
    uintptr_t mrBase; // MR base VA
    int mrIndex;      // MR table index

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      if (peerPtrs) {
        int index = team.rank + (peer - team.rank) * team.stride;
        return (char *)peerPtrs[index] + offset;
      }
      return nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      if (peerPtrs)
        return (char *)peerPtrs[intraRank] + offset;
      return (char *)rawPtr + offset;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      if (peerPtrs)
        return (char *)peerPtrs[peer] + offset;
      return nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getMulticastPointer(size_t offset, const Multimem &mm) const {
      (void)offset;
      (void)mm;
      return nullptr; // Multicast not available in fallback
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool operator==(const Window &o) const {
      return rawPtr == o.rawPtr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator!=(const Window &o) const {
      return !(*this == o);
    }
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

    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
      return intraRank;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
      return intraSize;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return nRanks; }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer() const {
      return fifoBuffer;
    }
  };

  // ---- CoopBlock: CTA-level cooperative group ----
  struct CoopBlock {
#ifdef FLAGCX_SIMT_WIDTH
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return FLAGCX_THREAD_IDX_X;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return FLAGCX_BLOCK_DIM_X;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { FLAGCX_DEVICE_SYNC_THREADS(); }
#else
    int threadRank() const { return FLAGCX_THREAD_IDX_X; }
    int size() const { return FLAGCX_BLOCK_DIM_X; }
    void sync() {}
#endif
  };

  // ---- CoopTile<N>: Tile of N threads ----
  template <int N>
  struct CoopTile {
#ifdef FLAGCX_SIMT_WIDTH
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= FLAGCX_SIMT_WIDTH,
                  "N must be a power of 2 and <= FLAGCX_SIMT_WIDTH");

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::lane() % N;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
      return (0xffffffffu >> (32 - N)) << (Intrin::lane() & -N);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      if (N > 1)
        Intrin::syncwarp(laneMask());
    }
#else
    int threadRank() const { return 0; }
    int size() const { return N; }
    void sync() {}
#endif
  };

  using CoopThread = CoopTile<1>;

#ifdef FLAGCX_SIMT_WIDTH
  using CoopWarp = CoopTile<FLAGCX_SIMT_WIDTH>;

  // ---- CoopTileSpan: consecutive tiles with named barrier ----
  struct CoopTileSpan {
    uint32_t t0 : 8, nTiles : 8, id : 8;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopTileSpan(int t0, int nTiles, int id)
        : t0(t0), nTiles(nTiles), id(id) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return FLAGCX_THREAD_IDX_X - FLAGCX_SIMT_WIDTH * t0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return FLAGCX_SIMT_WIDTH * nTiles;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      Intrin::namedBarrierSync(1 + id, FLAGCX_SIMT_WIDTH * nTiles);
    }
  };

  // ---- CoopLanes: arbitrary lane bitmask ----
  struct CoopLanes {
    uint32_t lmask;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopLanes(uint32_t lmask = 0xffffffffu)
        : lmask(lmask) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::popc(lmask & Intrin::lanemaskLt());
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return Intrin::popc(lmask);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { Intrin::syncwarp(lmask); }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t getLmask() const { return lmask; }
  };
#endif // FLAGCX_SIMT_WIDTH

  // ---- CoopAny: type-erased cooperative group ----
  struct CoopAny {
#ifdef FLAGCX_SIMT_WIDTH
    // SIMT: vtable-based type erasure
    struct Storage {
      alignas(alignof(void *)) char space[16];
    };
    struct VTable {
      int (*threadRank)(void const *);
      int (*size)(void const *);
      void (*sync)(void *);
    };

    template <typename Impl>
    FLAGCX_DEVICE_INLINE_DECORATOR static int threadRank_fn(void const *o) {
      return static_cast<Impl const *>(o)->threadRank();
    }
    template <typename Impl>
    FLAGCX_DEVICE_INLINE_DECORATOR static int size_fn(void const *o) {
      return static_cast<Impl const *>(o)->size();
    }
    template <typename Impl>
    FLAGCX_DEVICE_INLINE_DECORATOR static void sync_fn(void *o) {
      static_cast<Impl *>(o)->sync();
    }

    template <typename Impl>
    FLAGCX_DEVICE_INLINE_DECORATOR static VTable const *get_vtable() {
      static_assert(sizeof(Impl) <= sizeof(Storage), "Coop type too large");
      static_assert(alignof(Impl) <= alignof(Storage),
                    "Coop type alignment too large");
      static constexpr VTable v = {&threadRank_fn<Impl>, &size_fn<Impl>,
                                   &sync_fn<Impl>};
      return &v;
    }

    Storage storage;
    VTable const *vtable;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny()
        : storage{}, vtable(get_vtable<CoopThread>()) {}
    CoopAny(CoopAny const &) = default;

    template <typename Impl>
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(Impl impl) {
      char const *src = reinterpret_cast<char const *>(&impl);
      for (unsigned i = 0; i < sizeof(Impl); ++i)
        this->storage.space[i] = src[i];
      this->vtable = get_vtable<Impl>();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return vtable->threadRank(&storage);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return vtable->size(&storage);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { vtable->sync(&storage); }
#else
    // Non-SIMT: simple capture
    int _threadRank;
    int _size;

    CoopAny() : _threadRank(0), _size(1) {}
    CoopAny(CoopBlock b) : _threadRank(b.threadRank()), _size(b.size()) {}
    template <int N>
    CoopAny(CoopTile<N>) : _threadRank(0), _size(N) {}

    int threadRank() const { return _threadRank; }
    int size() const { return _size; }
    void sync() {}
#endif
  };

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    int nBarriers;
  };
  struct InterBarrierHandle {
    int placeholder;
  };
};

#endif // FLAGCX_FALLBACK_DEVICE_TRAITS_H_
