/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 * See LICENSE-NCCL.txt for NCCL license information
 *
 * NVIDIA Vendor Device Traits — wraps NCCL device API types.
 *
 * DeviceTraits<NvidiaVendor> provides:
 *   - Intrin, Atomic: from PlatformTraits<NvidiaPlatform> via using
 *   - Window:   wraps ncclWindow_t with member functions
 *   - DevComm:  wraps ncclDevComm with member functions
 *   - Team:     wraps ncclTeam_t with member functions
 *   - Multimem: wraps ncclMultimemHandle_t
 *
 * Also defines FLAGCX_DEVICE_API_VENDOR and the DeviceAPI selection.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_DEVICE_TRAITS_H_
#define FLAGCX_NVIDIA_DEVICE_TRAITS_H_

#include "nccl.h"

// ============================================================
// NVIDIA Vendor Backend (NCCL device API)
// ============================================================
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0) &&                              \
    !defined(FLAGCX_FORCE_FALLBACK)

#include "nccl_device.h"

struct NvidiaVendor {};

template <>
struct DeviceTraits<NvidiaVendor> {
  // Platform capabilities (via using, not inheritance)
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;

  // ---- Team: wraps ncclTeam_t ----
  // Exposes nRanks/rank/stride for direct field access (used by flagcxTeam)
  struct Team {
    int nRanks, rank, stride;

    FLAGCX_HOST_DEVICE_INLINE Team() : nRanks(0), rank(0), stride(0) {}
    FLAGCX_HOST_DEVICE_INLINE Team(int nr, int r, int s)
        : nRanks(nr), rank(r), stride(s) {}

    // Implicit conversion to ncclTeam_t for NCCL API calls
    FLAGCX_HOST_DEVICE_INLINE operator ncclTeam_t() const {
      ncclTeam_t t;
      t.nRanks = nRanks;
      t.rank = rank;
      t.stride = stride;
      return t;
    }
  };

  // ---- Multimem: wraps ncclMultimemHandle_t ----
  struct Multimem {
    ncclMultimemHandle_t _impl;

    FLAGCX_HOST_DEVICE_INLINE Multimem() : _impl() {}

    // Implicit conversion for NCCL API calls
    FLAGCX_HOST_DEVICE_INLINE operator ncclMultimemHandle_t() const {
      return _impl;
    }
  };

  // ---- Window: wraps ncclWindow_t ----
  struct Window {
    ncclWindow_t _impl;

    FLAGCX_HOST_DEVICE_INLINE Window() : _impl() {}

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      return ncclGetPeerPointer(_impl, offset, (ncclTeam_t)team, peer);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      return ncclGetLocalPointer(_impl, offset);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      return ncclGetLsaPointer(_impl, offset, peer);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getMulticastPointer(size_t offset, const Multimem &mm) const {
      return ncclGetMultimemPointer(_impl, offset, mm._impl);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool operator==(const Window &o) const {
      return _impl.base == o._impl.base && _impl.size == o._impl.size;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator!=(const Window &o) const {
      return !(*this == o);
    }
  };

  // ---- DevComm: wraps ncclDevComm ----
  struct DevComm {
    ncclDevComm _impl;

    FLAGCX_HOST_DEVICE_INLINE DevComm() : _impl() {}

    // Implicit conversion to ncclDevComm for NCCL API calls
    FLAGCX_HOST_DEVICE_INLINE operator const ncclDevComm &() const {
      return _impl;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
      return _impl.lsaRank;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
      return _impl.lsaSize;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return _impl.rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return _impl.nRanks; }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer() const {
      return nullptr;
    }
  };

  // ---- CoopBlock: wraps ncclCoopCta ----
  struct CoopBlock {
    ncclCoopCta _impl;

    FLAGCX_HOST_DEVICE_INLINE CoopBlock() : _impl() {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }

    FLAGCX_HOST_DEVICE_INLINE operator ncclCoopCta() const { return _impl; }
  };

  // ---- CoopTile<N>: wraps ncclCoopTile<N> ----
  template <int N>
  struct CoopTile {
    ncclCoopTile<N> _impl;

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
      return _impl.laneMask();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }

    FLAGCX_HOST_DEVICE_INLINE operator ncclCoopTile<N>() const { return _impl; }
  };

  using CoopThread = CoopTile<1>;
  using CoopWarp = CoopTile<32>;

  // ---- CoopTileSpan: wraps ncclCoopWarpSpan ----
  struct CoopTileSpan {
    ncclCoopWarpSpan _impl;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopTileSpan(int t0, int nTiles, int id)
        : _impl(t0, nTiles, id) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }

    FLAGCX_HOST_DEVICE_INLINE operator ncclCoopWarpSpan() const {
      return _impl;
    }
  };

  // ---- CoopLanes: wraps ncclCoopLanes ----
  struct CoopLanes {
    ncclCoopLanes _impl;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopLanes(uint32_t lmask = 0xffffffffu)
        : _impl{lmask} {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t getLmask() const {
      return _impl.lmask;
    }

    FLAGCX_HOST_DEVICE_INLINE operator ncclCoopLanes() const { return _impl; }
  };

  // ---- CoopAny: wraps ncclCoopAny ----
  struct CoopAny {
    ncclCoopAny _impl;

    CoopAny() = default;
    CoopAny(CoopAny const &) = default;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopBlock b) : _impl(b._impl) {}
    template <int N>
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopTile<N> t) : _impl(t._impl) {}
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopTileSpan s) : _impl(s._impl) {}
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopLanes l) : _impl(l._impl) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }

    FLAGCX_HOST_DEVICE_INLINE operator ncclCoopAny() const { return _impl; }
  };

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    ncclLsaBarrierHandle _impl;
  };
  struct InterBarrierHandle {
    ncclGinBarrierHandle _impl;
  };

  // ---- Barrier / GIN type aliases ----
  using Barrier = ncclLsaBarrier;
  using RemoteAction = ncclGinRemoteAction;
  using LocalAction = ncclGinLocalAction;
  using FenceLevel = ncclGinFenceLevel;
};

#define FLAGCX_DEVICE_API_VENDOR 1
using DeviceAPI = DeviceTraits<NvidiaVendor>;

#else
// ============================================================
// NVIDIA Fallback Backend (IPC barriers + FIFO one-sided)
// Uses common Fallback<> partial specialization with NVIDIA platform
// ============================================================
using DeviceAPI = DeviceTraits<Fallback<NvidiaPlatform>>;

#endif // NCCL version check

#endif // FLAGCX_NVIDIA_DEVICE_TRAITS_H_
