/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Bounded, test-only probe for the IPC inbox used by the intra barrier.
 ************************************************************************/

#ifndef FLAGCX_TEST_DU_INTRA_BARRIER_PROBE_CUH_
#define FLAGCX_TEST_DU_INTRA_BARRIER_PROBE_CUH_

#include <cuda/atomic>

// CTA slot zero is reserved only while this probe runs. Callers must
// synchronize all ranks before snapshotting and clearing it.
static __device__ void flagcxDuProbeIntraBarrier(const flagcxDevComm &comm,
                                                  int *live) {
  const auto &dc = comm._commBase;
  int peer = threadIdx.x;
  if (peer < dc.intraSize) {
    live[peer] = 0;
    if (peer != dc.intraRank) {
      int slot = dc.intraRank * dc.nBarriers;
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> remote(
          dc.barrierPeers[peer][slot]);
      remote.store((uint64_t)dc.intraRank + 1, cuda::memory_order_release);
    }
  }
  __syncthreads();

  if (peer < dc.intraSize && peer != dc.intraRank) {
    int slot = peer * dc.nBarriers;
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> local(
        dc.barrierPeers[dc.intraRank][slot]);
    uint64_t observed = 0;
    // A missing peer must fail the test promptly instead of consuming the
    // entire CI timeout. The production barrier remains unchanged.
    for (int iter = 0; iter < 100000; ++iter) {
      observed = local.load(cuda::memory_order_acquire);
      if (observed >= (uint64_t)peer + 1)
        break;
      PlatformTraits<DuPlatform>::Intrin::spinBackoff(iter);
    }
    live[peer] = (int)observed;
  }
}

static __device__ void flagcxDuSnapshotIntraBarrier(const flagcxDevComm &comm,
                                                     int *after) {
  const auto &dc = comm._commBase;
  int peer = threadIdx.x;
  if (peer < dc.intraSize) {
    after[peer] = 0;
    if (peer == dc.intraRank) {
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> epoch(
          dc.epochBuffer[0]);
      after[peer] = (int)epoch.load(cuda::memory_order_acquire);
    } else {
      int slot = peer * dc.nBarriers;
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system> local(
          dc.barrierPeers[dc.intraRank][slot]);
      after[peer] = (int)local.load(cuda::memory_order_acquire);
      local.store(0, cuda::memory_order_relaxed);
    }
  }
}

static __global__ void flagcxDuIntraBarrierProbeByValue(flagcxDevComm comm,
                                                        int *live) {
  flagcxDuProbeIntraBarrier(comm, live);
}

static __global__ void flagcxDuIntraBarrierSnapshotByValue(flagcxDevComm comm,
                                                           int *after) {
  flagcxDuSnapshotIntraBarrier(comm, after);
}

static __global__ void flagcxDuIntraBarrierProbeByPointer(
    const void *commPtr, int *live) {
  flagcxDuProbeIntraBarrier(*static_cast<const flagcxDevComm *>(commPtr), live);
}

static __global__ void flagcxDuIntraBarrierSnapshotByPointer(
    const void *commPtr, int *after) {
  flagcxDuSnapshotIntraBarrier(*static_cast<const flagcxDevComm *>(commPtr),
                               after);
}

#endif // FLAGCX_TEST_DU_INTRA_BARRIER_PROBE_CUH_
