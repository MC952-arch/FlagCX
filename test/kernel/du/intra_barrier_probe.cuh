/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Bounded, test-only probe for the inbox used by the intra barrier.
 ************************************************************************/

#ifndef FLAGCX_TEST_DU_INTRA_BARRIER_PROBE_CUH_
#define FLAGCX_TEST_DU_INTRA_BARRIER_PROBE_CUH_

// CTA slot zero is reserved only while this probe runs. Callers must
// synchronize all ranks before snapshotting and clearing it.
static __device__ void flagcxDuProbeIntraBarrier(const flagcxDevComm &comm,
                                                  int *live) {
  const auto &dc = comm._commBase;
  int peer = threadIdx.x;
  // The device_api suite reuses this communicator across sizes. Make this
  // probe's values newer than inbox values left by earlier barrier epochs.
  uint64_t tagBase = DeviceAPI::Atomic::load(
                         &dc.epochBuffer[0], flagcxDeviceMemoryOrderAcquire) *
                     ((uint64_t)dc.intraSize + 1);
  if (peer < dc.intraSize) {
    live[peer] = 0;
    if (peer != dc.intraRank) {
      int slot = dc.intraRank * dc.nBarriers;
      DeviceAPI::Atomic::store(&dc.barrierPeers[peer][slot],
                               tagBase + (uint64_t)dc.intraRank + 1,
                               flagcxDeviceMemoryOrderRelease);
    }
  }
  __syncthreads();

  if (peer < dc.intraSize && peer != dc.intraRank) {
    int slot = peer * dc.nBarriers;
    uint64_t observed = 0;
    // A missing peer must fail the test promptly instead of consuming the
    // entire CI timeout. The production barrier remains unchanged.
    for (int iter = 0; iter < 100000; ++iter) {
      observed = DeviceAPI::Atomic::load(
          &dc.barrierPeers[dc.intraRank][slot], flagcxDeviceMemoryOrderAcquire);
      if (observed >= tagBase + (uint64_t)peer + 1)
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
      after[peer] = (int)DeviceAPI::Atomic::load(
          &dc.epochBuffer[0], flagcxDeviceMemoryOrderAcquire);
    } else {
      int slot = peer * dc.nBarriers;
      after[peer] = (int)DeviceAPI::Atomic::load(
          &dc.barrierPeers[dc.intraRank][slot], flagcxDeviceMemoryOrderAcquire);
      DeviceAPI::Atomic::store(&dc.barrierPeers[dc.intraRank][slot], uint64_t{0},
                               flagcxDeviceMemoryOrderRelaxed);
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
