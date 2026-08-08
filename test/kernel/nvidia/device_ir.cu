/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR kernel implementations — CUDA kernels exercising FlagCX
 * Device API IR functions via device pointers.
 *
 * Intra-node (S1–S10): aligned with device_api_intra K1–K10.
 * Inter-node transport: separate section.
 *
 * Compiled by nvcc into device_ir.o, linked by g++ into test_device_ir.
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "nvidia_adaptor.h"
#include "flagcx_device_internal.h"

// IR wrapper declarations + implementations (needed for nvcc inline compilation)
#include "flagcx_device_wrapper.h"
#include "flagcx_device_wrapper_impl.h" // also pulls in scalar_ir_impl.h

#include "device_ir.h"

// ===========================================================================
// Scalar IR (S-suffixed) kernels — Intra-Node (S1–S10)
// ===========================================================================

// ---------------------------------------------------------------------------
// S1: Comm Queries (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelCommQueriesS(const void *devCommPtr, int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    results[0] = flagcxDevCommGetRank(devCommPtr);
    results[1] = flagcxDevCommGetSize(devCommPtr);
    results[2] = flagcxDevCommGetIntraRank(devCommPtr);
    results[3] = flagcxDevCommGetIntraSize(devCommPtr);
  }
}

void launchKernelCommQueriesS(const void *devCommPtr, int *devResults,
                              flagcxStream_t stream) {
  kernelCommQueriesS<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// S2: Coop Groups (Scalar) — block, tile_span, lanes in one kernel
// ---------------------------------------------------------------------------

// Sub-kernel: block-level coop check (1 block, 32 threads)
__global__ void kernelCoopGroupsS_block(int *results) {
  int rank = flagcxCoopThreadRankS(FLAGCX_COOP_BLOCK);
  int size = flagcxCoopSizeS(FLAGCX_COOP_BLOCK);
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Thread 0 checks all threads got correct rank/size
  __shared__ int pass;
  if (FLAGCX_THREAD_IDX_X == 0) pass = 1;
  __syncthreads();
  if (rank != (int)FLAGCX_THREAD_IDX_X || size != (int)FLAGCX_BLOCK_DIM_X)
    atomicExch(&pass, 0);
  __syncthreads();
  if (FLAGCX_THREAD_IDX_X == 0) results[0] = pass;
}

// Sub-kernel: tile_span coop check (1 block, 128 threads = 4 tiles of 32)
__global__ void kernelCoopGroupsS_tileSpan(int *results) {
  int tileIdx = FLAGCX_THREAD_IDX_X / 32;
  uint32_t t0 = (uint32_t)tileIdx;
  uint32_t nTiles = 1;
  uint32_t id = 0;

  int rank = flagcxCoopThreadRankExS(FLAGCX_COOP_TILE_SPAN, t0, nTiles, id);
  int size = flagcxCoopSizeExS(FLAGCX_COOP_TILE_SPAN, t0, nTiles, id);
  flagcxCoopSyncExS(FLAGCX_COOP_TILE_SPAN, t0, nTiles, id);

  // Expected: rank = threadIdx % 32, size = 32
  __shared__ int pass;
  if (FLAGCX_THREAD_IDX_X == 0) pass = 1;
  __syncthreads();
  if (rank != (int)(FLAGCX_THREAD_IDX_X % 32) || size != 32)
    atomicExch(&pass, 0);
  __syncthreads();
  if (FLAGCX_THREAD_IDX_X == 0) results[1] = pass;
}

// Sub-kernel: lanes coop check (1 block, 32 threads, full warp mask)
__global__ void kernelCoopGroupsS_lanes(int *results) {
  uint32_t laneMask = 0xFFFFFFFF;

  int rank = flagcxCoopThreadRankExS(FLAGCX_COOP_LANES, laneMask, 0, 0);
  int size = flagcxCoopSizeExS(FLAGCX_COOP_LANES, laneMask, 0, 0);
  flagcxCoopSyncExS(FLAGCX_COOP_LANES, laneMask, 0, 0);

  // Expected: rank = lane index, size = 32
  __shared__ int pass;
  if (FLAGCX_THREAD_IDX_X == 0) pass = 1;
  __syncthreads();
  if (rank != (int)FLAGCX_THREAD_IDX_X || size != 32)
    atomicExch(&pass, 0);
  __syncthreads();
  if (FLAGCX_THREAD_IDX_X == 0) results[2] = pass;
}

void launchKernelCoopGroupsS(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream) {
  kernelCoopGroupsS_block<<<1, 32, 0, stream->base>>>(devResults);
  kernelCoopGroupsS_tileSpan<<<1, 128, 0, stream->base>>>(devResults);
  kernelCoopGroupsS_lanes<<<1, 32, 0, stream->base>>>(devResults);
}

// ---------------------------------------------------------------------------
// S3: Team Queries (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelTeamQueriesS(const void *devCommPtr, int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
    int worldRank =
        flagcxTeamRankToWorldS(devCommPtr, FLAGCX_TEAM_INTRA, intraRank);

    results[0] = intraRank;
    results[1] = worldRank;
  }
}

void launchKernelTeamQueriesS(const void *devCommPtr, int *devResults,
                                   flagcxStream_t stream) {
  kernelTeamQueriesS<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// S4: Local Pointer (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelLocalPointerS(const void *devMemPtr, void *rawBuff,
                                         int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    void *localPtr = flagcxGetLocalPointerS(devMemPtr, 0);
    // Verify local pointer is non-null and points to same data as rawBuff
    // (may be a different VA due to VMM flat-mapping)
    if (localPtr == nullptr) {
      results[0] = 0;
    } else {
      float val = *((volatile float *)localPtr);
      float expected = *((volatile float *)rawBuff);
      results[0] = (val == expected) ? 1 : 0;
    }
  }
}

void launchKernelLocalPointerS(const void *devMemPtr, void *rawBuff,
                                    int *devResults, flagcxStream_t stream) {
  kernelLocalPointerS<<<1, 1, 0, stream->base>>>(devMemPtr, rawBuff,
                                                       devResults);
}

// ---------------------------------------------------------------------------
// S5: Intra Pointer (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelIntraPointerS(const void *devCommPtr,
                                    const void *devMemPtr,
                                    float *output, int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;

  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  for (int i = tid; i < count; i += nthreads) {
    size_t offset = i * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
    output[i] = *peerPtr;
  }
}

void launchKernelIntraPointerS(const void *devCommPtr,
                                    const void *devMemPtr, float *devOutput,
                                    int count,
                                    flagcxStream_t stream) {
  kernelIntraPointerS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, devOutput, count);
}

// ---------------------------------------------------------------------------
// S8: Intra Barrier Sync (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelIntraBarrierSyncS(const void *devCommPtr,
                                        const void *devMemPtr,
                                        float *buffer, float *output,
                                        int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (int i = tid; i < count; i += nthreads) {
    buffer[i] = (float)(myRank + 1);
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcqRel);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  for (int i = tid; i < count; i += nthreads) {
    size_t offset = i * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
    output[i] = *peerPtr;
  }
}

void launchKernelIntraBarrierSyncS(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream) {
  kernelIntraBarrierSyncS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, buffer, output, N);
}

// ---------------------------------------------------------------------------
// S9: Intra Barrier Sync Split (Release + read + Acquire)
// ---------------------------------------------------------------------------

__global__ void kernelIntraBarrierArriveWaitS(const void *devCommPtr,
                                              const void *devMemPtr,
                                              float *buffer, float *output,
                                              int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (int i = tid; i < count; i += nthreads) {
    buffer[i] = (float)(myRank + 500);
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelease);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  for (int i = tid; i < count; i += nthreads) {
    size_t offset = i * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
    output[i] = *peerPtr;
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcquire);
}

void launchKernelIntraBarrierArriveWaitS(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream) {
  kernelIntraBarrierArriveWaitS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, buffer, output, N);
}

// ---------------------------------------------------------------------------
// S6: Peer Pointer (Scalar) — team-based peer memory access
// ---------------------------------------------------------------------------

__global__ void kernelPeerPointerS(const void *devCommPtr,
                                   const void *devMemPtr,
                                   float *output, int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;

  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  for (int i = tid; i < count; i += nthreads) {
    size_t offset = i * sizeof(float);
    float *peerPtr = (float *)flagcxGetPeerPointerS(
        devMemPtr, offset, devCommPtr, FLAGCX_TEAM_INTRA, peer);
    output[i] = *peerPtr;
  }
}

void launchKernelPeerPointerS(const void *devCommPtr,
                              const void *devMemPtr, float *devOutput,
                              int count,
                              flagcxStream_t stream) {
  kernelPeerPointerS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, devOutput, count);
}

// ---------------------------------------------------------------------------
// S10: Intra AllReduce (Scalar) — composite using barriers + pointers
// ---------------------------------------------------------------------------

__global__ void kernelIntraAllReduceS(const void *devCommPtr,
                                           const void *devMemPtr,
                                           float *buffer, int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);

  // Cooperative indexing: partition elements across all ranks so each element
  // is processed by exactly one rank (eliminates cross-GPU race).
  int localNthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  int globalTid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_DIM_X * (myRank + FLAGCX_BLOCK_IDX_X * nRanks);
  int globalNthreads = localNthreads * nRanks;

  // Pre-reduce barrier (acquire — ensure peer writes are visible)
  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcquire);

  // Reduce + write: each rank handles a disjoint subset of elements,
  // reads from all peers, writes result to all peers.
  for (int i = globalTid; i < count; i += globalNthreads) {
    float sum = 0.0f;
    for (int peer = 0; peer < nRanks; peer++) {
      size_t offset = i * sizeof(float);
      float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
      sum += *peerPtr;
    }
    for (int peer = 0; peer < nRanks; peer++) {
      size_t offset = i * sizeof(float);
      float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
      *peerPtr = sum;
    }
  }

  // Post-reduce barrier (release — ensure writes are visible)
  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelease);
}

void launchKernelIntraAllReduceS(const void *devCommPtr,
                                  const void *devMemPtr, float *buffer,
                                  int count, flagcxStream_t stream) {
  kernelIntraAllReduceS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, buffer, count);
}

// ---------------------------------------------------------------------------
// S7: Multicast Pointer (Scalar) — NVLS-dependent, commented out
// ---------------------------------------------------------------------------

// __global__ void kernelScalarMulticastPointer(const void *devCommPtr,
//                                              const void *devMemPtr,
//                                              float *output, int nElems) {
//   int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
//   if (tid < nElems) {
//     size_t offset = tid * sizeof(float);
//     float *mcPtr = (float *)flagcxGetMulticastPointerS(
//         devMemPtr, offset, devCommPtr);
//     output[tid] = *mcPtr;
//   }
// }
//
// void launchKernelMulticastPointerS(const void *devCommPtr,
//                                    const void *devMemPtr, float *devOutput,
//                                    int nBlocks, int nThreads,
//                                    flagcxStream_t stream) {
//   int nElems = nBlocks * nThreads;
//   kernelScalarMulticastPointer<<<nBlocks, nThreads, 0, stream->base>>>(
//       devCommPtr, devMemPtr, devOutput, nElems);
// }

// ===========================================================================
// Inter-Node Transport Tests (S1–S15, aligned with device_api_inter K1–K15)
// ===========================================================================

// ---------------------------------------------------------------------------
// S1: Transport Handle — GetFromCommS
// ---------------------------------------------------------------------------

__global__ void kernelNetGetFromCommS(const void *devCommPtr, int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
    results[0] = (net != nullptr) ? 1 : 0;
    results[1] = flagcxDevCommGetIntraSize(devCommPtr);
  }
}

void launchKernelNetGetFromCommS(const void *devCommPtr, int *devResults,
                                 flagcxStream_t stream) {
  kernelNetGetFromCommS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// S2: Signal/Counter Reset
// ---------------------------------------------------------------------------

__global__ void kernelNetResetS(const void *devCommPtr, int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
    if (net == nullptr) {
      results[0] = 0;
      return;
    }

    const flagcxDevNet *netObj = (const flagcxDevNet *)net;
    if (!netObj->isValid()) {
      results[0] = 0;
      return;
    }

    // Reset signal slot 0
    flagcxDevNetResetSignal(net, (flagcxDevSignal_t)0);
    // Read it — should be 0
    uint64_t sig0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                            flagcxDeviceMemoryOrderRelaxed);
    results[0] = (sig0 == 0) ? 1 : 0;

    // Increase shadow by 5, read signal (still 0, shadow is separate)
    flagcxDevNetIncreaseSignalShadow(net, (flagcxDevSignal_t)0, 5);
    uint64_t sig1 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                            flagcxDeviceMemoryOrderRelaxed);
    results[1] = (sig1 == 0) ? 1 : 0;

    // Reset counter slot 0
    flagcxDevNetResetCounter(net, (flagcxDevCounter_t)0);
    // Read counter — should be 0
    uint64_t ctr0 = flagcxDevNetReadCounterS(net, (flagcxDevCounter_t)0, 64,
                                             flagcxDeviceMemoryOrderRelaxed);
    results[2] = (ctr0 == 0) ? 1 : 0;
  }
}

void launchKernelNetResetS(const void *devCommPtr, int *devResults,
                           flagcxStream_t stream) {
  kernelNetResetS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, devResults);
}

// ===========================================================================
// S3–S8: One-sided transport kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// S11: WaitSignalS + FlushS (standalone)
// Each rank signals all inter peers, waits for signals from all inter peers.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// S11: WaitSignal + Flush (standalone)
// Reset signal, signal all inter peers, wait for signals, then flush.
// ---------------------------------------------------------------------------

__global__ void kernelNetWaitSignalFlushS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // Reset signal slot 0 (aligned with K11:1410) — no guard, matches K11
  flagcxDevNetResetSignal(net, (flagcxDevSignal_t)0);


  // Read baseline signal (aligned with K11:1411)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K11:1412)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Signal all inter peers (aligned with K11:1417-1420)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer < intraBase || peer >= intraBase + intraSize) {
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevSignal_t)0);
    }
  }


  // Wait for signals from all inter peers (aligned with K11:1423-1424)
  if (nInterRanks > 0) {
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                            s0 + (uint64_t)nInterRanks, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }


  // Flush (aligned with K11:1427)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final world barrier (aligned with K11:1429)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetWaitSignalFlushS(const void *devCommPtr,
                                     flagcxStream_t stream) {
  kernelNetWaitSignalFlushS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// WaitCounterS (COMMENTED — standalone SignalCtrIncS is not supported
// by the GIN protocol. Counter wait is tested in S5 via PutS_RSigInc_LCtrInc.)
// ---------------------------------------------------------------------------

// __global__ void kernelNetWaitCounterS(const void *devCommPtr) {
//   if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
//     int myRank = flagcxDevCommGetRank(devCommPtr);
//     int nRanks = flagcxDevCommGetSize(devCommPtr);
//     int next = (myRank + 1) % nRanks;
//
//     const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
//     if (!net) return;
//
//     uint64_t c0 = flagcxDevNetReadCounterS(net, (flagcxDevCounter_t)0, 64,
//                                            flagcxDeviceMemoryOrderRelaxed);
//
//     flagcxDevNetSignalCtrIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, next,
//                               FLAGCX_COOP_BLOCK, (flagcxDevCounter_t)0);
//
//     flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevCounter_t)0,
//                              c0 + 1, 64, flagcxDeviceMemoryOrderAcquire);
//   }
// }
//
// void launchKernelNetWaitCounterS(const void *devCommPtr,
//                                  flagcxStream_t stream) {
//   kernelNetWaitCounterS<<<1, 32, 0, stream->base>>>(devCommPtr);
// }

// ---------------------------------------------------------------------------
// S10: Shadow (MeetShadowS — commented in test driver)
// increaseSignalShadow + signalSigInc to inter peers + waitSignalMeetShadow
// ---------------------------------------------------------------------------

__global__ void kernelNetWaitSignalMeetShadowS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  int nInterPeers = nRanks - intraSize;

  // Reset signal slot 2 and increase shadow
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevSignal_t)2);
    flagcxDevNetIncreaseSignalShadow(net, (flagcxDevSignal_t)2,
                                     (uint64_t)nInterPeers);
  }

  // First world barrier: ensure all ranks have reset + increased shadow
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Second world barrier: ensure all ranks are ready before signaling
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: signal all inter peers, then wait
  if (FLAGCX_THREAD_IDX_X == 0) {
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevSignal_t)2);
    }

    // Wait until signal meets shadow
    flagcxDevNetWaitSignalMeetShadowS(net, FLAGCX_COOP_THREAD,
                                      (flagcxDevSignal_t)2, 64,
                                      flagcxDeviceMemoryOrderAcquire);
  }
}

void launchKernelNetWaitSignalMeetShadowS(const void *devCommPtr,
                                          flagcxStream_t stream) {
  kernelNetWaitSignalMeetShadowS<<<1, 32, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S12: Inter-Barrier Test
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// S12: Inter Barrier (stress test)
// Tests inter-node barrier synchronization with multiple iterations.
// ---------------------------------------------------------------------------

__global__ void kernelInterBarrierStress(const void *devCommPtr,
                                         int *devResults, int nIters) {
  int myRank = flagcxDevCommGetRank(devCommPtr);


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
      devResults[0] = -1; // no net context
    }
    return;
  }


  // Inter barrier loop (aligned with K12:1461-1463)
  for (int i = 0; i < nIters; i++) {
    flagcxInterBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X,
                            flagcxDeviceMemoryOrderAcqRel,
                            flagcxDevNetFenceLevel::Relaxed);
  }

  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    devResults[0] = 1; // success (aligned with K12:1466)
  }
}

void launchKernelInterBarrierS(const void *devCommPtr, int *devResults,
                               int nIters, flagcxStream_t stream) {
  kernelInterBarrierStress<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, devResults,
                                                         nIters);
}

// ---------------------------------------------------------------------------
// S6: FlushDecouple — PutS(None,None) + FlushS + SignalSigIncS + WaitSignalS + FlushS
// AlltoAll: put with no signal, flush, then signal separately, wait, flush.
// ---------------------------------------------------------------------------

__global__ void kernelNetFlushDecoupleS(const void *devCommPtr,
                                        const void *sendMemPtr,
                                        const void *recvMemPtr,
                                        size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);
  int nInterRanks = nRanks - intraSize;

  // Read baseline signal (aligned with K6:692)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // Pre-barrier (aligned with K6:693)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // Thread-parallelized put with None+None (aligned with K6:701-708)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                     recvMemPtr, (size_t)myRank * chunkBytes,
                     sendMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
  }


  // Flush BEFORE signaling (aligned with K6:709)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Thread-parallelized signal loop (aligned with K6:712-716)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                              FLAGCX_COOP_THREAD, (flagcxDevSignal_t)0);
  }


  // WaitSignal (aligned with K6:717)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + (uint64_t)nInterRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);


  // Flush after wait (aligned with K6:718)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final barrier (aligned with K6:719)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetFlushDecoupleS(const void *devCommPtr,
                                   const void *sendMemPtr,
                                   const void *recvMemPtr, size_t countPerPeer,
                                   flagcxStream_t stream) {
  kernelNetFlushDecoupleS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                                        recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S3: PutS_RSigInc + WaitSignalS + FlushS
// AlltoAll with fused remote signal increment.
// ---------------------------------------------------------------------------

__global__ void kernelNetPutSignalIncS(const void *devCommPtr,
                                       const void *sendMemPtr,
                                       const void *recvMemPtr,
                                       size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }

  size_t chunkBytes = countPerPeer * sizeof(float);

  // World barrier before reading baseline signal (aligned with K3:386-387)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  // Read baseline signal (aligned with K3:388)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  // World barrier sync (aligned with K3:395)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  // Thread-parallelized put loop (aligned with K3:411-422)
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                             recvMemPtr, (size_t)myRank * chunkBytes,
                             sendMemPtr, (size_t)peer * chunkBytes,
                             chunkBytes, FLAGCX_COOP_THREAD,
                             (flagcxDevSignal_t)0);
  }

  // WaitSignal + Flush (aligned with K3:429-430)
  int nInterRanks = nRanks - intraSize;
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + (uint64_t)nInterRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);

  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  // Final world barrier (aligned with K3:436)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);
}

void launchKernelNetPutSignalIncS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream) {
  kernelNetPutSignalIncS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                                       recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S4: PutS_RSigAdd + WaitSignalS + FlushS
// AlltoAll with remote signal add (value = 1 per peer).
// ---------------------------------------------------------------------------

__global__ void kernelNetPutSignalAddS(const void *devCommPtr,
                                       const void *sendMemPtr,
                                       const void *recvMemPtr,
                                       size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);

  // World barrier before reading baseline signal (aligned with K4:470-471)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Read baseline signal (aligned with K4:472)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K4:473)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // Thread-parallelized put + separate signal loop (aligned with K4:477-486)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    // Put with None+None (aligned with K4:479-483)
    flagcxDevNetPutS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                     recvMemPtr, (size_t)myRank * chunkBytes,
                     sendMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
    // Separate SignalAdd with value=2 (aligned with K4:484-485)
    flagcxDevNetSignalSigAddS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                              FLAGCX_COOP_THREAD, (flagcxDevSignal_t)0, 2);
  }


  // WaitSignal for s0 + nInterRanks * 2 (aligned with K4:487)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + (uint64_t)nInterRanks * 2, 64,
                          flagcxDeviceMemoryOrderAcquire);


  // Flush (aligned with K4:488)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final world barrier (aligned with K4:489)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetPutSignalAddS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream) {
  kernelNetPutSignalAddS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                                       recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S5: PutS_RSigInc_LCtrInc + WaitSignalS + WaitCounterS + FlushS
// AlltoAll with both remote signal inc and local counter inc.
// ---------------------------------------------------------------------------

__global__ void kernelNetCounterPipelineS(const void *devCommPtr,
                                          const void *sendMemPtr,
                                          const void *recvMemPtr,
                                          size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);
  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // World barrier before reading baselines (aligned with K5:521-522)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Read baseline signal and counter (aligned with K5:523-524)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);
  uint64_t c0 = flagcxDevNetReadCounterS(net, (flagcxDevCounter_t)0, 64,
                                          flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K5:525)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Round 1: Put with SignalInc + CounterInc (aligned with K5:529-536)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc_LCtrInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                     recvMemPtr, (size_t)myRank * chunkBytes,
                                     sendMemPtr, (size_t)peer * chunkBytes,
                                     chunkBytes, FLAGCX_COOP_THREAD,
                                     (flagcxDevSignal_t)0,
                                     (flagcxDevCounter_t)0);
  }


  // WaitCounter (aligned with K5:537)
  flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevCounter_t)0,
                           c0 + (uint64_t)nInterRanks, 64,
                           flagcxDeviceMemoryOrderAcquire);


  // Stamp sentinel (aligned with K5:540-541)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    float *slot = (float *)flagcxGetLocalPointerS(sendMemPtr, (size_t)peer * chunkBytes);
    *slot = 999.0f;
  }

  // Barrier between rounds (aligned with K5:542)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Round 2: Put with SignalInc + CounterInc again (aligned with K5:546-553)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc_LCtrInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                     recvMemPtr, (size_t)myRank * chunkBytes,
                                     sendMemPtr, (size_t)peer * chunkBytes,
                                     chunkBytes, FLAGCX_COOP_THREAD,
                                     (flagcxDevSignal_t)0,
                                     (flagcxDevCounter_t)0);
  }


  // WaitCounter for c0 + 2*nInterRanks (aligned with K5:554)
  flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevCounter_t)0,
                           c0 + 2 * (uint64_t)nInterRanks, 64,
                           flagcxDeviceMemoryOrderAcquire);


  // WaitSignal for s0 + 2*nInterRanks (aligned with K5:555)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + 2 * (uint64_t)nInterRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);


  // Flush (aligned with K5:556)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final world barrier (aligned with K5:562)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetCounterPipelineS(const void *devCommPtr,
                                     const void *sendMemPtr,
                                     const void *recvMemPtr,
                                     size_t countPerPeer,
                                     flagcxStream_t stream) {
  kernelNetCounterPipelineS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(
      devCommPtr, sendMemPtr, recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S9: Signal (SigInc + SigAdd) — merged into single kernel
// Tests both SignalSigIncS and SignalSigAddS + WaitSignalS in sequence.
// ---------------------------------------------------------------------------

__global__ void kernelNetSignalS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // World barrier before reading baseline (aligned with K9:653-654)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Read baseline signal on slot 1 (aligned with K9:655)
  uint64_t s1 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)1, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K9:656)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Signal loop (aligned with K9:659-662)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer != myRank && (peer < intraBase || peer >= intraBase + intraSize)) {
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevSignal_t)1);
    }
  }


  // WaitSignal (aligned with K9:663-664)
  if (nInterRanks > 0) {
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)1,
                            s1 + (uint64_t)nInterRanks, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }


  // Final world barrier (aligned with K9:665)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetSignalS(const void *devCommPtr, flagcxStream_t stream) {
  kernelNetSignalS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S7: PutValue — tests both PutValueS(None)+Signal and PutValueS_RSigInc
// Each rank writes uint64_t value = myRank*1000 + peer to peer's recv area.
// Phase 1: PutValueS(None) + SignalSigIncS + WaitSignalS
// Phase 2: PutValueS_RSigInc + WaitSignalS (fused putValue + signal)
// ---------------------------------------------------------------------------

__global__ void kernelNetPutValueS(const void *devCommPtr,
                                   const void *recvMemPtr,
                                   size_t putValBase) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // World barrier before reading baseline (aligned with K7:604-605)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Read baseline signal on slot 1 (aligned with K7:606)
  uint64_t s1 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)1, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K7:607)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // PutValue loop (aligned with K7:608-620)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
    flagcxDevNetPutValueS_RSigInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                  recvMemPtr, putValBase + (size_t)myRank * sizeof(uint64_t),
                                  val, FLAGCX_COOP_THREAD, (flagcxDevSignal_t)1);
  }


  // WaitSignal (aligned with K7:622-623)
  if (nInterRanks > 0) {
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)1,
                            s1 + (uint64_t)nInterRanks, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }


  // Final world barrier (aligned with K7:624)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetPutValueS(const void *devCommPtr, const void *recvMemPtr,
                              size_t putValBase, flagcxStream_t stream) {
  kernelNetPutValueS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, recvMemPtr,
                                                  putValBase);
}

// ---------------------------------------------------------------------------
// S8: GetS + FlushS
// AlltoAll via one-sided get: each rank pulls from every inter peer.
// ---------------------------------------------------------------------------

__global__ void kernelNetGetS(const void *devCommPtr, const void *sendMemPtr,
                              const void *recvMemPtr, size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // World barrier (aligned with K8:975)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Get loop (aligned with K8:981-989)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetGetS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                     sendMemPtr, (size_t)myRank * chunkBytes,
                     recvMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
  }


  // Flush (aligned with K8:990)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final world barrier (aligned with K8:991)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetGetS(const void *devCommPtr, const void *sendMemPtr,
                         const void *recvMemPtr, size_t countPerPeer,
                         flagcxStream_t stream) {
  kernelNetGetS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                              recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S15: Two-sided (COMMENTED)
// ---------------------------------------------------------------------------
// __global__ void kernelNetTwoSidedS(const void *devCommPtr,
//                                    const void *sendMemPtr,
//                                    const void *recvMemPtr,
//                                    size_t countPerPeer) {
//   int myRank = flagcxDevCommGetRank(devCommPtr);
//   int nRanks = flagcxDevCommGetSize(devCommPtr);
//   const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
//   if (!net) return;
//   size_t chunkBytes = countPerPeer * sizeof(float);
//   // Post receives from all peers
//   for (int peer = 0; peer < nRanks; peer++) {
//     if (peer == myRank) continue;
//     flagcxDevNetRecvS(net, FLAGCX_COOP_BLOCK, recvMemPtr,
//                       (size_t)peer * chunkBytes, countPerPeer,
//                       flagcxFloat, peer);
//   }
//   // Send to all peers
//   for (int peer = 0; peer < nRanks; peer++) {
//     if (peer == myRank) continue;
//     flagcxDevNetSendS(net, FLAGCX_COOP_BLOCK, sendMemPtr,
//                       (size_t)peer * chunkBytes, countPerPeer,
//                       flagcxFloat, peer);
//   }
//   flagcxDevNetTermS(net, FLAGCX_COOP_BLOCK);
//   flagcxDevNetWaitS(net, FLAGCX_COOP_BLOCK);
// }
//
// void launchKernelNetTwoSidedS(const void *devCommPtr, const void *sendMemPtr,
//                               const void *recvMemPtr, size_t countPerPeer,
//                               flagcxStream_t stream) {
//   kernelNetTwoSidedS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
//                                                    recvMemPtr, countPerPeer);
// }

// ---------------------------------------------------------------------------
// S13: WorldBarrierS — sync + arrive/wait split in one kernel
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// S13: WorldBarrierS — sync + arrive/wait split in one kernel
// Tests world barrier synchronization in both sync and split (arrive/wait) modes.
// ---------------------------------------------------------------------------

__global__ void kernelWorldBarrierS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  // Test sync (aligned with K13:1496)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);


  // Test arrive + wait (split) (aligned with K13:1499-1500)
  flagcxWorldBarrierArriveS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                            flagcxDeviceMemoryOrderRelease,
                            flagcxDevNetFenceLevel::Relaxed);


  flagcxWorldBarrierWaitS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcquire,
                          flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelWorldBarrierS(const void *devCommPtr, flagcxStream_t stream) {
  kernelWorldBarrierS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S14: OneSidedAlltoAll (composite) — put + signal + wait + flush + world barrier
// Each rank puts its chunk to every inter peer using PutS_RSigInc,
// waits for signals, flushes, then world barrier for completion.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// S14: OneSidedAlltoAll — composite put + signal + wait + flush + world barrier
// One-sided alltoall pattern using put with signal increment.
// ---------------------------------------------------------------------------

__global__ void kernelNetOneSidedAlltoAllS(const void *devCommPtr,
                                           const void *sendMemPtr,
                                           const void *recvMemPtr,
                                           size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);

  // Read signal baseline (aligned with K14:210)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // Pre-communication barrier (aligned with K14:213)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Thread-parallelized put loop (aligned with K14:217-221)
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    flagcxDevNetPutS_RSigInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                             recvMemPtr, (size_t)myRank * chunkBytes,
                             sendMemPtr, (size_t)peer * chunkBytes,
                             chunkBytes, FLAGCX_COOP_THREAD,
                             (flagcxDevSignal_t)0);
  }


  // Wait for all incoming signals (aligned with K14:223)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + (uint64_t)nRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);


  // Flush to ensure data visibility (aligned with K14:224)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Post-communication barrier (aligned with K14:227)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetOneSidedAlltoAllS(const void *devCommPtr,
                                      const void *sendMemPtr,
                                      const void *recvMemPtr,
                                      size_t countPerPeer,
                                      flagcxStream_t stream) {
  kernelNetOneSidedAlltoAllS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(
      devCommPtr, sendMemPtr, recvMemPtr, countPerPeer);
}

// ===========================================================================
// Unified One-Sided IR Tests (S16–S22)
// ===========================================================================

// ---------------------------------------------------------------------------
// S16: flagcxDevPut — All three teams (INTRA / WORLD / INTER)
// Uses 3 offset regions in dst/src: [0, bytes), [bytes, 2*bytes), [2*bytes, 3*bytes)
// ---------------------------------------------------------------------------
__global__ void kernelDevPutS(const void *devCommPtr,
                              const void *dstMemPtr,
                              const void *srcMemPtr,
                              int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();

  // Early exit for blocks beyond nContexts (only nContexts blocks launched)
  if (FLAGCX_BLOCK_IDX_X >= nContexts) return;

  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  // --- INTRA team: peer = next intra-rank ---
  int intraPeer = (intraRank + 1) % intraSize;
  flagcxDevContext_t contextId = FLAGCX_BLOCK_IDX_X; // blockIdx IS contextId (one-to-one)

  // DEBUG: trace dispatch path (block 0, thread 0 only)
  if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
    const flagcxDevMem *dst = (const flagcxDevMem *)dstMemPtr;
    const flagcxDevMem *src = (const flagcxDevMem *)srcMemPtr;
    flagcxTeam team = flagcxMakeTeamFromKind(*comm, FLAGCX_TEAM_INTRA);
    void *peerPtr = flagcxGetPeerPointer(*dst, 0, team, intraPeer);
    void *localSrc = flagcxGetLocalPointer(*src, 0);
    const void *netOpaque = flagcxDevNetGetFromCommS(devCommPtr, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    printf("[rank %d] S16 DEBUG: nContexts=%d contextId=%d intraPeer=%d "
           "peerPtr=%p localSrc=%p net=%p fifo=%p\n",
           worldRank, nContexts, contextId, intraPeer,
           peerPtr, localSrc, net, net ? net->fifoBuffer : nullptr);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // DEBUG: print CoopGrid info from each block
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxCoopAny gridCoop = flagcxMakeCoopFromKind(FLAGCX_COOP_GRID);
    printf("[rank %d blk %d] S16 GRID: threadRank=%d size=%d gridDim=%d blockDim=%d nContexts=%d bytes=%zu n4=%zu\n",
           worldRank, (int)FLAGCX_BLOCK_IDX_X,
           gridCoop.threadRank(), gridCoop.size(),
           (int)FLAGCX_GRID_DIM_X, (int)FLAGCX_BLOCK_DIM_X,
           nContexts, bytes, bytes / 4);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // DEBUG: verify peerPtr from each block (not just block 0)
  if (FLAGCX_THREAD_IDX_X == 0) {
    const flagcxDevMem *dst = (const flagcxDevMem *)dstMemPtr;
    flagcxTeam team = flagcxMakeTeamFromKind(*comm, FLAGCX_TEAM_INTRA);
    void *peerPtr = flagcxGetPeerPointer(*dst, 0, team, intraPeer);
    printf("[rank %d blk %d] S16 peerPtr(INTRA,peer=%d)=%p\n",
           worldRank, (int)FLAGCX_BLOCK_IDX_X, intraPeer, peerPtr);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  flagcxDevPut(devCommPtr, dstMemPtr, /*dstOff=*/0,
               srcMemPtr, /*srcOff=*/0, bytes,
               FLAGCX_TEAM_INTRA, intraPeer, contextId, FLAGCX_COOP_GRID,
               flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);

  // DEBUG: after INTRA put, check what was written (from block 0 only)
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
    __threadfence_system();
    // Check local recvBuff (what the PREVIOUS rank wrote to us)
    const flagcxDevMem *dst = (const flagcxDevMem *)dstMemPtr;
    float *localRecv = (float *)flagcxGetLocalPointer(*dst, 0);
    printf("[rank %d] S16 POST-INTRA localRecv: [0]=%f [1]=%f [last]=%f\n",
           worldRank, localRecv[0], localRecv[1], localRecv[bytes/4 - 1]);
    // Check what we wrote to peer (read via peerPtr)
    flagcxTeam team = flagcxMakeTeamFromKind(*comm, FLAGCX_TEAM_INTRA);
    float *peerRecv = (float *)flagcxGetPeerPointer(*dst, 0, team, intraPeer);
    if (peerRecv) {
      printf("[rank %d] S16 POST-INTRA peerRecv: [0]=%f [1]=%f [last]=%f\n",
             worldRank, peerRecv[0], peerRecv[1], peerRecv[bytes/4 - 1]);
    }
    // Also check src to verify what we SHOULD have written
    const flagcxDevMem *src = (const flagcxDevMem *)srcMemPtr;
    float *localSrc = (float *)flagcxGetLocalPointer(*src, 0);
    printf("[rank %d] S16 SRC: [0]=%f [1]=%f [last]=%f\n",
           worldRank, localSrc[0], localSrc[1], localSrc[bytes/4 - 1]);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // --- WORLD team: peer = next world-rank ---
  int worldPeer = (worldRank + 1) % nRanks;
  flagcxDevPut(devCommPtr, dstMemPtr, /*dstOff=*/bytes,
               srcMemPtr, /*srcOff=*/bytes, bytes,
               FLAGCX_TEAM_WORLD, worldPeer, contextId, FLAGCX_COOP_GRID,
               flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);

  // --- INTER team: peer = next node-index (self-op if nNodes==1) ---
  int interPeer = (nNodes > 1) ? ((nodeIdx + 1) % nNodes) : nodeIdx;
  flagcxDevPut(devCommPtr, dstMemPtr, /*dstOff=*/2 * bytes,
               srcMemPtr, /*srcOff=*/2 * bytes, bytes,
               FLAGCX_TEAM_INTER, interPeer, contextId, FLAGCX_COOP_GRID,
               flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);

  // Flush to ensure all FIFO operations complete before kernel returns
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_THREAD,
                   flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) result[0] = 1;
}

void launchKernelDevPutS(const void *devCommPtr, const void *dstMemPtr,
                         const void *srcMemPtr, int *devResult, size_t bytes,
                         flagcxStream_t stream) {
  kernelDevPutS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, dstMemPtr,
                                                    srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S17: flagcxDevPut_RSigInc + flagcxDevWaitSignal — pipeline (3 teams)
// Uses signal slots 0, 1, 2 for INTRA, WORLD, INTER respectively.
// Uses 3 offset regions: [0, bytes), [bytes, 2*bytes), [2*bytes, 3*bytes).
// NOTE: Requires concurrent multi-rank launch (ring dependency).
// ---------------------------------------------------------------------------
__global__ void kernelDevPutSignalWaitS(const void *devCommPtr,
                                        const void *dstMemPtr,
                                        const void *srcMemPtr,
                                        int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();

  // Early exit for blocks beyond nContexts
  if (FLAGCX_BLOCK_IDX_X >= nContexts) return;

  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  flagcxDevContext_t contextId = FLAGCX_BLOCK_IDX_X; // blockIdx IS contextId

  // DEBUG: trace dispatch path (block 0, thread 0 only)
  if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
    const flagcxDevMem *dst = (const flagcxDevMem *)dstMemPtr;
    flagcxTeam team = flagcxMakeTeamFromKind(*comm, FLAGCX_TEAM_INTRA);
    int intraPeerDbg = (intraRank + 1) % intraSize;
    void *peerPtr = flagcxGetPeerPointer(*dst, 0, team, intraPeerDbg);
    const void *netOpaque = flagcxDevNetGetFromCommS(devCommPtr, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    printf("[rank %d] S17 DEBUG: nContexts=%d contextId=%d peerPtr=%p "
           "net=%p fifo=%p signalBuf=%p signalCount=%d\n",
           worldRank, nContexts, contextId, peerPtr,
           net, net ? net->fifoBuffer : nullptr,
           net ? (void*)net->signalBuffer : nullptr,
           net ? net->signalCount : -1);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // --- INTRA: put + signal slot 0 ---
  int intraPeer = (intraRank + 1) % intraSize;
  flagcxDevPut_RSigInc(devCommPtr, dstMemPtr, 0, srcMemPtr, 0, bytes,
                       FLAGCX_TEAM_INTRA, intraPeer, contextId, FLAGCX_COOP_GRID,
                       flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease,
                       (flagcxDevSignal_t)0);

  // Wait for INTRA signal from previous peer
  flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)0, /*least=*/1,
                      /*bits=*/64, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceMemoryOrderAcquire);

  // --- WORLD: put + signal slot 1 ---
  int worldPeer = (worldRank + 1) % nRanks;
  flagcxDevPut_RSigInc(devCommPtr, dstMemPtr, bytes, srcMemPtr, bytes, bytes,
                       FLAGCX_TEAM_WORLD, worldPeer, contextId, FLAGCX_COOP_GRID,
                       flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease,
                       (flagcxDevSignal_t)1);

  // Wait for WORLD signal from previous peer
  flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)1, /*least=*/1,
                      /*bits=*/64, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceMemoryOrderAcquire);

  // --- INTER: put + signal slot 2 (self-op if nNodes==1) ---
  int interPeer = (nNodes > 1) ? ((nodeIdx + 1) % nNodes) : nodeIdx;
  flagcxDevPut_RSigInc(devCommPtr, dstMemPtr, 2 * bytes, srcMemPtr, 2 * bytes,
                       bytes, FLAGCX_TEAM_INTER, interPeer, contextId, FLAGCX_COOP_GRID,
                       flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease,
                       (flagcxDevSignal_t)2);

  // Wait for INTER signal (self-signal if nNodes==1)
  flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)2, /*least=*/1,
                      /*bits=*/64, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceMemoryOrderAcquire);

  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) result[0] = 1;
}
void launchKernelDevPutSignalWaitS(const void *devCommPtr,
                                   const void *dstMemPtr,
                                   const void *srcMemPtr, int *devResult,
                                   size_t bytes,
                                   flagcxStream_t stream) {
  kernelDevPutSignalWaitS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S18: flagcxDevGet — All three teams (INTRA / WORLD / INTER)
// Uses 3 offset regions in remote/local: [0, bytes), [bytes, 2*bytes), [2*bytes, 3*bytes)
// ---------------------------------------------------------------------------
__global__ void kernelDevGetS(const void *devCommPtr,
                              const void *remoteMemPtr,
                              const void *localMemPtr,
                              int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();

  // Early exit for blocks beyond nContexts
  if (FLAGCX_BLOCK_IDX_X >= nContexts) return;

  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  // --- INTRA team: get from next intra-rank ---
  int intraPeer = (intraRank + 1) % intraSize;
  flagcxDevContext_t contextId = FLAGCX_BLOCK_IDX_X; // blockIdx IS contextId

  // DEBUG (block 0, thread 0)
  if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
    const flagcxDevMem *remote = (const flagcxDevMem *)remoteMemPtr;
    flagcxTeam team = flagcxMakeTeamFromKind(*comm, FLAGCX_TEAM_INTRA);
    void *peerPtr = flagcxGetPeerPointer(*remote, 0, team, intraPeer);
    const void *netOpaque = flagcxDevNetGetFromCommS(devCommPtr, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    printf("[rank %d] S18 DEBUG: nContexts=%d contextId=%d intraPeer=%d "
           "peerPtr=%p net=%p fifo=%p\n",
           worldRank, nContexts, contextId, intraPeer,
           peerPtr, net, net ? net->fifoBuffer : nullptr);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  flagcxDevGet(devCommPtr, remoteMemPtr, 0, localMemPtr, 0, bytes,
               FLAGCX_TEAM_INTRA, intraPeer, contextId, FLAGCX_COOP_GRID,
               flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);

  // --- WORLD team: get from next world-rank ---
  int worldPeer = (worldRank + 1) % nRanks;
  flagcxDevGet(devCommPtr, remoteMemPtr, bytes, localMemPtr, bytes, bytes,
               FLAGCX_TEAM_WORLD, worldPeer, contextId, FLAGCX_COOP_GRID,
               flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);

  // --- INTER team: get from next node-index (self-op if nNodes==1) ---
  int interPeer = (nNodes > 1) ? ((nodeIdx + 1) % nNodes) : nodeIdx;
  flagcxDevGet(devCommPtr, remoteMemPtr, 2 * bytes, localMemPtr, 2 * bytes,
               bytes, FLAGCX_TEAM_INTER, interPeer, contextId, FLAGCX_COOP_GRID,
               flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);

  // Flush to ensure all FIFO operations complete before kernel returns
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_THREAD,
                   flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) result[0] = 1;
}
void launchKernelDevGetS(const void *devCommPtr, const void *remoteMemPtr,
                         const void *localMemPtr, int *devResult, size_t bytes,
                         flagcxStream_t stream) {
  kernelDevGetS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, remoteMemPtr,
                                                    localMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S19: flagcxDevBarrierSync — Intra-node
// ---------------------------------------------------------------------------
__global__ void kernelDevBarrierIntraS(const void *devCommPtr, int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? FLAGCX_BLOCK_IDX_X % nContexts : 0;

  // DEBUG (block 0, thread 0)
  if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int worldRank = flagcxDevCommGetRank(devCommPtr);
    const void *netOpaque = flagcxDevNetGetFromCommS(devCommPtr, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    printf("[rank %d] S19 DEBUG: nContexts=%d contextId=%d net=%p fifo=%p\n",
           worldRank, nContexts, contextId, net,
           net ? net->fifoBuffer : nullptr);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_INTRA, /*index=*/FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcqRel,
                       flagcxDeviceScopeSystem);
  if (FLAGCX_THREAD_IDX_X == 0) result[FLAGCX_BLOCK_IDX_X] = 1;
}

void launchKernelDevBarrierIntraS(const void *devCommPtr, int *devResult,
                                  flagcxStream_t stream) {
  kernelDevBarrierIntraS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, devResult);
}

// ---------------------------------------------------------------------------
// S20: flagcxDevBarrierSync — World (intra + inter)
// ---------------------------------------------------------------------------
__global__ void kernelDevBarrierWorldS(const void *devCommPtr, int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? FLAGCX_BLOCK_IDX_X % nContexts : 0;

  // DEBUG (block 0, thread 0)
  if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int worldRank = flagcxDevCommGetRank(devCommPtr);
    const void *netOpaque = flagcxDevNetGetFromCommS(devCommPtr, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    printf("[rank %d] S20 DEBUG: nContexts=%d contextId=%d net=%p fifo=%p\n",
           worldRank, nContexts, contextId, net,
           net ? net->fifoBuffer : nullptr);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_WORLD, /*index=*/FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcqRel,
                       flagcxDeviceScopeSystem);
  if (FLAGCX_THREAD_IDX_X == 0) result[FLAGCX_BLOCK_IDX_X] = 1;
}

void launchKernelDevBarrierWorldS(const void *devCommPtr, int *devResult,
                                  flagcxStream_t stream) {
  kernelDevBarrierWorldS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, devResult);
}

// ---------------------------------------------------------------------------
// S21: flagcxDevPut — Warp-level (fine-grained), all three teams
// Uses 3 offset regions: [0, bytes), [bytes, 2*bytes), [2*bytes, 3*bytes)
// Each warp (32 threads) handles one put; uses first warp of block per team.
// ---------------------------------------------------------------------------
__global__ void kernelDevPutWarpS(const void *devCommPtr,
                                  const void *dstMemPtr,
                                  const void *srcMemPtr,
                                  int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();

  // Early exit for blocks beyond nContexts
  if (FLAGCX_BLOCK_IDX_X >= nContexts) return;

  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  flagcxDevContext_t contextId = FLAGCX_BLOCK_IDX_X; // blockIdx IS contextId

  // DEBUG (block 0, thread 0)
  if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
    const flagcxDevMem *dst = (const flagcxDevMem *)dstMemPtr;
    flagcxTeam team = flagcxMakeTeamFromKind(*comm, FLAGCX_TEAM_INTRA);
    int intraPeerDbg = (intraRank + 1) % intraSize;
    void *peerPtr = flagcxGetPeerPointer(*dst, 0, team, intraPeerDbg);
    const void *netOpaque = flagcxDevNetGetFromCommS(devCommPtr, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    printf("[rank %d] S21 DEBUG: nContexts=%d contextId=%d peerPtr=%p "
           "net=%p fifo=%p\n",
           worldRank, nContexts, contextId, peerPtr,
           net, net ? net->fifoBuffer : nullptr);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Warp 0: INTRA put
  if (FLAGCX_THREAD_IDX_X < 32) {
    int intraPeer = (intraRank + 1) % intraSize;
    flagcxDevPut(devCommPtr, dstMemPtr, 0, srcMemPtr, 0, bytes,
                 FLAGCX_TEAM_INTRA, intraPeer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  // Warp 1: WORLD put
  if (FLAGCX_THREAD_IDX_X >= 32 && FLAGCX_THREAD_IDX_X < 64) {
    int worldPeer = (worldRank + 1) % nRanks;
    flagcxDevPut(devCommPtr, dstMemPtr, bytes, srcMemPtr, bytes, bytes,
                 FLAGCX_TEAM_WORLD, worldPeer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  // Warp 2: INTER put (self-op if nNodes==1)
  if (FLAGCX_THREAD_IDX_X >= 64 && FLAGCX_THREAD_IDX_X < 96) {
    int interPeer = (nNodes > 1) ? ((nodeIdx + 1) % nNodes) : nodeIdx;
    flagcxDevPut(devCommPtr, dstMemPtr, 2 * bytes, srcMemPtr, 2 * bytes, bytes,
                 FLAGCX_TEAM_INTER, interPeer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Flush to ensure all FIFO operations complete before kernel returns
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_THREAD,
                   flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) result[0] = 1;
}

void launchKernelDevPutWarpS(const void *devCommPtr, const void *dstMemPtr,
                             const void *srcMemPtr, int *devResult, size_t bytes,
                             flagcxStream_t stream) {
  kernelDevPutWarpS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, dstMemPtr,
                                                        srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S22: flagcxDevSignalInc + flagcxDevWaitSignal — standalone signal (3 teams)
// Uses signal slots 0, 1, 2 for INTRA, WORLD, INTER respectively.
// NOTE: Requires concurrent multi-rank launch (ring dependency).
// ---------------------------------------------------------------------------
__global__ void kernelDevSignalStandaloneS(const void *devCommPtr,
                                           int *result) {
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? FLAGCX_BLOCK_IDX_X % nContexts : 0;

  // DEBUG (block 0, thread 0)
  if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
    const void *netOpaque = flagcxDevNetGetFromCommS(devCommPtr, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    printf("[rank %d] S22 DEBUG: nContexts=%d contextId=%d net=%p "
           "fifo=%p signalBuf=%p signalCount=%d\n",
           worldRank, nContexts, contextId, net,
           net ? net->fifoBuffer : nullptr,
           net ? (void*)net->signalBuffer : nullptr,
           net ? net->signalCount : -1);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // --- INTRA: signal slot 0 ---
  int intraPeer = (intraRank + 1) % intraSize;
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevSignalInc(devCommPtr, FLAGCX_TEAM_INTRA, intraPeer,
                       (flagcxDevSignal_t)0, contextId, FLAGCX_COOP_THREAD,
                       flagcxDeviceScopeSystem);
  }
  flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)0, /*least=*/1,
                      /*bits=*/64, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceMemoryOrderAcquire);

  // --- WORLD: signal slot 1 ---
  int worldPeer = (worldRank + 1) % nRanks;
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevSignalInc(devCommPtr, FLAGCX_TEAM_WORLD, worldPeer,
                       (flagcxDevSignal_t)1, contextId, FLAGCX_COOP_THREAD,
                       flagcxDeviceScopeSystem);
  }
  flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)1, /*least=*/1,
                      /*bits=*/64, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceMemoryOrderAcquire);

  // --- INTER: signal slot 2 (self-signal if nNodes==1) ---
  int interPeer = (nNodes > 1) ? ((nodeIdx + 1) % nNodes) : nodeIdx;
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevSignalInc(devCommPtr, FLAGCX_TEAM_INTER, interPeer,
                       (flagcxDevSignal_t)2, contextId, FLAGCX_COOP_THREAD,
                       flagcxDeviceScopeSystem);
  }
  flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)2, /*least=*/1,
                      /*bits=*/64, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceMemoryOrderAcquire);

  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) result[0] = 1;
}

void launchKernelDevSignalStandaloneS(const void *devCommPtr, int *devResult,
                                      flagcxStream_t stream) {
  kernelDevSignalStandaloneS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr,
                                                           devResult);
}

// ---------------------------------------------------------------------------
// S23: Team-resolution correctness test
//
// Validates that team-scoped peer resolution works correctly for all three
// team kinds. Each rank puts its own world rank (as float tag) into the
// peer's recv buffer at a deterministic offset using each team.
//
// The host-side test driver pre-fills sendBuff[0] = (float)worldRank,
// then verifies the recv buffer after the kernel + barrier.
//
// Layout in dst: 3 regions of sizeof(float) each
//   [0..4): INTRA put result (sender's tag at offset = sender's intraRank * 4)
//   [intraSize*4 .. intraSize*4 + nRanks*4): WORLD put result
//   [intraSize*4 + nRanks*4 .. + nNodes*4): INTER put result
//
// Kernel writes sendBuff[0] (=myWorldRank) to peer's recvBuff at:
//   INTRA: offset = myIntraRank * sizeof(float)
//   WORLD: offset = intraSize*4 + myWorldRank * sizeof(float)
//   INTER: offset = intraSize*4 + nRanks*4 + myNodeIdx * sizeof(float)
// ---------------------------------------------------------------------------
__global__ void kernelDevTeamResolutionS(const void *devCommPtr,
                                         const void *dstMemPtr,
                                         const void *srcMemPtr,
                                         int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();

  // Early exit for blocks beyond nContexts
  if (FLAGCX_BLOCK_IDX_X >= nContexts) return;

  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  // Offsets for each team's region
  size_t intraOff = 0;
  size_t worldOff = (size_t)intraSize * sizeof(float);
  size_t interOff = worldOff + (size_t)nRanks * sizeof(float);

  flagcxDevContext_t contextId = FLAGCX_BLOCK_IDX_X; // blockIdx IS contextId

  // DEBUG (block 0, thread 0)
  if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
    const flagcxDevMem *dst = (const flagcxDevMem *)dstMemPtr;
    flagcxTeam team = flagcxMakeTeamFromKind(*comm, FLAGCX_TEAM_INTRA);
    int intraPeerDbg = (intraRank + 1) % intraSize;
    void *peerPtr = flagcxGetPeerPointer(*dst, 0, team, intraPeerDbg);
    const void *netOpaque = flagcxDevNetGetFromCommS(devCommPtr, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    printf("[rank %d] S23 DEBUG: nContexts=%d contextId=%d peerPtr=%p "
           "net=%p fifo=%p\n",
           worldRank, nContexts, contextId, peerPtr,
           net, net ? net->fifoBuffer : nullptr);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // --- INTRA: put my tag to next intra-rank's buffer at offset[myIntraRank] ---
  int intraPeer = (intraRank + 1) % intraSize;
  size_t intraDstOff = intraOff + (size_t)intraRank * sizeof(float);
  flagcxDevPut(devCommPtr, dstMemPtr, intraDstOff,
               srcMemPtr, 0, sizeof(float),
               FLAGCX_TEAM_INTRA, intraPeer, contextId, FLAGCX_COOP_BLOCK,
               flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);

  // --- WORLD: put my tag to next world-rank's buffer at offset[myWorldRank] ---
  int worldPeer = (worldRank + 1) % nRanks;
  size_t worldDstOff = worldOff + (size_t)worldRank * sizeof(float);
  flagcxDevPut(devCommPtr, dstMemPtr, worldDstOff,
               srcMemPtr, 0, sizeof(float),
               FLAGCX_TEAM_WORLD, worldPeer, contextId, FLAGCX_COOP_BLOCK,
               flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);

  // --- INTER: put my tag to next node-index's buffer at offset[myNodeIdx] ---
  int interPeer = (nNodes > 1) ? ((nodeIdx + 1) % nNodes) : nodeIdx;
  size_t interDstOff = interOff + (size_t)nodeIdx * sizeof(float);
  flagcxDevPut(devCommPtr, dstMemPtr, interDstOff,
               srcMemPtr, 0, sizeof(float),
               FLAGCX_TEAM_INTER, interPeer, contextId, FLAGCX_COOP_BLOCK,
               flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);

  // Flush to ensure all FIFO operations complete before barrier
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_THREAD,
                   flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Barrier to ensure all puts land before host reads
  flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_INTRA, /*index=*/0,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcqRel,
                       flagcxDeviceScopeSystem);

  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) result[0] = 1;
}

void launchKernelDevTeamResolutionS(const void *devCommPtr,
                                    const void *dstMemPtr,
                                    const void *srcMemPtr, int *devResult,
                                    flagcxStream_t stream) {
  kernelDevTeamResolutionS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult);
}
