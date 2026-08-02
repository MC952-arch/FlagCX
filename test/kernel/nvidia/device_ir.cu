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
  if (threadIdx.x == 0 && blockIdx.x == 0) {
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
  if (threadIdx.x == 0) pass = 1;
  __syncthreads();
  if (rank != (int)threadIdx.x || size != (int)blockDim.x)
    atomicExch(&pass, 0);
  __syncthreads();
  if (threadIdx.x == 0) results[0] = pass;
}

// Sub-kernel: tile_span coop check (1 block, 128 threads = 4 tiles of 32)
__global__ void kernelCoopGroupsS_tileSpan(int *results) {
  int tileIdx = threadIdx.x / 32;
  uint32_t t0 = (uint32_t)tileIdx;
  uint32_t nTiles = 1;
  uint32_t id = 0;

  int rank = flagcxCoopThreadRankExS(FLAGCX_COOP_TILE_SPAN, t0, nTiles, id);
  int size = flagcxCoopSizeExS(FLAGCX_COOP_TILE_SPAN, t0, nTiles, id);
  flagcxCoopSyncExS(FLAGCX_COOP_TILE_SPAN, t0, nTiles, id);

  // Expected: rank = threadIdx % 32, size = 32
  __shared__ int pass;
  if (threadIdx.x == 0) pass = 1;
  __syncthreads();
  if (rank != (int)(threadIdx.x % 32) || size != 32)
    atomicExch(&pass, 0);
  __syncthreads();
  if (threadIdx.x == 0) results[1] = pass;
}

// Sub-kernel: lanes coop check (1 block, 32 threads, full warp mask)
__global__ void kernelCoopGroupsS_lanes(int *results) {
  uint32_t laneMask = 0xFFFFFFFF;

  int rank = flagcxCoopThreadRankExS(FLAGCX_COOP_LANES, laneMask, 0, 0);
  int size = flagcxCoopSizeExS(FLAGCX_COOP_LANES, laneMask, 0, 0);
  flagcxCoopSyncExS(FLAGCX_COOP_LANES, laneMask, 0, 0);

  // Expected: rank = lane index, size = 32
  __shared__ int pass;
  if (threadIdx.x == 0) pass = 1;
  __syncthreads();
  if (rank != (int)threadIdx.x || size != 32)
    atomicExch(&pass, 0);
  __syncthreads();
  if (threadIdx.x == 0) results[2] = pass;
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
  if (threadIdx.x == 0 && blockIdx.x == 0) {
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
  if (threadIdx.x == 0 && blockIdx.x == 0) {
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

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
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
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  for (int i = tid; i < count; i += nthreads) {
    buffer[i] = (float)(myRank + 1);
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
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
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  for (int i = tid; i < count; i += nthreads) {
    buffer[i] = (float)(myRank + 500);
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelease);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  for (int i = tid; i < count; i += nthreads) {
    size_t offset = i * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
    output[i] = *peerPtr;
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
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

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
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
  int localNthreads = blockDim.x * gridDim.x;
  int globalTid = threadIdx.x + blockDim.x * (myRank + blockIdx.x * nRanks);
  int globalNthreads = localNthreads * nRanks;

  // Pre-reduce barrier (acquire — ensure peer writes are visible)
  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
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
  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
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
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
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
  if (threadIdx.x == 0 && blockIdx.x == 0) {
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
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
    if (net == nullptr) {
      results[0] = 0; // cannot test without transport
      return;
    }

    const flagcxDevNet *netObj = (const flagcxDevNet *)net;
    if (!netObj->isValid()) {
      results[0] = 0;
      return;
    }

    // Reset signal slot 0
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)0);
    // Read it — should be 0
    uint64_t sig0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                            flagcxDeviceMemoryOrderRelaxed);
    results[0] = (sig0 == 0) ? 1 : 0;

    // Increase shadow by 5, read signal (still 0, shadow is separate)
    flagcxDevNetIncreaseSignalShadow(net, (flagcxDevNetSignal_t)0, 5);
    uint64_t sig1 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                            flagcxDeviceMemoryOrderRelaxed);
    results[1] = (sig1 == 0) ? 1 : 0;

    // Reset counter slot 0
    flagcxDevNetResetCounter(net, (flagcxDevNetCounter_t)0);
    // Read counter — should be 0
    uint64_t ctr0 = flagcxDevNetReadCounterS(net, (flagcxDevNetCounter_t)0, 64,
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S11 rank %d] kernel start\n", myRank);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S11 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S11 rank %d] got net\n", myRank);
  }

  int nInterRanks = nRanks - intraSize;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // Reset signal slot 0 (aligned with K11:1410) — no guard, matches K11
  flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)0);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S11 rank %d] after reset\n", myRank);
  }

  // Read baseline signal (aligned with K11:1411)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S11 rank %d] after ReadSignal, s0=%lu\n", myRank, s0);
  }

  // World barrier sync (aligned with K11:1412)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S11 rank %d] after barrier, before Signal loop\n", myRank);
  }

  // Signal all inter peers (aligned with K11:1417-1420)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer < intraBase || peer >= intraBase + intraSize) {
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)0);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S11 rank %d] after Signal loop\n", myRank);
  }

  // Wait for signals from all inter peers (aligned with K11:1423-1424)
  if (nInterRanks > 0) {
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                            s0 + (uint64_t)nInterRanks, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S11 rank %d] after WaitSignal\n", myRank);
  }

  // Flush (aligned with K11:1427)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S11 rank %d] after Flush\n", myRank);
  }

  // Final world barrier (aligned with K11:1429)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S11 rank %d] after final barrier, kernel done\n", myRank);
  }
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
//   if (threadIdx.x == 0 && blockIdx.x == 0) {
//     int myRank = flagcxDevCommGetRank(devCommPtr);
//     int nRanks = flagcxDevCommGetSize(devCommPtr);
//     int next = (myRank + 1) % nRanks;
//
//     const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
//     if (!net) return;
//
//     uint64_t c0 = flagcxDevNetReadCounterS(net, (flagcxDevNetCounter_t)0, 64,
//                                            flagcxDeviceMemoryOrderRelaxed);
//
//     flagcxDevNetSignalCtrIncS(net, devCommPtr, FLAGCX_TEAM_INTER, next,
//                               FLAGCX_COOP_BLOCK, (flagcxDevNetCounter_t)0);
//
//     flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetCounter_t)0,
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
  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)2);
    flagcxDevNetIncreaseSignalShadow(net, (flagcxDevNetSignal_t)2,
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
  if (threadIdx.x == 0) {
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)2);
    }

    // Wait until signal meets shadow
    flagcxDevNetWaitSignalMeetShadowS(net, FLAGCX_COOP_THREAD,
                                      (flagcxDevNetSignal_t)2, 64,
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S12 rank %d] kernel start, nIters=%d\n", myRank, nIters);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S12 rank %d] net is NULL, skipping\n", myRank);
      devResults[0] = -1; // no net context
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S12 rank %d] got net, starting iterations\n", myRank);
  }

  // Inter barrier loop (aligned with K12:1461-1463)
  for (int i = 0; i < nIters; i++) {
    flagcxInterBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x,
                            flagcxDeviceMemoryOrderAcqRel,
                            flagcxDevNetFenceLevel::Relaxed);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S12 rank %d] after iterations, marking success\n", myRank);
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] kernel start\n", myRank);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S6 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] got net\n", myRank);
  }

  size_t chunkBytes = countPerPeer * sizeof(float);
  int nInterRanks = nRanks - intraSize;

  // Read baseline signal (aligned with K6:692)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] readSignal s0=%lu\n", myRank, s0);
  }

  // Pre-barrier (aligned with K6:693)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] after pre-barrier\n", myRank);
  }

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // Thread-parallelized put with None+None (aligned with K6:701-708)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                     recvMemPtr, (size_t)myRank * chunkBytes,
                     sendMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] after put loop\n", myRank);
  }

  // Flush BEFORE signaling (aligned with K6:709)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] after flush 1\n", myRank);
  }

  // Thread-parallelized signal loop (aligned with K6:712-716)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                              FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)0);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] after signal loop\n", myRank);
  }

  // WaitSignal (aligned with K6:717)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + (uint64_t)nInterRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] after waitSignal\n", myRank);
  }

  // Flush after wait (aligned with K6:718)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] after flush 2\n", myRank);
  }

  // Final barrier (aligned with K6:719)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S6 rank %d] after final barrier, kernel done\n", myRank);
  }
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S3 rank %d] kernel start\n", myRank);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S3 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S3 rank %d] got net\n", myRank);
  }

  size_t chunkBytes = countPerPeer * sizeof(float);

  // World barrier before reading baseline signal (aligned with K3:386-387)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S3 rank %d] after barrier 0\n", myRank);
  }

  // Read baseline signal (aligned with K3:388)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S3 rank %d] after ReadSignal, s0=%lu\n", myRank, s0);
  }

  // World barrier sync (aligned with K3:395)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S3 rank %d] after barrier 1, before Put loop\n", myRank);
  }

  // Thread-parallelized put loop (aligned with K3:411-422)
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                             recvMemPtr, (size_t)myRank * chunkBytes,
                             sendMemPtr, (size_t)peer * chunkBytes,
                             chunkBytes, FLAGCX_COOP_THREAD,
                             (flagcxDevNetSignal_t)0);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S3 rank %d] after Put loop\n", myRank);
  }

  // WaitSignal + Flush (aligned with K3:429-430)
  int nInterRanks = nRanks - intraSize;
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + (uint64_t)nInterRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S3 rank %d] after WaitSignal\n", myRank);
  }

  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S3 rank %d] after Flush\n", myRank);
  }

  // Final world barrier (aligned with K3:436)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S3 rank %d] after final barrier, kernel done\n", myRank);
  }
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S4 rank %d] kernel start\n", myRank);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S4 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S4 rank %d] got net\n", myRank);
  }

  size_t chunkBytes = countPerPeer * sizeof(float);

  // World barrier before reading baseline signal (aligned with K4:470-471)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S4 rank %d] after barrier 0\n", myRank);
  }

  // Read baseline signal (aligned with K4:472)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S4 rank %d] after ReadSignal, s0=%lu\n", myRank, s0);
  }

  // World barrier sync (aligned with K4:473)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S4 rank %d] after barrier 1, before Put+Signal loop\n", myRank);
  }

  int nInterRanks = nRanks - intraSize;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

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
                              FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)0, 2);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S4 rank %d] after Put+Signal loop\n", myRank);
  }

  // WaitSignal for s0 + nInterRanks * 2 (aligned with K4:487)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + (uint64_t)nInterRanks * 2, 64,
                          flagcxDeviceMemoryOrderAcquire);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S4 rank %d] after WaitSignal\n", myRank);
  }

  // Flush (aligned with K4:488)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S4 rank %d] after Flush\n", myRank);
  }

  // Final world barrier (aligned with K4:489)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S4 rank %d] after final barrier, kernel done\n", myRank);
  }
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] kernel start\n", myRank);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S5 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] got net\n", myRank);
  }

  size_t chunkBytes = countPerPeer * sizeof(float);
  int nInterRanks = nRanks - intraSize;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // World barrier before reading baselines (aligned with K5:521-522)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after barrier 0\n", myRank);
  }

  // Read baseline signal and counter (aligned with K5:523-524)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);
  uint64_t c0 = flagcxDevNetReadCounterS(net, (flagcxDevNetCounter_t)0, 64,
                                          flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after read baselines, s0=%lu, c0=%lu\n", myRank, s0, c0);
  }

  // World barrier sync (aligned with K5:525)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after barrier 1, Round 1 start\n", myRank);
  }

  // Round 1: Put with SignalInc + CounterInc (aligned with K5:529-536)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc_LCtrInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                     recvMemPtr, (size_t)myRank * chunkBytes,
                                     sendMemPtr, (size_t)peer * chunkBytes,
                                     chunkBytes, FLAGCX_COOP_THREAD,
                                     (flagcxDevNetSignal_t)0,
                                     (flagcxDevNetCounter_t)0);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after Round 1 put loop\n", myRank);
  }

  // WaitCounter (aligned with K5:537)
  flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetCounter_t)0,
                           c0 + (uint64_t)nInterRanks, 64,
                           flagcxDeviceMemoryOrderAcquire);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after WaitCounter\n", myRank);
  }

  // Stamp sentinel (aligned with K5:540-541)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    float *slot = (float *)flagcxGetLocalPointerS(sendMemPtr, (size_t)peer * chunkBytes);
    *slot = 999.0f;
  }

  // Barrier between rounds (aligned with K5:542)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after barrier 2, Round 2 start\n", myRank);
  }

  // Round 2: Put with SignalInc + CounterInc again (aligned with K5:546-553)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc_LCtrInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                     recvMemPtr, (size_t)myRank * chunkBytes,
                                     sendMemPtr, (size_t)peer * chunkBytes,
                                     chunkBytes, FLAGCX_COOP_THREAD,
                                     (flagcxDevNetSignal_t)0,
                                     (flagcxDevNetCounter_t)0);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after Round 2 put loop\n", myRank);
  }

  // WaitCounter for c0 + 2*nInterRanks (aligned with K5:554)
  flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetCounter_t)0,
                           c0 + 2 * (uint64_t)nInterRanks, 64,
                           flagcxDeviceMemoryOrderAcquire);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after WaitCounter round 2\n", myRank);
  }

  // WaitSignal for s0 + 2*nInterRanks (aligned with K5:555)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + 2 * (uint64_t)nInterRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after WaitSignal\n", myRank);
  }

  // Flush (aligned with K5:556)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after Flush\n", myRank);
  }

  // Final world barrier (aligned with K5:562)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S5 rank %d] after final barrier, kernel done\n", myRank);
  }
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S9 rank %d] kernel start\n", myRank);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S9 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S9 rank %d] got net\n", myRank);
  }

  int nInterRanks = nRanks - intraSize;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // World barrier before reading baseline (aligned with K9:653-654)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S9 rank %d] after barrier 0\n", myRank);
  }

  // Read baseline signal on slot 1 (aligned with K9:655)
  uint64_t s1 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S9 rank %d] after ReadSignal, s1=%lu\n", myRank, s1);
  }

  // World barrier sync (aligned with K9:656)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S9 rank %d] after barrier 1, before Signal loop\n", myRank);
  }

  // Signal loop (aligned with K9:659-662)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer != myRank && (peer < intraBase || peer >= intraBase + intraSize)) {
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S9 rank %d] after Signal loop\n", myRank);
  }

  // WaitSignal (aligned with K9:663-664)
  if (nInterRanks > 0) {
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)1,
                            s1 + (uint64_t)nInterRanks, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S9 rank %d] after WaitSignal\n", myRank);
  }

  // Final world barrier (aligned with K9:665)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S9 rank %d] after final barrier, kernel done\n", myRank);
  }
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S7 rank %d] kernel start\n", myRank);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S7 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S7 rank %d] got net\n", myRank);
  }

  int nInterRanks = nRanks - intraSize;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // World barrier before reading baseline (aligned with K7:604-605)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S7 rank %d] after barrier 0\n", myRank);
  }

  // Read baseline signal on slot 1 (aligned with K7:606)
  uint64_t s1 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S7 rank %d] after ReadSignal, s1=%lu\n", myRank, s1);
  }

  // World barrier sync (aligned with K7:607)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S7 rank %d] after barrier 1, before PutValue loop\n", myRank);
  }

  // PutValue loop (aligned with K7:608-620)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
    flagcxDevNetPutValueS_RSigInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                  recvMemPtr, putValBase + (size_t)myRank * sizeof(uint64_t),
                                  val, FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S7 rank %d] after PutValue loop\n", myRank);
  }

  // WaitSignal (aligned with K7:622-623)
  if (nInterRanks > 0) {
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)1,
                            s1 + (uint64_t)nInterRanks, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S7 rank %d] after WaitSignal\n", myRank);
  }

  // Final world barrier (aligned with K7:624)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S7 rank %d] after final barrier, kernel done\n", myRank);
  }
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S8 rank %d] kernel start\n", myRank);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S8 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S8 rank %d] got net\n", myRank);
  }

  size_t chunkBytes = countPerPeer * sizeof(float);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // World barrier (aligned with K8:975)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S8 rank %d] after barrier 0, before Get loop\n", myRank);
  }

  // Get loop (aligned with K8:981-989)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetGetS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                     sendMemPtr, (size_t)myRank * chunkBytes,
                     recvMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S8 rank %d] after Get loop, before Flush\n", myRank);
  }

  // Flush (aligned with K8:990)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S8 rank %d] after Flush\n", myRank);
  }

  // Final world barrier (aligned with K8:991)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S8 rank %d] after final barrier, kernel done\n", myRank);
  }
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S13 rank %d] kernel start\n", myRank);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S13 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S13 rank %d] got net\n", myRank);
  }

  // Test sync (aligned with K13:1496)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S13 rank %d] after sync\n", myRank);
  }

  // Test arrive + wait (split) (aligned with K13:1499-1500)
  flagcxWorldBarrierArriveS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                            flagcxDeviceMemoryOrderRelease,
                            flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S13 rank %d] after arrive\n", myRank);
  }

  flagcxWorldBarrierWaitS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderAcquire,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S13 rank %d] after wait, kernel done\n", myRank);
  }
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

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S14 rank %d] kernel start, nRanks=%d\n", myRank, nRanks);
  }

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[S14 rank %d] net is NULL\n", myRank);
    }
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S14 rank %d] got net\n", myRank);
  }

  size_t chunkBytes = countPerPeer * sizeof(float);

  // Read signal baseline (aligned with K14:210)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S14 rank %d] readSignal s0=%lu\n", myRank, s0);
  }

  // Pre-communication barrier (aligned with K14:213)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S14 rank %d] after pre-barrier\n", myRank);
  }

  // Thread-parallelized put loop (aligned with K14:217-221)
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    flagcxDevNetPutS_RSigInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                             recvMemPtr, (size_t)myRank * chunkBytes,
                             sendMemPtr, (size_t)peer * chunkBytes,
                             chunkBytes, FLAGCX_COOP_THREAD,
                             (flagcxDevNetSignal_t)0);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S14 rank %d] after put loop\n", myRank);
  }

  // Wait for all incoming signals (aligned with K14:223)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + (uint64_t)nRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S14 rank %d] after waitSignal\n", myRank);
  }

  // Flush to ensure data visibility (aligned with K14:224)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S14 rank %d] after flush\n", myRank);
  }

  // Post-communication barrier (aligned with K14:227)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[S14 rank %d] after post-barrier, kernel done\n", myRank);
  }
}

void launchKernelNetOneSidedAlltoAllS(const void *devCommPtr,
                                      const void *sendMemPtr,
                                      const void *recvMemPtr,
                                      size_t countPerPeer,
                                      flagcxStream_t stream) {
  kernelNetOneSidedAlltoAllS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(
      devCommPtr, sendMemPtr, recvMemPtr, countPerPeer);
}
