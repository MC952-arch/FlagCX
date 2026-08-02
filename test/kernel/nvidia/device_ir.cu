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
  kernelNetGetFromCommS<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
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
  kernelNetResetS<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
}

// ===========================================================================
// S3–S8: One-sided transport kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// S11: WaitSignalS + FlushS (standalone)
// Each rank signals all inter peers, waits for signals from all inter peers.
// ---------------------------------------------------------------------------

__global__ void kernelNetWaitSignalFlushS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  // Reset signal slot 0 (single-thread op)
  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)0);
  }

  // Barrier: ensure all ranks have reset before anyone reads or signals
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: read baseline
  uint64_t s0 = 0;
  int nInterPeers = nRanks - intraSize;
  if (threadIdx.x == 0) {
    s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                 flagcxDeviceMemoryOrderRelaxed);
  }

  // Barrier: ensure all ranks have read s0 before anyone starts signaling
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: signal, wait, flush
  if (threadIdx.x == 0) {
    // Signal all inter peers
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)0);
    }

    // Wait for signals from all inter peers
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)0,
                            s0 + (uint64_t)nInterPeers, 64,
                            flagcxDeviceMemoryOrderAcquire);

    // Flush
    flagcxDevNetFlushS(net, FLAGCX_COOP_THREAD, flagcxDeviceMemoryOrderRelaxed);
  }
}

void launchKernelNetWaitSignalFlushS(const void *devCommPtr,
                                     flagcxStream_t stream) {
  kernelNetWaitSignalFlushS<<<1, 32, 0, stream->base>>>(devCommPtr);
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

__global__ void kernelInterBarrierStress(const void *devCommPtr,
                                         int *devResults, int nIters) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
      devResults[0] = -1; // no net context
    return;
  }

  for (int i = 0; i < nIters; i++) {
    flagcxInterBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0,
                            flagcxDeviceMemoryOrderAcqRel,
                            flagcxDevNetFenceLevel::Relaxed);
    __syncthreads();
  }

  if (threadIdx.x == 0 && blockIdx.x == 0)
    devResults[0] = 1; // success
}

void launchKernelInterBarrierS(const void *devCommPtr, int *devResults,
                               int nIters, flagcxStream_t stream) {
  kernelInterBarrierStress<<<1, 128, 0, stream->base>>>(devCommPtr, devResults,
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
  if (!net) return;

  // Reset signals and world barrier before reading baseline
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)0);
  }
  __syncthreads();
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  size_t chunkBytes = countPerPeer * sizeof(float);
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  // World barrier after reading baseline to prevent race
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // Put with no signal/counter (None, None)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                     recvMemPtr, (size_t)myRank * chunkBytes,
                     sendMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
  }

  // Flush BEFORE signaling (the "decouple" — ensures put data is visible)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  // Signal all inter peers (separate from put)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)0);
    }
  }
  __syncthreads();

  // Wait for signals from all inter peers
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + (uint64_t)(nRanks - intraSize), 64,
                          flagcxDeviceMemoryOrderAcquire);

  // Flush after wait
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);
}

void launchKernelNetFlushDecoupleS(const void *devCommPtr,
                                   const void *sendMemPtr,
                                   const void *recvMemPtr, size_t countPerPeer,
                                   flagcxStream_t stream) {
  kernelNetFlushDecoupleS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
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
  if (!net) return;

  // Reset + world barrier before reading baseline signal
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)0);
  }
  __syncthreads();
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  size_t chunkBytes = countPerPeer * sizeof(float);
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  // World barrier after reading baseline to prevent race
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                             recvMemPtr, (size_t)myRank * chunkBytes,
                             sendMemPtr, (size_t)peer * chunkBytes,
                             chunkBytes, FLAGCX_COOP_THREAD,
                             (flagcxDevNetSignal_t)0);
  }

  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + (uint64_t)(nRanks - intraSize), 64,
                          flagcxDeviceMemoryOrderAcquire);

  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);
}

void launchKernelNetPutSignalIncS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream) {
  kernelNetPutSignalIncS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
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
  if (!net) return;

  // Reset + world barrier before reading baseline signal
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)0);
  }
  __syncthreads();
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  size_t chunkBytes = countPerPeer * sizeof(float);
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);
  int nInterPeers = nRanks - intraSize;

  // World barrier after reading baseline to prevent race
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigAdd(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                             recvMemPtr, (size_t)myRank * chunkBytes,
                             sendMemPtr, (size_t)peer * chunkBytes,
                             chunkBytes, FLAGCX_COOP_THREAD,
                             (flagcxDevNetSignal_t)0, 2);
  }

  // Each inter peer adds 2, so wait for nInterPeers * 2
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + (uint64_t)nInterPeers * 2, 64,
                          flagcxDeviceMemoryOrderAcquire);

  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);
}

void launchKernelNetPutSignalAddS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream) {
  kernelNetPutSignalAddS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
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
  if (!net) return;

  // Reset signals + counter and world barrier before reading baselines
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)0);
    flagcxDevNetResetCounter(net, (flagcxDevNetCounter_t)0);
  }
  __syncthreads();
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  size_t chunkBytes = countPerPeer * sizeof(float);
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);
  uint64_t c0 = flagcxDevNetReadCounterS(net, (flagcxDevNetCounter_t)0, 64,
                                          flagcxDeviceMemoryOrderRelaxed);
  int nInterPeers = nRanks - intraSize;

  // World barrier after reading baselines to prevent race
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc_LCtrInc(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                     recvMemPtr, (size_t)myRank * chunkBytes,
                                     sendMemPtr, (size_t)peer * chunkBytes,
                                     chunkBytes, FLAGCX_COOP_THREAD,
                                     (flagcxDevNetSignal_t)0,
                                     (flagcxDevNetCounter_t)0);
  }

  // Wait for remote signals (from peers putting to us)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + (uint64_t)nInterPeers, 64,
                          flagcxDeviceMemoryOrderAcquire);

  // Wait for local counter (our puts completed)
  flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetCounter_t)0,
                           c0 + (uint64_t)nInterPeers, 64,
                           flagcxDeviceMemoryOrderAcquire);

  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);
}

void launchKernelNetCounterPipelineS(const void *devCommPtr,
                                     const void *sendMemPtr,
                                     const void *recvMemPtr,
                                     size_t countPerPeer,
                                     flagcxStream_t stream) {
  kernelNetCounterPipelineS<<<1, 128, 0, stream->base>>>(
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
  if (!net) return;

  int nInterPeers = nRanks - intraSize;

  // --- Part 1: SignalSigInc ---
  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)1);
  }
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  uint64_t s0 = 0;
  if (threadIdx.x == 0) {
    s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                 flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0) {
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1);
    }
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1,
                            s0 + (uint64_t)nInterPeers, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }

  // --- Part 2: SignalSigAdd ---
  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)1);
  }
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0) {
    s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                 flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0) {
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigAddS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1, 5);
    }
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1,
                            s0 + (uint64_t)nInterPeers * 5, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }
}

void launchKernelNetSignalS(const void *devCommPtr, flagcxStream_t stream) {
  kernelNetSignalS<<<1, 32, 0, stream->base>>>(devCommPtr);
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
  if (!net) return;

  int nInterPeers = nRanks - intraSize;

  // === Phase 1: PutValueS(None) + SignalSigIncS + WaitSignalS ===

  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)1);
  }

  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  uint64_t s0 = 0;
  if (threadIdx.x == 0) {
    s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                 flagcxDeviceMemoryOrderRelaxed);
  }

  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0) {
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
      size_t dstOff = putValBase + (size_t)myRank * sizeof(uint64_t);
      flagcxDevNetPutValueS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                            recvMemPtr, dstOff, val, FLAGCX_COOP_THREAD);
    }

    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1);
    }

    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1,
                            s0 + (uint64_t)nInterPeers, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }

  // === Phase 2: PutValueS_RSigInc + WaitSignalS (fused) ===

  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)1);
  }

  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0) {
    s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                 flagcxDeviceMemoryOrderRelaxed);
  }

  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  if (threadIdx.x == 0) {
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
      size_t dstOff = putValBase + (size_t)myRank * sizeof(uint64_t);
      flagcxDevNetPutValueS_RSigInc(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                    recvMemPtr, dstOff, val, FLAGCX_COOP_THREAD,
                                    (flagcxDevNetSignal_t)1);
    }

    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1,
                            s0 + (uint64_t)nInterPeers, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }
}

void launchKernelNetPutValueS(const void *devCommPtr, const void *recvMemPtr,
                              size_t putValBase, flagcxStream_t stream) {
  kernelNetPutValueS<<<1, 32, 0, stream->base>>>(devCommPtr, recvMemPtr,
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
  if (!net) return;

  size_t chunkBytes = countPerPeer * sizeof(float);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    // Get: read from peer's sendBuff[myRank chunk] into our recvBuff[peer chunk]
    flagcxDevNetGetS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                     sendMemPtr, (size_t)myRank * chunkBytes,
                     recvMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
  }

  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderAcquire);
}

void launchKernelNetGetS(const void *devCommPtr, const void *sendMemPtr,
                         const void *recvMemPtr, size_t countPerPeer,
                         flagcxStream_t stream) {
  kernelNetGetS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
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

__global__ void kernelWorldBarrierS(const void *devCommPtr) {
  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  // Test sync
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Test arrive + wait (split)
  flagcxWorldBarrierArriveS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                            flagcxDeviceMemoryOrderRelease,
                            flagcxDevNetFenceLevel::Relaxed);

  flagcxWorldBarrierWaitS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderAcquire,
                          flagcxDevNetFenceLevel::Relaxed);
}

void launchKernelWorldBarrierS(const void *devCommPtr, flagcxStream_t stream) {
  kernelWorldBarrierS<<<1, 32, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S14: OneSidedAlltoAll (composite) — put + signal + wait + flush + world barrier
// Each rank puts its chunk to every inter peer using PutS_RSigInc,
// waits for signals, flushes, then world barrier for completion.
// ---------------------------------------------------------------------------

__global__ void kernelNetOneSidedAlltoAllS(const void *devCommPtr,
                                           const void *sendMemPtr,
                                           const void *recvMemPtr,
                                           size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  // Reset signal slot 0 and world barrier before reading baseline
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)0);
  }
  __syncthreads();
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  size_t chunkBytes = countPerPeer * sizeof(float);
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);
  int nInterPeers = nRanks - intraSize;

  // World barrier after reading baseline
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // Put with fused remote signal increment to all inter peers
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                             recvMemPtr, (size_t)myRank * chunkBytes,
                             sendMemPtr, (size_t)peer * chunkBytes,
                             chunkBytes, FLAGCX_COOP_THREAD,
                             (flagcxDevNetSignal_t)0);
  }

  // Wait for all incoming signals
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevNetSignal_t)0,
                          s0 + (uint64_t)nInterPeers, 64,
                          flagcxDeviceMemoryOrderAcquire);

  // Flush to ensure data visibility
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  // World barrier for global completion
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 2, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);
}

void launchKernelNetOneSidedAlltoAllS(const void *devCommPtr,
                                      const void *sendMemPtr,
                                      const void *recvMemPtr,
                                      size_t countPerPeer,
                                      flagcxStream_t stream) {
  kernelNetOneSidedAlltoAllS<<<1, 128, 0, stream->base>>>(
      devCommPtr, sendMemPtr, recvMemPtr, countPerPeer);
}
