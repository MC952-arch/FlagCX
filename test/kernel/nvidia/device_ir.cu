/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR kernel implementations — CUDA kernels exercising FlagCX
 * Device API IR functions via device pointers.
 *
 * Covers both:
 *   - Struct-based API: K1–K8
 *   - S-suffixed (scalar) API:       S1–S10
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

// ---------------------------------------------------------------------------
// K1: Comm Queries
// ---------------------------------------------------------------------------

__global__ void kernelCommQueries(const void *devCommPtr, int *results) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    results[0] = flagcxDevCommGetRank(devCommPtr);
    results[1] = flagcxDevCommGetSize(devCommPtr);
    results[2] = flagcxDevCommGetIntraRank(devCommPtr);
    results[3] = flagcxDevCommGetIntraSize(devCommPtr);
  }
}

void launchKernelCommQueries(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream) {
  kernelCommQueries<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// K2: Cooperative Group
// ---------------------------------------------------------------------------

__global__ void kernelCoopGroup(const void *devCommPtr, int *results) {
  flagcxCoopAny coop;
  flagcxCoopAnyInitBlock(&coop);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  results[tid * 2 + 0] = flagcxCoopThreadRankC(&coop);
  results[tid * 2 + 1] = flagcxCoopSizeC(&coop);

  flagcxCoopSyncC(&coop);
}

void launchKernelCoopGroup(const void *devCommPtr, int *devResults,
                           int nBlocks, int nThreads, flagcxStream_t stream) {
  kernelCoopGroup<<<nBlocks, nThreads, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// K3: Team Queries
// ---------------------------------------------------------------------------

__global__ void kernelTeamQueries(const void *devCommPtr, int *results) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    flagcxTeam teamIntra;
    flagcxGetTeamIntra(devCommPtr, &teamIntra);

    int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
    int worldRank = flagcxTeamRankToWorldC(devCommPtr, &teamIntra, intraRank);

    results[0] = intraRank;
    results[1] = worldRank;
  }
}

void launchKernelTeamQueries(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream) {
  kernelTeamQueries<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// K4: Local Pointer
// ---------------------------------------------------------------------------

__global__ void kernelLocalPointer(const void *devMemPtr, void *rawBuff,
                                   int *results) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void *localPtr = flagcxGetLocalPointerC(devMemPtr, 0);
    results[0] = (localPtr == rawBuff) ? 1 : 0;
    results[1] = (uintptr_t)localPtr & 0xFFFFFFFF;
    results[2] = ((uintptr_t)localPtr >> 32) & 0xFFFFFFFF;
  }
}

void launchKernelLocalPointer(const void *devMemPtr, void *rawBuff,
                              int *devResults, flagcxStream_t stream) {
  kernelLocalPointer<<<1, 1, 0, stream->base>>>(devMemPtr, rawBuff, devResults);
}

// ---------------------------------------------------------------------------
// K5: Intra Pointer (LSA read)
// ---------------------------------------------------------------------------

__global__ void kernelIntraPointer(const void *devCommPtr,
                                   const void *devMemPtr, float *output) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;

  size_t offset = tid * sizeof(float);
  float *peerPtr = (float *)flagcxGetIntraPointerC(devMemPtr, offset, peer);
  output[tid] = *peerPtr;
}

void launchKernelIntraPointer(const void *devCommPtr, const void *devMemPtr,
                              float *devOutput, int nBlocks, int nThreads,
                              flagcxStream_t stream) {
  kernelIntraPointer<<<nBlocks, nThreads, 0, stream->base>>>(devCommPtr, devMemPtr,
                                                       devOutput);
}

// ---------------------------------------------------------------------------
// K6: Data Type Size
// ---------------------------------------------------------------------------

__global__ void kernelDataTypeSize(int *results) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    results[0] = (int)flagcxDataTypeSizeDevice(flagcxFloat);
    results[1] = (int)flagcxDataTypeSizeDevice(flagcxHalf);
    results[2] = (int)flagcxDataTypeSizeDevice(flagcxDouble);
    results[3] = (int)flagcxDataTypeSizeDevice(flagcxInt32);
    results[4] = (int)flagcxDataTypeSizeDevice(flagcxUint64);
  }
}

void launchKernelDataTypeSize(int *devResults, flagcxStream_t stream) {
  kernelDataTypeSize<<<1, 1, 0, stream->base>>>(devResults);
}

// ---------------------------------------------------------------------------
// K7: Intra Barrier (Sync)
// ---------------------------------------------------------------------------

__global__ void kernelIntraBarrierSync(const void *devCommPtr,
                                       const void *devMemPtr, float *buffer,
                                       float *output, int N) {
  flagcxCoopAny coop;
  flagcxCoopAnyInitBlock(&coop);

  flagcxTeam teamIntra;
  flagcxGetTeamIntra(devCommPtr, &teamIntra);

  flagcxIntraBarrierSession_C session;
  flagcxIntraBarrierSessionInit(&session, &coop, devCommPtr, &teamIntra,
                                blockIdx.x, false);

  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    buffer[tid] = (float)myRank;
  }

  flagcxIntraBarrierSessionSync(&session, flagcxDeviceMemoryOrderRelease);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  if (tid < N) {
    size_t offset = tid * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerC(devMemPtr, offset, peer);
    output[tid] = *peerPtr;
  }

  flagcxIntraBarrierSessionSync(&session, flagcxDeviceMemoryOrderAcquire);
}

void launchKernelIntraBarrierSync(const void *devCommPtr,
                                  const void *devMemPtr, float *buffer,
                                  float *output, int N, flagcxStream_t stream) {
  kernelIntraBarrierSync<<<4, 256, 0, stream->base>>>(devCommPtr, devMemPtr, buffer,
                                                output, N);
}

// ---------------------------------------------------------------------------
// K8: Intra Barrier Arrive/Wait
// ---------------------------------------------------------------------------

__global__ void kernelIntraBarrierArriveWait(const void *devCommPtr,
                                             const void *devMemPtr,
                                             float *buffer, float *output,
                                             int N) {
  flagcxCoopAny coop;
  flagcxCoopAnyInitBlock(&coop);

  flagcxTeam teamIntra;
  flagcxGetTeamIntra(devCommPtr, &teamIntra);

  flagcxIntraBarrierSession_C session;
  flagcxIntraBarrierSessionInit(&session, &coop, devCommPtr, &teamIntra,
                                blockIdx.x, false);

  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    buffer[tid] = (float)(myRank + 100);
  }

  flagcxIntraBarrierSessionArrive(&session, flagcxDeviceMemoryOrderRelease);
  flagcxIntraBarrierSessionWait(&session, flagcxDeviceMemoryOrderAcquire);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  if (tid < N) {
    size_t offset = tid * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerC(devMemPtr, offset, peer);
    output[tid] = *peerPtr;
  }

  flagcxIntraBarrierSessionSync(&session, flagcxDeviceMemoryOrderAcquire);
}

void launchKernelIntraBarrierArriveWait(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream) {
  kernelIntraBarrierArriveWait<<<4, 256, 0, stream->base>>>(devCommPtr, devMemPtr,
                                                      buffer, output, N);
}

// ===========================================================================
// Scalar IR (S-suffixed) kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// S1: Cooperative Group (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelScalarCoopGroup(const void *devCommPtr, int *results) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  results[tid * 2 + 0] = flagcxCoopThreadRankS(FLAGCX_COOP_BLOCK);
  results[tid * 2 + 1] = flagcxCoopSizeS(FLAGCX_COOP_BLOCK);

  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
}

void launchKernelCoopGroupS(const void *devCommPtr, int *devResults,
                                 int nBlocks, int nThreads,
                                 flagcxStream_t stream) {
  kernelScalarCoopGroup<<<nBlocks, nThreads, 0, stream->base>>>(devCommPtr,
                                                                 devResults);
}

// ---------------------------------------------------------------------------
// S2: Team Queries (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelScalarTeamQueries(const void *devCommPtr, int *results) {
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
  kernelScalarTeamQueries<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// S3: Local Pointer (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelScalarLocalPointer(const void *devMemPtr, void *rawBuff,
                                         int *results) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void *localPtr = flagcxGetLocalPointerS(devMemPtr, 0);
    results[0] = (localPtr == rawBuff) ? 1 : 0;
  }
}

void launchKernelLocalPointerS(const void *devMemPtr, void *rawBuff,
                                    int *devResults, flagcxStream_t stream) {
  kernelScalarLocalPointer<<<1, 1, 0, stream->base>>>(devMemPtr, rawBuff,
                                                       devResults);
}

// ---------------------------------------------------------------------------
// S4: Intra Pointer (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelScalarIntraPointer(const void *devCommPtr,
                                         const void *devMemPtr,
                                         float *output) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t offset = tid * sizeof(float);

  float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
  output[tid] = *peerPtr;
}

void launchKernelIntraPointerS(const void *devCommPtr,
                                    const void *devMemPtr, float *devOutput,
                                    int nBlocks, int nThreads,
                                    flagcxStream_t stream) {
  kernelScalarIntraPointer<<<nBlocks, nThreads, 0, stream->base>>>(
      devCommPtr, devMemPtr, devOutput);
}

// ---------------------------------------------------------------------------
// S5: Intra Barrier Sync (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelScalarIntraBarrierSync(const void *devCommPtr,
                                             const void *devMemPtr,
                                             float *buffer, float *output,
                                             int N) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    buffer[tid] = (float)(myRank + 1);
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderAcqRel);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  if (tid < N) {
    size_t offset = tid * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
    output[tid] = *peerPtr;
  }
}

void launchKernelIntraBarrierSyncS(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream) {
  kernelScalarIntraBarrierSync<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, buffer, output, N);
}

// ---------------------------------------------------------------------------
// S6: SyncS(Release) + read + SyncS(Acquire)
// ---------------------------------------------------------------------------

__global__ void kernelScalarIntraBarrierSyncSplit(const void *devCommPtr,
                                                  const void *devMemPtr,
                                                  float *buffer, float *output,
                                                  int N) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    buffer[tid] = (float)(myRank + 500);
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderRelease);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  if (tid < N) {
    size_t offset = tid * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
    output[tid] = *peerPtr;
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderAcquire);
}

void launchKernelIntraBarrierSyncSplitS(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream) {
  kernelScalarIntraBarrierSyncSplit<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, buffer, output, N);
}

// ===========================================================================
// Fix 2: Extended Coop Kinds
// ===========================================================================

// ---------------------------------------------------------------------------
// S7: TILE_SPAN — threadRankEx / sizeEx / syncEx
// ---------------------------------------------------------------------------

__global__ void kernelCoopTileSpanS(int *results) {
  // Each block: 128 threads = 4 tiles of 32
  int tileIdx = threadIdx.x / 32;
  int t0 = tileIdx;  // tile index within the block (threadRank = threadIdx.x - 32*t0)
  uint32_t nTiles = 1;
  uint32_t id = 0;

  int rank = flagcxCoopThreadRankExS(FLAGCX_COOP_TILE_SPAN, (uint32_t)t0,
                                     nTiles, id);
  int size = flagcxCoopSizeExS(FLAGCX_COOP_TILE_SPAN, (uint32_t)t0, nTiles,
                               id);

  flagcxCoopSyncExS(FLAGCX_COOP_TILE_SPAN, (uint32_t)t0, nTiles, id);

  int globalTid = threadIdx.x + blockIdx.x * blockDim.x;
  results[globalTid * 2 + 0] = rank;
  results[globalTid * 2 + 1] = size;
}

void launchKernelCoopTileSpanS(int *devResults, int nBlocks, int nThreads,
                               flagcxStream_t stream) {
  kernelCoopTileSpanS<<<nBlocks, nThreads, 0, stream->base>>>(devResults);
}

// ---------------------------------------------------------------------------
// S8: LANES — threadRankEx / sizeEx / syncEx (full warp mask)
// ---------------------------------------------------------------------------

__global__ void kernelCoopLanesS(int *results) {
  // Full warp mask — equivalent to COOP_WARP
  uint32_t laneMask = 0xFFFFFFFF;

  int rank =
      flagcxCoopThreadRankExS(FLAGCX_COOP_LANES, laneMask, 0, 0);
  int size = flagcxCoopSizeExS(FLAGCX_COOP_LANES, laneMask, 0, 0);

  flagcxCoopSyncExS(FLAGCX_COOP_LANES, laneMask, 0, 0);

  int tid = threadIdx.x;
  results[tid * 2 + 0] = rank;
  results[tid * 2 + 1] = size;
}

void launchKernelCoopLanesS(int *devResults, flagcxStream_t stream) {
  kernelCoopLanesS<<<1, 32, 0, stream->base>>>(devResults);
}

// ===========================================================================
// Fix 3: S-API Transport Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// S9: GetFromCommS — verify transport handle non-null
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
// S10: Signal/Counter local read/reset/shadow
// ---------------------------------------------------------------------------

__global__ void kernelNetSignalCounterS(const void *devCommPtr, int *results) {
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

void launchKernelNetSignalCounterS(const void *devCommPtr, int *devResults,
                                   flagcxStream_t stream) {
  kernelNetSignalCounterS<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
}

// ===========================================================================
// S-API Inter-Node Transport Kernels (S11-S27)
// ===========================================================================

// ---------------------------------------------------------------------------
// S11: WaitSignalS + FlushS
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
// S12: WaitCounterS (COMMENTED — standalone SignalCtrIncS is not supported
// by the GIN protocol. Counters are a local-action mechanism incremented as
// part of put() completion. Counter wait is tested in S17 via PutS_RSigInc_LCtrInc.)
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
// S13: WaitSignalMeetShadowS
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
// S25: Inter-Barrier Test
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

void launchKernelInterBarrierStress(const void *devCommPtr, int *devResults,
                                    int nIters, flagcxStream_t stream) {
  kernelInterBarrierStress<<<1, 128, 0, stream->base>>>(devCommPtr, devResults,
                                                         nIters);
}

// ---------------------------------------------------------------------------
// S14: PutS (None, None) + SignalSigIncS + WaitSignalS + FlushS
// AlltoAll: each rank puts its chunk to every inter peer, signals, waits.
// ---------------------------------------------------------------------------

__global__ void kernelNetPutS(const void *devCommPtr, const void *sendMemPtr,
                              const void *recvMemPtr, size_t countPerPeer) {
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

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                     recvMemPtr, (size_t)myRank * chunkBytes,
                     sendMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
  }
  __syncthreads();

  // Signal all inter peers
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

  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);
}

void launchKernelNetPutS(const void *devCommPtr, const void *sendMemPtr,
                         const void *recvMemPtr, size_t countPerPeer,
                         flagcxStream_t stream) {
  kernelNetPutS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                              recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S15: PutS_RSigInc + WaitSignalS + FlushS
// AlltoAll with fused remote signal increment.
// ---------------------------------------------------------------------------

__global__ void kernelNetPutRSigIncS(const void *devCommPtr,
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

void launchKernelNetPutRSigIncS(const void *devCommPtr, const void *sendMemPtr,
                                const void *recvMemPtr, size_t countPerPeer,
                                flagcxStream_t stream) {
  kernelNetPutRSigIncS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                                     recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S16: PutS_RSigAdd + WaitSignalS + FlushS
// AlltoAll with remote signal add (value = 1 per peer).
// ---------------------------------------------------------------------------

__global__ void kernelNetPutRSigAddS(const void *devCommPtr,
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

void launchKernelNetPutRSigAddS(const void *devCommPtr, const void *sendMemPtr,
                                const void *recvMemPtr, size_t countPerPeer,
                                flagcxStream_t stream) {
  kernelNetPutRSigAddS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                                     recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S17: PutS_RSigInc_LCtrInc + WaitSignalS + WaitCounterS + FlushS
// AlltoAll with both remote signal inc and local counter inc.
// ---------------------------------------------------------------------------

__global__ void kernelNetPutRSigLCtrS(const void *devCommPtr,
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

void launchKernelNetPutRSigLCtrS(const void *devCommPtr, const void *sendMemPtr,
                                 const void *recvMemPtr, size_t countPerPeer,
                                 flagcxStream_t stream) {
  kernelNetPutRSigLCtrS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                                      recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S18: SignalSigIncS + WaitSignalS
// Each rank increments signal on next peer, waits for signal from prev.
// ---------------------------------------------------------------------------

__global__ void kernelNetSignalSigIncS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  // Reset signal slot 1 (single-thread)
  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)1);
  }

  // All threads participate in world barrier
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: read baseline
  uint64_t s0 = 0;
  if (threadIdx.x == 0) {
    s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                 flagcxDeviceMemoryOrderRelaxed);
  }

  // Second world barrier: ensure all ranks have read s0 before signaling
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: signal and wait
  if (threadIdx.x == 0) {
    int nInterPeers = nRanks - intraSize;

    // Signal all inter peers
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1);
    }

    // Wait for signals from all inter peers
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1,
                            s0 + (uint64_t)nInterPeers, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }
}

void launchKernelNetSignalSigIncS(const void *devCommPtr,
                                  flagcxStream_t stream) {
  kernelNetSignalSigIncS<<<1, 32, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S19: SignalSigAddS + WaitSignalS
// Each rank adds value to signal on next peer, waits for signal from prev.
// ---------------------------------------------------------------------------

__global__ void kernelNetSignalSigAddS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  // Reset signal slot 1 (single-thread)
  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)1);
  }

  // All threads participate in world barrier
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: read baseline
  uint64_t s0 = 0;
  if (threadIdx.x == 0) {
    s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                 flagcxDeviceMemoryOrderRelaxed);
  }

  // Second world barrier: ensure all ranks have read s0 before signaling
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: signal and wait
  if (threadIdx.x == 0) {
    int nInterPeers = nRanks - intraSize;

    // Add value to all inter peers' signal slot
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigAddS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1, 5);
    }

    // Wait for signals from all inter peers (each adds 5)
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1,
                            s0 + (uint64_t)nInterPeers * 5, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }
}

void launchKernelNetSignalSigAddS(const void *devCommPtr,
                                  flagcxStream_t stream) {
  kernelNetSignalSigAddS<<<1, 32, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S21: PutValueS (None) + SignalSigIncS + WaitSignalS
// Each rank writes uint64_t value = myRank*1000 + peer to peer's recv area.
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

  // Reset signal slot 1 (single-thread)
  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)1);
  }

  // All threads participate in world barrier
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: read baseline
  uint64_t s0 = 0;
  if (threadIdx.x == 0) {
    s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                 flagcxDeviceMemoryOrderRelaxed);
  }

  // Second world barrier: ensure all ranks have read s0 before signaling
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: put values, signal, wait
  if (threadIdx.x == 0) {
    int nInterPeers = nRanks - intraSize;

    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
      size_t dstOff = putValBase + (size_t)myRank * sizeof(uint64_t);
      flagcxDevNetPutValueS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                            recvMemPtr, dstOff, val, FLAGCX_COOP_THREAD);
    }

    // Signal all inter peers
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_INTER, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevNetSignal_t)1);
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
// S22: PutValueS_RSigInc + WaitSignalS
// Same as S21 but uses fused putValue + signal increment.
// ---------------------------------------------------------------------------

__global__ void kernelNetPutValueRSigS(const void *devCommPtr,
                                       const void *recvMemPtr,
                                       size_t putValBase) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  // Reset signal slot 1 (single-thread)
  if (threadIdx.x == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevNetSignal_t)1);
  }

  // All threads participate in world barrier
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: read baseline
  uint64_t s0 = 0;
  if (threadIdx.x == 0) {
    s0 = flagcxDevNetReadSignalS(net, (flagcxDevNetSignal_t)1, 64,
                                 flagcxDeviceMemoryOrderRelaxed);
  }

  // Second world barrier: ensure all ranks have read s0 before signaling
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: put values with signal, wait
  if (threadIdx.x == 0) {
    int nInterPeers = nRanks - intraSize;

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

void launchKernelNetPutValueRSigS(const void *devCommPtr,
                                  const void *recvMemPtr, size_t putValBase,
                                  flagcxStream_t stream) {
  kernelNetPutValueRSigS<<<1, 32, 0, stream->base>>>(devCommPtr, recvMemPtr,
                                                      putValBase);
}

// ---------------------------------------------------------------------------
// S23: GetS + FlushS
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
// S24: Two-sided (COMMENTED per user request)
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
// ---------------------------------------------------------------------------
// S26: WorldBarrierSyncS
// ---------------------------------------------------------------------------

__global__ void kernelWorldBarrierS(const void *devCommPtr) {
  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);
}

void launchKernelWorldBarrierS(const void *devCommPtr, flagcxStream_t stream) {
  kernelWorldBarrierS<<<1, 32, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S27: WorldBarrierArriveS + WorldBarrierWaitS
// ---------------------------------------------------------------------------

__global__ void kernelWorldBarrierSplitS(const void *devCommPtr) {
  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  flagcxWorldBarrierArriveS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                            flagcxDeviceMemoryOrderRelease,
                            flagcxDevNetFenceLevel::Relaxed);

  flagcxWorldBarrierWaitS(net, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderAcquire,
                          flagcxDevNetFenceLevel::Relaxed);
}

void launchKernelWorldBarrierSplitS(const void *devCommPtr,
                                    flagcxStream_t stream) {
  kernelWorldBarrierSplitS<<<1, 32, 0, stream->base>>>(devCommPtr);
}
