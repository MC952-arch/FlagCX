/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR kernel implementations — CUDA kernels exercising FlagCX
 * Device API IR functions via device pointers.
 *
 * Covers both:
 *   - C-suffixed (struct-based) API: K1–K8
 *   - S-suffixed (scalar) API:       S1–S6
 *
 * Compiled by nvcc into device_ir.o, linked by g++ into test_device_ir.
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#if defined(USE_DU_ADAPTOR)
#include "du_adaptor.h"
#else
#include "nvidia_adaptor.h"
#endif
#include "flagcx_device_internal.h"

// IR wrapper declarations + implementations (needed for nvcc inline compilation)
#include "flagcx_device_wrapper.h"
#include "flagcx_device_wrapper_impl.h"

// Scalar IR declarations + implementations
#include "flagcx_device_scalar_ir.h"
#include "flagcx_device_scalar_ir_impl.h"

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
  flagcxCoopAnyInitBlockC(&coop);

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
    flagcxGetTeamIntraC(devCommPtr, &teamIntra);

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
  flagcxCoopAnyInitBlockC(&coop);

  flagcxTeam teamIntra;
  flagcxGetTeamIntraC(devCommPtr, &teamIntra);

  flagcxIntraBarrierSession_C session;
  flagcxIntraBarrierSessionInitC(&session, &coop, devCommPtr, &teamIntra,
                                blockIdx.x, false);

  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    buffer[tid] = (float)myRank;
  }

  flagcxIntraBarrierSessionSyncC(&session, flagcxDeviceMemoryOrderRelease);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  if (tid < N) {
    size_t offset = tid * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerC(devMemPtr, offset, peer);
    output[tid] = *peerPtr;
  }

  flagcxIntraBarrierSessionSyncC(&session, flagcxDeviceMemoryOrderAcquire);
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
  flagcxCoopAnyInitBlockC(&coop);

  flagcxTeam teamIntra;
  flagcxGetTeamIntraC(devCommPtr, &teamIntra);

  flagcxIntraBarrierSession_C session;
  flagcxIntraBarrierSessionInitC(&session, &coop, devCommPtr, &teamIntra,
                                blockIdx.x, false);

  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    buffer[tid] = (float)(myRank + 100);
  }

  flagcxIntraBarrierSessionArriveC(&session, flagcxDeviceMemoryOrderRelease);
  flagcxIntraBarrierSessionWaitC(&session, flagcxDeviceMemoryOrderAcquire);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  if (tid < N) {
    size_t offset = tid * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerC(devMemPtr, offset, peer);
    output[tid] = *peerPtr;
  }

  flagcxIntraBarrierSessionSyncC(&session, flagcxDeviceMemoryOrderAcquire);
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
// S6: Intra Barrier Arrive/Wait (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelScalarIntraBarrierArriveWait(const void *devCommPtr,
                                                   const void *devMemPtr,
                                                   float *buffer,
                                                   float *output, int N) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    buffer[tid] = (float)(myRank + 100);
  }

  flagcxIntraBarrierArriveS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                            flagcxDeviceMemoryOrderRelease);
  flagcxIntraBarrierWaitS(devCommPtr, FLAGCX_COOP_BLOCK, blockIdx.x, false,
                          flagcxDeviceMemoryOrderAcquire);

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

void launchKernelIntraBarrierArriveWaitS(const void *devCommPtr,
                                              const void *devMemPtr,
                                              float *buffer, float *output,
                                              int N, flagcxStream_t stream) {
  kernelScalarIntraBarrierArriveWait<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, buffer, output, N);
}
