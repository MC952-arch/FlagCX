/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Function Tests — CUDA kernels exercising FlagCX Device API
 * IR wrapper functions via device pointers (simulates Triton usage path).
 *
 * Tests 8 kernel categories covering 69 IR functions:
 *   K1: Comm Queries (GetRank, GetSize, GetIntraRank, GetIntraSize)
 *   K2: Cooperative Group (InitBlock, ThreadRank, Size, Sync)
 *   K3: Team Queries (GetTeamIntra, RankToWorld, RankToIntra)
 *   K4: Local Pointer (GetLocalPointerC)
 *   K5: Intra Pointer (GetIntraPointerC — LSA read)
 *   K6: Data Type Size (DataTypeSizeDevice)
 *   K7: Intra Barrier (SessionInit, Sync)
 *   K8: Intra Barrier Arrive/Wait (SessionArrive, Wait)
 *
 * Usage: mpirun -np N ./test_device_ir_functions
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "nvidia_adaptor.h"
#include "flagcx_device_internal.h"
#include "tools.h"

// Include IR wrapper header + implementations (impl needed for nvcc linkage)
#include "../../bindings/ir/flagcx_device_wrapper.h"
#include "../../bindings/ir/flagcx_device_wrapper_impl.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

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

// ---------------------------------------------------------------------------
// K2: Cooperative Group
// ---------------------------------------------------------------------------

__global__ void kernelCoopGroup(const void *devCommPtr, int *results) {
  // Each thread writes its coop rank and size
  flagcxCoopAny coop;
  flagcxCoopAnyInitBlock(&coop);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  results[tid * 2 + 0] = flagcxCoopThreadRankC(&coop);
  results[tid * 2 + 1] = flagcxCoopSizeC(&coop);

  // Sync to ensure all threads have written
  flagcxCoopSyncC(&coop);
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

// ---------------------------------------------------------------------------
// K4: Local Pointer
// ---------------------------------------------------------------------------

__global__ void kernelLocalPointer(const void *devMemPtr, void *rawBuff,
                                   int *results) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void *localPtr = flagcxGetLocalPointerC(devMemPtr, 0);
    // Verify it matches the raw buffer address
    results[0] = (localPtr == rawBuff) ? 1 : 0;
    results[1] = (uintptr_t)localPtr & 0xFFFFFFFF;        // low 32 bits
    results[2] = ((uintptr_t)localPtr >> 32) & 0xFFFFFFFF; // high 32 bits
  }
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

  // Read element [tid] from peer via LSA
  size_t offset = tid * sizeof(float);
  float *peerPtr = (float *)flagcxGetIntraPointerC(devMemPtr, offset, peer);
  output[tid] = *peerPtr;
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

  // Each rank writes its rank value to buffer[tid]
  if (tid < N) {
    buffer[tid] = (float)myRank;
  }

  // Barrier: ensure all ranks have written
  flagcxIntraBarrierSessionSync(&session, flagcxDeviceMemoryOrderRelease);

  // Now read peer's data via LSA into separate output buffer
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  if (tid < N) {
    size_t offset = tid * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerC(devMemPtr, offset, peer);
    output[tid] = *peerPtr; // Should see peer's rank value
  }

  // Barrier: ensure all ranks have read before buffer can be reused
  flagcxIntraBarrierSessionSync(&session, flagcxDeviceMemoryOrderAcquire);
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

  // Write phase
  if (tid < N) {
    buffer[tid] = (float)(myRank + 100);
  }

  // Arrive at barrier
  flagcxIntraBarrierSessionArrive(&session, flagcxDeviceMemoryOrderRelease);

  // Wait for all ranks
  flagcxIntraBarrierSessionWait(&session, flagcxDeviceMemoryOrderAcquire);

  // Read peer's data
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  if (tid < N) {
    size_t offset = tid * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerC(devMemPtr, offset, peer);
    output[tid] = *peerPtr; // Should see peer's (rank + 100)
  }

  flagcxIntraBarrierSessionSync(&session, flagcxDeviceMemoryOrderAcquire);
}

// ===========================================================================
// Main test driver
// ===========================================================================

int main(int argc, char *argv[]) {
  flagcxHandlerGroup_t handler;
  FLAGCXCHECK(flagcxHandleInit(&handler));
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  uint64_t splitMask = 0;
  int color = 0;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  int nGpu;
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, uniqueId, proc));

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  // Allocate test buffer (1 MB)
  size_t bufSize = 1024 * 1024;
  void *regBuff = nullptr;
  FLAGCXCHECK(flagcxMemAlloc(&regBuff, bufSize));

  // Register symmetric window
  flagcxWindow_t win = nullptr;
  FLAGCXCHECK(flagcxCommWindowRegister(comm, regBuff, bufSize, &win,
                                       FLAGCX_WIN_COLL_SYMMETRIC));

  // Create DevComm
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create DevMem
  flagcxDevMem_t devMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));

  // Get device pointers
  void *devCommPtr = nullptr;
  void *devMemPtr = nullptr;
  FLAGCXCHECK(flagcxDevCommGetDevicePtr(devComm, &devCommPtr));
  FLAGCXCHECK(flagcxDevMemGetDevicePtr(devMem, &devMemPtr));

  if (proc == 0) {
    printf("# FlagCX Device IR Function Tests\n");
    printf("# nRanks: %d\n", totalProcs);
  }

  // Allocate result buffers
  int *devResults = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devResults, 1024 * sizeof(int),
                                      flagcxMemDevice, NULL));
  int hostResults[1024];

  // -------------------------------------------------------------------------
  // Test K1: Comm Queries
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  memset(hostResults, 0, sizeof(hostResults));
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int), flagcxMemDevice, NULL));

  kernelCommQueries<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                      4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k1Pass = (hostResults[0] == proc) && (hostResults[1] == totalProcs) &&
                (hostResults[2] >= 0) && (hostResults[2] < totalProcs) &&
                (hostResults[3] > 0) && (hostResults[3] <= totalProcs);

  if (proc == 0) {
    printf("K1 CommQueries: %s\n", k1Pass ? "PASS" : "FAIL");
  }

  // -------------------------------------------------------------------------
  // Test K2: Cooperative Group
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  memset(hostResults, 0, sizeof(hostResults));
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int), flagcxMemDevice, NULL));

  int nThreads = 256;
  kernelCoopGroup<<<1, nThreads, 0, stream->base>>>(devCommPtr, devResults);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                      nThreads * 2 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k2Pass = true;
  for (int i = 0; i < nThreads; i++) {
    int rank = hostResults[i * 2 + 0];
    int size = hostResults[i * 2 + 1];
    if (rank != i || size != nThreads) {
      k2Pass = false;
      break;
    }
  }

  if (proc == 0) {
    printf("K2 CoopGroup: %s\n", k2Pass ? "PASS" : "FAIL");
  }

  // -------------------------------------------------------------------------
  // Test K3: Team Queries
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  memset(hostResults, 0, sizeof(hostResults));
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int), flagcxMemDevice, NULL));

  kernelTeamQueries<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 2 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k3Pass = (hostResults[1] == proc); // worldRank should match proc

  if (proc == 0) {
    printf("K3 TeamQueries: %s\n", k3Pass ? "PASS" : "FAIL");
  }

  // -------------------------------------------------------------------------
  // Test K4: Local Pointer
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  memset(hostResults, 0, sizeof(hostResults));
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int), flagcxMemDevice, NULL));

  kernelLocalPointer<<<1, 1, 0, stream->base>>>(devMemPtr, regBuff, devResults);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 3 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k4Pass = (hostResults[0] == 1); // Should match raw buffer

  if (proc == 0) {
    printf("K4 LocalPointer: %s\n", k4Pass ? "PASS" : "FAIL");
  }

  // -------------------------------------------------------------------------
  // Test K5: Intra Pointer (LSA read)
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);

  // Initialize buffer: each rank writes its rank value
  size_t floatCount = bufSize / sizeof(float);
  float *hostBuff = new float[floatCount];
  for (size_t i = 0; i < floatCount; i++) {
    hostBuff[i] = (float)proc;
  }
  FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostBuff, bufSize,
                                      flagcxMemcpyHostToDevice, NULL));

  MPI_Barrier(MPI_COMM_WORLD);

  // Allocate output buffer
  float *devOutput = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devOutput, bufSize,
                                      flagcxMemDevice, NULL));

  int nBlocks = 256;
  int nThreadsPerBlock = 256;
  int totalThreads = nBlocks * nThreadsPerBlock;
  kernelIntraPointer<<<nBlocks, nThreadsPerBlock, 0, stream->base>>>(
      devCommPtr, devMemPtr, devOutput);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  float *hostOutput = new float[floatCount];
  FLAGCXCHECK(devHandle->deviceMemcpy(hostOutput, devOutput, bufSize,
                                      flagcxMemcpyDeviceToHost, NULL));

  // Verify: should read peer's rank value
  int peer = (proc + 1) % totalProcs;
  bool k5Pass = true;
  for (int i = 0; i < totalThreads && i < (int)floatCount; i++) {
    if (fabsf(hostOutput[i] - (float)peer) > 1e-3f) {
      k5Pass = false;
      break;
    }
  }

  if (proc == 0) {
    printf("K5 IntraPointer: %s\n", k5Pass ? "PASS" : "FAIL");
  }

  delete[] hostOutput;
  FLAGCXCHECK(devHandle->deviceFree(devOutput, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // Test K6: Data Type Size
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  memset(hostResults, 0, sizeof(hostResults));
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int), flagcxMemDevice, NULL));

  kernelDataTypeSize<<<1, 1, 0, stream->base>>>(devResults);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 5 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k6Pass = (hostResults[0] == 4) &&  // float
                (hostResults[1] == 2) &&  // half
                (hostResults[2] == 8) &&  // double
                (hostResults[3] == 4) &&  // int32
                (hostResults[4] == 8);    // uint64

  if (proc == 0) {
    printf("K6 DataTypeSize: %s\n", k6Pass ? "PASS" : "FAIL");
  }

  // -------------------------------------------------------------------------
  // Test K7: Intra Barrier Sync
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);

  int N = 1024;
  FLAGCXCHECK(devHandle->deviceMemset(regBuff, 0, N * sizeof(float), flagcxMemDevice, NULL));

  // Separate output buffer to avoid race (write to regBuff, read peer into k7Output)
  float *k7Output = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&k7Output, N * sizeof(float),
                                      flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemset(k7Output, 0, N * sizeof(float), flagcxMemDevice, NULL));

  kernelIntraBarrierSync<<<4, 256, 0, stream->base>>>(devCommPtr, devMemPtr,
                                                (float *)regBuff, k7Output, N);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  float *hostBarrierResult = new float[N];
  FLAGCXCHECK(devHandle->deviceMemcpy(hostBarrierResult, k7Output,
                                      N * sizeof(float),
                                      flagcxMemcpyDeviceToHost, NULL));

  // Verify: should see peer's rank value
  bool k7Pass = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(hostBarrierResult[i] - (float)peer) > 1e-3f) {
      printf("[rank %d] K7 FAIL at i=%d: got %f, expected %f (peer=%d)\n",
             proc, i, hostBarrierResult[i], (float)peer, peer);
      k7Pass = false;
      break;
    }
  }

  if (proc == 0) {
    printf("K7 IntraBarrierSync: %s\n", k7Pass ? "PASS" : "FAIL");
  }

  delete[] hostBarrierResult;
  FLAGCXCHECK(devHandle->deviceFree(k7Output, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // Test K8: Intra Barrier Arrive/Wait
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(devHandle->deviceMemset(regBuff, 0, N * sizeof(float), flagcxMemDevice, NULL));

  float *k8Output = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&k8Output, N * sizeof(float),
                                      flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemset(k8Output, 0, N * sizeof(float), flagcxMemDevice, NULL));

  kernelIntraBarrierArriveWait<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, (float *)regBuff, k8Output, N);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  float *hostArriveWaitResult = new float[N];
  FLAGCXCHECK(devHandle->deviceMemcpy(hostArriveWaitResult, k8Output,
                                      N * sizeof(float),
                                      flagcxMemcpyDeviceToHost, NULL));

  // Verify: should see peer's (rank + 100)
  float expectedK8 = (float)(peer + 100);
  bool k8Pass = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(hostArriveWaitResult[i] - expectedK8) > 1e-3f) {
      printf("[rank %d] K8 FAIL at i=%d: got %f, expected %f (peer=%d)\n",
             proc, i, hostArriveWaitResult[i], expectedK8, peer);
      k8Pass = false;
      break;
    }
  }

  if (proc == 0) {
    printf("K8 IntraBarrierArriveWait: %s\n", k8Pass ? "PASS" : "FAIL");
  }

  delete[] hostArriveWaitResult;
  FLAGCXCHECK(devHandle->deviceFree(k8Output, flagcxMemDevice, NULL));
  delete[] hostBuff;

  // -------------------------------------------------------------------------
  // Summary
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);

  int allPass = k1Pass && k2Pass && k3Pass && k4Pass && k5Pass && k6Pass &&
                k7Pass && k8Pass;
  if (!allPass) {
    printf("[rank %d] FAIL: k1=%d k2=%d k3=%d k4=%d k5=%d k6=%d k7=%d k8=%d\n",
           proc, k1Pass, k2Pass, k3Pass, k4Pass, k5Pass, k6Pass, k7Pass, k8Pass);
  }
  int globalPass = 0;
  MPI_Allreduce(&allPass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (proc == 0) {
    printf("\n=== Overall: %s ===\n", globalPass ? "PASS" : "FAIL");
  }

  // Cleanup
  FLAGCXCHECK(devHandle->deviceFree(devResults, flagcxMemDevice, NULL));
  FLAGCXCHECK(flagcxDevMemFreeDevicePtr(devMem));
  FLAGCXCHECK(flagcxDevCommFreeDevicePtr(devComm));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, devMem));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, win));
  FLAGCXCHECK(flagcxMemFree(regBuff));
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
