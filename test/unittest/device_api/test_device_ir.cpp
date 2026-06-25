/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Function Tests — host driver exercising FlagCX Device API
 * IR wrapper functions via device pointers (simulates Triton usage path).
 *
 * Tests 8 kernel categories covering C-suffixed (struct-based) IR functions:
 *   K1: Comm Queries (GetRank, GetSize, GetIntraRank, GetIntraSize)
 *   K2: Cooperative Group (InitBlock, ThreadRank, Size, Sync)
 *   K3: Team Queries (GetTeamIntra, RankToWorld, RankToIntra)
 *   K4: Local Pointer (GetLocalPointerC)
 *   K5: Intra Pointer (GetIntraPointerC — LSA read)
 *   K6: Data Type Size (DataTypeSizeDevice)
 *   K7: Intra Barrier (SessionInit, Sync)
 *   K8: Intra Barrier Arrive/Wait (SessionArrive, Wait)
 *
 * Tests 6 kernel categories covering S-suffixed (scalar) IR functions:
 *   S1: Cooperative Group (CoopThreadRankS, CoopSizeS, CoopSyncS)
 *   S2: Team Queries (TeamRankToWorldS)
 *   S3: Local Pointer (GetLocalPointerS)
 *   S4: Intra Pointer (GetIntraPointerS)
 *   S5: Intra Barrier Sync (IntraBarrierSyncS)
 *   S6: Intra Barrier Arrive/Wait (IntraBarrierArriveS, WaitS)
 *
 * Usage: mpirun -np N ./test_device_ir
 ************************************************************************/

#include "device_ir.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

// ===========================================================================
// Main test driver
// ===========================================================================

int main(int argc, char *argv[]) {
  flagcxDeviceHandle_t devHandle;
  FLAGCXCHECK(flagcxDeviceHandleInit(&devHandle));
  flagcxComm_t comm;
  flagcxUniqueId uniqueId;

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
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, &uniqueId, proc));

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
  reqs.intraBarrierCount = 4;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create DevMem
  flagcxDevMem_t devMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));

  // Get device pointers
  void *devCommPtr = nullptr;
  FLAGCXCHECK(flagcxDevCommGetDevicePtr(devComm, &devCommPtr));
  void *devMemPtr = nullptr;
  FLAGCXCHECK(flagcxDevMemGetDevicePtr(devMem, &devMemPtr));

  // Allocate results buffer
  int *devResults = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devResults, 1024 * sizeof(int),
                                      flagcxMemDevice, NULL));

  int hostResults[1024];

  // -------------------------------------------------------------------------
  // Test K1: Comm Queries
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  memset(hostResults, 0, sizeof(hostResults));
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelCommQueries(devCommPtr, devResults, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k1Pass = (hostResults[0] == proc) && (hostResults[1] == totalProcs) &&
                (hostResults[2] == proc) && // single-node: intraRank == rank
                (hostResults[3] == totalProcs);

  if (proc == 0) {
    printf("K1 CommQueries: %s (rank=%d size=%d intraRank=%d intraSize=%d)\n",
           k1Pass ? "PASS" : "FAIL", hostResults[0], hostResults[1],
           hostResults[2], hostResults[3]);
  }

  // -------------------------------------------------------------------------
  // Test K2: Cooperative Group
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  memset(hostResults, 0, sizeof(hostResults));
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int),
                                      flagcxMemDevice, NULL));

  int k2Blocks = 1, k2Threads = 32;
  launchKernelCoopGroup(devCommPtr, devResults, k2Blocks, k2Threads, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                      k2Threads * 2 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k2Pass = true;
  for (int i = 0; i < k2Threads; i++) {
    if (hostResults[i * 2] != i || hostResults[i * 2 + 1] != k2Threads) {
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
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelTeamQueries(devCommPtr, devResults, stream);
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
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelLocalPointer(devMemPtr, regBuff, devResults, stream);
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
  launchKernelIntraPointer(devCommPtr, devMemPtr, devOutput, nBlocks,
                           nThreadsPerBlock, stream);
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
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 1024 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelDataTypeSize(devResults, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 5 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k6Pass = (hostResults[0] == 4) && // float
                (hostResults[1] == 2) && // half
                (hostResults[2] == 8) && // double
                (hostResults[3] == 4) && // int32
                (hostResults[4] == 8);   // uint64

  if (proc == 0) {
    printf("K6 DataTypeSize: %s\n", k6Pass ? "PASS" : "FAIL");
  }

  // -------------------------------------------------------------------------
  // Test K7: Intra Barrier Sync
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);

  int N = 1024;
  FLAGCXCHECK(devHandle->deviceMemset(regBuff, 0, N * sizeof(float),
                                      flagcxMemDevice, NULL));

  float *k7Output = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&k7Output, N * sizeof(float),
                                      flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemset(k7Output, 0, N * sizeof(float),
                                      flagcxMemDevice, NULL));

  launchKernelIntraBarrierSync(devCommPtr, devMemPtr, (float *)regBuff,
                               k7Output, N, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  float *hostBarrierResult = new float[N];
  FLAGCXCHECK(devHandle->deviceMemcpy(hostBarrierResult, k7Output,
                                      N * sizeof(float),
                                      flagcxMemcpyDeviceToHost, NULL));

  // Verify: should see peer's rank value
  bool k7Pass = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(hostBarrierResult[i] - (float)peer) > 1e-3f) {
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

  FLAGCXCHECK(devHandle->deviceMemset(regBuff, 0, N * sizeof(float),
                                      flagcxMemDevice, NULL));

  float *k8Output = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&k8Output, N * sizeof(float),
                                      flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemset(k8Output, 0, N * sizeof(float),
                                      flagcxMemDevice, NULL));

  launchKernelIntraBarrierArriveWait(devCommPtr, devMemPtr, (float *)regBuff,
                                     k8Output, N, stream);
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

  // =========================================================================
  // Scalar IR Tests (S1 - S6)
  // =========================================================================

  if (proc == 0) {
    printf("\n--- Scalar IR Tests ---\n");
  }

  // -------------------------------------------------------------------------
  // S1: Cooperative Group (Scalar)
  // -------------------------------------------------------------------------
  int nBlocksS1 = 2, nThreadsS1 = 32;
  int totalThreadsS1 = nBlocksS1 * nThreadsS1;
  int *s1Results = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&s1Results,
                                      totalThreadsS1 * 2 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelCoopGroupS(devCommPtr, s1Results, nBlocksS1, nThreadsS1, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  int *hostS1 = new int[totalThreadsS1 * 2];
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS1, s1Results,
                                      totalThreadsS1 * 2 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool s1Pass = true;
  for (int i = 0; i < totalThreadsS1; i++) {
    int expectedRank = i % nThreadsS1; // block-level: threadIdx within block
    int expectedSize = nThreadsS1;
    if (hostS1[i * 2 + 0] != expectedRank ||
        hostS1[i * 2 + 1] != expectedSize) {
      s1Pass = false;
      break;
    }
  }
  if (proc == 0) {
    printf("S1 CoopGroup(Scalar): %s\n", s1Pass ? "PASS" : "FAIL");
  }
  delete[] hostS1;
  FLAGCXCHECK(devHandle->deviceFree(s1Results, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // S2: Team Queries (Scalar)
  // -------------------------------------------------------------------------
  int *s2Results = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&s2Results, 2 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelTeamQueriesS(devCommPtr, s2Results, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  int hostS2[2];
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS2, s2Results, 2 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  // intraRank -> worldRank via INTRA team should give back our proc rank
  bool s2Pass = (hostS2[1] == proc);
  if (proc == 0) {
    printf("S2 TeamQueries(Scalar): %s\n", s2Pass ? "PASS" : "FAIL");
  }
  FLAGCXCHECK(devHandle->deviceFree(s2Results, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // S3: Local Pointer (Scalar)
  // -------------------------------------------------------------------------
  int *s3Results = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&s3Results, sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelLocalPointerS(devMemPtr, regBuff, s3Results, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  int hostS3 = 0;
  FLAGCXCHECK(devHandle->deviceMemcpy(&hostS3, s3Results, sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool s3Pass = (hostS3 == 1);
  if (proc == 0) {
    printf("S3 LocalPointer(Scalar): %s\n", s3Pass ? "PASS" : "FAIL");
  }
  FLAGCXCHECK(devHandle->deviceFree(s3Results, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // S4: Intra Pointer (Scalar)
  // -------------------------------------------------------------------------
  // Write known pattern to local buffer first
  int nElemsS4 = 256;
  float *hostInitS4 = new float[nElemsS4];
  for (int i = 0; i < nElemsS4; i++)
    hostInitS4[i] = (float)(proc * 1000 + i);
  FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostInitS4,
                                      nElemsS4 * sizeof(float),
                                      flagcxMemcpyHostToDevice, NULL));
  MPI_Barrier(MPI_COMM_WORLD);

  float *s4Output = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc(
      (void **)&s4Output, nElemsS4 * sizeof(float), flagcxMemDevice, NULL));

  launchKernelIntraPointerS(devCommPtr, devMemPtr, s4Output, 1, nElemsS4,
                            stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  float *hostS4 = new float[nElemsS4];
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS4, s4Output,
                                      nElemsS4 * sizeof(float),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool s4Pass = true;
  for (int i = 0; i < nElemsS4; i++) {
    float expected = (float)(peer * 1000 + i);
    if (fabsf(hostS4[i] - expected) > 1e-3f) {
      s4Pass = false;
      break;
    }
  }
  if (proc == 0) {
    printf("S4 IntraPointer(Scalar): %s\n", s4Pass ? "PASS" : "FAIL");
  }
  delete[] hostInitS4;
  delete[] hostS4;
  FLAGCXCHECK(devHandle->deviceFree(s4Output, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // S5: Intra Barrier Sync (Scalar)
  // -------------------------------------------------------------------------
  int NS5 = 4 * 256;
  FLAGCXCHECK(devHandle->deviceMemset(regBuff, 0, NS5 * sizeof(float),
                                      flagcxMemDevice, NULL));

  float *s5Output = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&s5Output, NS5 * sizeof(float),
                                      flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemset(s5Output, 0, NS5 * sizeof(float),
                                      flagcxMemDevice, NULL));

  launchKernelIntraBarrierSyncS(devCommPtr, devMemPtr, (float *)regBuff,
                                s5Output, NS5, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  float *hostS5 = new float[NS5];
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS5, s5Output, NS5 * sizeof(float),
                                      flagcxMemcpyDeviceToHost, NULL));

  float expectedS5 = (float)(peer + 1);
  bool s5Pass = true;
  for (int i = 0; i < NS5; i++) {
    if (fabsf(hostS5[i] - expectedS5) > 1e-3f) {
      s5Pass = false;
      break;
    }
  }
  if (proc == 0) {
    printf("S5 IntraBarrierSync(Scalar): %s\n", s5Pass ? "PASS" : "FAIL");
  }
  delete[] hostS5;
  FLAGCXCHECK(devHandle->deviceFree(s5Output, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // S6: Intra Barrier Arrive/Wait (Scalar)
  // -------------------------------------------------------------------------
  int NS6 = 4 * 256;
  FLAGCXCHECK(devHandle->deviceMemset(regBuff, 0, NS6 * sizeof(float),
                                      flagcxMemDevice, NULL));

  float *s6Output = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&s6Output, NS6 * sizeof(float),
                                      flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemset(s6Output, 0, NS6 * sizeof(float),
                                      flagcxMemDevice, NULL));

  launchKernelIntraBarrierArriveWaitS(devCommPtr, devMemPtr, (float *)regBuff,
                                      s6Output, NS6, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  float *hostS6 = new float[NS6];
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS6, s6Output, NS6 * sizeof(float),
                                      flagcxMemcpyDeviceToHost, NULL));

  float expectedS6 = (float)(peer + 100);
  bool s6Pass = true;
  for (int i = 0; i < NS6; i++) {
    if (fabsf(hostS6[i] - expectedS6) > 1e-3f) {
      s6Pass = false;
      break;
    }
  }
  if (proc == 0) {
    printf("S6 IntraBarrierArriveWait(Scalar): %s\n", s6Pass ? "PASS" : "FAIL");
  }
  delete[] hostS6;
  FLAGCXCHECK(devHandle->deviceFree(s6Output, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // Summary
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);

  int allPass = k1Pass && k2Pass && k3Pass && k4Pass && k5Pass && k6Pass &&
                k7Pass && k8Pass && s1Pass && s2Pass && s3Pass && s4Pass &&
                s5Pass && s6Pass;
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
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
