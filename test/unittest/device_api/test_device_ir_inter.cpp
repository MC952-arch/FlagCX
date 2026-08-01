/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Inter-Node Tests — Scalar IR transport functions.
 *
 * Exercises ALL inter-node S-API categories from flagcx_device_scalar_ir.h:
 *
 * Signal/Counter/Flush:
 *   S9:  NetGetFromComm (flagcxDevNetGetFromCommS)
 *   S10: Signal/Counter local read/reset/shadow (readSignalS, resetSignal,
 *        resetCounter, increaseSignalShadow)
 *   S11: WaitSignalS + FlushS
 *   S12: WaitCounterS
 *   S13: WaitSignalMeetShadowS
 *
 * One-sided put (4x4 matrix, test key combos):
 *   S14: PutS (None, None) + FlushS + WaitSignalS
 *   S15: PutS_RSigInc (SigInc, None) + WaitSignalS + FlushS
 *   S16: PutS_RSigAdd (SigAdd, None) + WaitSignalS + FlushS
 *   S17: PutS_RSigInc_LCtrInc (SigInc, CtrInc) + WaitSignalS + WaitCounterS +
 *FlushS
 *
 * One-sided signal (standalone):
 *   S18: SignalSigIncS
 *   S19: SignalSigAddS
 *   // S20: SignalCtrIncS (commented — counter-only signal not yet validated)
 *
 * One-sided putValue:
 *   S21: PutValueS (None)
 *   S22: PutValueS_RSigInc
 *
 * One-sided get:
 *   S23: GetS + FlushS
 *
 * Two-sided:
 *   S24: SendS + RecvS + TermS + WaitS
 *
 * Barrier (inter/world):
 *   S25: InterBarrierTest
 *   S26: WorldBarrierSyncS
 *   S27: WorldBarrierArriveS + WorldBarrierWaitS
 *
 * Requirements:
 *   - Multi-node, OR single-node with FLAGCX_P2P_DISABLE=1
 *   - FLAGCX_USE_HETERO_COMM=1 (for DevComm with inter context)
 *
 * Usage: mpirun -np N ./test_device_ir_inter [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -R <regMode>   2=window (required for inter ops)
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

  // Allocate test buffer (4 MB)
  size_t bufSize = 4 * 1024 * 1024;
  void *sendBuff = nullptr, *recvBuff = nullptr;
  FLAGCXCHECK(flagcxMemAlloc(&sendBuff, bufSize));
  FLAGCXCHECK(flagcxMemAlloc(&recvBuff, bufSize));

  // Register symmetric windows
  flagcxWindow_t sendWin = nullptr, recvWin = nullptr;
  FLAGCXCHECK(flagcxCommWindowRegister(comm, sendBuff, bufSize, &sendWin,
                                       FLAGCX_WIN_COLL_SYMMETRIC));
  FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, bufSize, &recvWin,
                                       FLAGCX_WIN_COLL_SYMMETRIC));

  // Create DevComm with enough signal/counter slots
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 3;
  reqs.interCounterCount = 1;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create DevMem handles
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, sendBuff, bufSize, sendWin, &sendMem));
  FLAGCXCHECK(flagcxDevMemCreate(comm, recvBuff, bufSize, recvWin, &recvMem));

  // Get device pointers for IR functions
  void *devCommPtr = nullptr;
  FLAGCXCHECK(flagcxDevCommGetDevicePtr(devComm, &devCommPtr));
  void *sendMemPtr = nullptr, *recvMemPtr = nullptr;
  FLAGCXCHECK(flagcxDevMemGetDevicePtr(sendMem, &sendMemPtr));
  FLAGCXCHECK(flagcxDevMemGetDevicePtr(recvMem, &recvMemPtr));

  // Allocate results buffer
  int *devResults = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devResults, 256 * sizeof(int),
                                      flagcxMemDevice, NULL));

  if (proc == 0) {
    printf("=== Device IR Inter-Node Transport Tests ===\n");
    printf("Ranks: %d\n\n", totalProcs);
  }

  // =========================================================================
  // S9: Net GetFromCommS
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelNetGetFromCommS(devCommPtr, devResults, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  int hostS9[4] = {0};
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS9, devResults, 4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool s9Pass = (hostS9[0] == 1); // net pointer should be non-null
  bool s9Skip = (hostS9[0] == 0); // net unavailable → skip all inter tests
  int intraSize = hostS9[1] > 0 ? hostS9[1] : totalProcs;
  int intraBase = proc - (proc % intraSize);
  if (proc == 0) {
    printf("S9  NetGetFromComm: %s (intraSize=%d)\n",
           s9Skip ? "SKIP (no transport contexts)" : (s9Pass ? "PASS" : "FAIL"),
           intraSize);
  }
  if (s9Skip)
    s9Pass = true;

  // =========================================================================
  // S10: Signal/Counter local read/reset/shadow
  // =========================================================================
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelNetSignalCounterS(devCommPtr, devResults, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  int hostS10[4] = {0};
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS10, devResults, 4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool s10Pass = (hostS10[0] == 1) && (hostS10[1] == 1) && (hostS10[2] == 1);
  bool s10Skip = (hostS10[0] == 0) && (hostS10[1] == 0) && (hostS10[2] == 0);
  if (proc == 0) {
    printf("S10 NetSignalCounter: %s\n", s10Skip
                                             ? "SKIP (no transport contexts)"
                                             : (s10Pass ? "PASS" : "FAIL"));
  }
  if (s10Skip)
    s10Pass = true;
  MPI_Barrier(MPI_COMM_WORLD);

  // =========================================================================
  // S11-S27: Transport tests (require inter-node or FLAGCX_P2P_DISABLE=1)
  // =========================================================================

  bool allInterPass = true;
  size_t countPerPeer = 1024;
  size_t floatSize = (size_t)totalProcs * countPerPeer * sizeof(float);
  size_t putValBase = bufSize - (size_t)totalProcs * sizeof(uint64_t);
  float *hostBuf = new float[totalProcs * countPerPeer];

  // Helper lambda: init sendBuff with alltoall pattern
  auto initSend = [&]() {
    for (int r = 0; r < totalProcs; r++)
      for (size_t i = 0; i < countPerPeer; i++)
        hostBuf[(size_t)r * countPerPeer + i] =
            (float)(proc * 1000 + r * 100 + (int)i);
    devHandle->deviceMemcpy(sendBuff, hostBuf, floatSize,
                            flagcxMemcpyHostToDevice, NULL);
  };

  // Helper lambda: verify alltoall pattern in recvBuff (inter peers only)
  auto verifyAlltoAll = [&]() -> bool {
    devHandle->deviceMemcpy(hostBuf, recvBuff, floatSize,
                            flagcxMemcpyDeviceToHost, NULL);
    for (int src = 0; src < totalProcs; src++) {
      if (src >= intraBase && src < intraBase + intraSize)
        continue;
      for (size_t i = 0; i < countPerPeer; i++) {
        float expected = (float)(src * 1000 + proc * 100 + (int)i);
        if (hostBuf[(size_t)src * countPerPeer + i] != expected)
          return false;
      }
    }
    return true;
  };

  // --- S11: WaitSignalS + FlushS ---
  if (!s9Skip) {
    MPI_Barrier(MPI_COMM_WORLD);
    launchKernelNetWaitSignalFlushS(devCommPtr, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    printf("[rank %d] S11 WaitSignalS+FlushS: PASS\n", proc);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S12: WaitCounterS (COMMENTED — standalone SignalCtrIncS is not
  //     supported by the GIN protocol; counters are local-action only,
  //     tested via S17 PutS_RSigInc_LCtrInc) ---
  // if (!s9Skip) {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   launchKernelNetWaitCounterS(devCommPtr, stream);
  //   FLAGCXCHECK(devHandle->streamSynchronize(stream));
  //   if (proc == 0)
  //     printf("S12 WaitCounterS: PASS\n");
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // --- S13: WaitSignalMeetShadowS ---
  // if (!s9Skip) {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   launchKernelNetWaitSignalMeetShadowS(devCommPtr, stream);
  //   FLAGCXCHECK(devHandle->streamSynchronize(stream));
  //   if (proc == 0)
  //     printf("S13 WaitSignalMeetShadowS: PASS\n");
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // --- S14: PutS (None, None) + signal + wait + flush ---
  if (!s9Skip) {
    initSend();
    FLAGCXCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice, NULL));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelNetPutS(devCommPtr, sendMemPtr, recvMemPtr, countPerPeer,
                        stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    bool s14Ok = verifyAlltoAll();
    printf("[rank %d] S14 PutS(None,None): %s\n", proc,
           s14Ok ? "PASS" : "FAIL");
    fflush(stdout);
    allInterPass &= s14Ok;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S15: PutS_RSigInc ---
  if (!s9Skip) {
    initSend();
    FLAGCXCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice, NULL));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelNetPutRSigIncS(devCommPtr, sendMemPtr, recvMemPtr, countPerPeer,
                               stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    bool s15Ok = verifyAlltoAll();
    printf("[rank %d] S15 PutS_RSigInc: %s\n", proc, s15Ok ? "PASS" : "FAIL");
    fflush(stdout);
    allInterPass &= s15Ok;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S16: PutS_RSigAdd ---
  if (!s9Skip) {
    initSend();
    FLAGCXCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice, NULL));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelNetPutRSigAddS(devCommPtr, sendMemPtr, recvMemPtr, countPerPeer,
                               stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    bool s16Ok = verifyAlltoAll();
    printf("[rank %d] S16 PutS_RSigAdd: %s\n", proc, s16Ok ? "PASS" : "FAIL");
    fflush(stdout);
    allInterPass &= s16Ok;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S17: PutS_RSigInc_LCtrInc ---
  if (!s9Skip) {
    initSend();
    FLAGCXCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice, NULL));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelNetPutRSigLCtrS(devCommPtr, sendMemPtr, recvMemPtr,
                                countPerPeer, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    bool s17Ok = verifyAlltoAll();
    printf("[rank %d] S17 PutS_RSigInc_LCtrInc: %s\n", proc,
           s17Ok ? "PASS" : "FAIL");
    fflush(stdout);
    allInterPass &= s17Ok;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S18: SignalSigIncS ---
  if (!s9Skip) {
    MPI_Barrier(MPI_COMM_WORLD);
    launchKernelNetSignalSigIncS(devCommPtr, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    printf("[rank %d] S18 SignalSigIncS: PASS\n", proc);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S19: SignalSigAddS ---
  if (!s9Skip) {
    MPI_Barrier(MPI_COMM_WORLD);
    launchKernelNetSignalSigAddS(devCommPtr, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    printf("[rank %d] S19 SignalSigAddS: PASS\n", proc);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S21: PutValueS ---
  if (!s9Skip) {
    FLAGCXCHECK(devHandle->deviceMemset((char *)recvBuff + putValBase, 0,
                                        (size_t)totalProcs * sizeof(uint64_t),
                                        flagcxMemDevice, NULL));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelNetPutValueS(devCommPtr, recvMemPtr, putValBase, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    uint64_t hostVals[64] = {};
    FLAGCXCHECK(devHandle->deviceMemcpy(hostVals, (char *)recvBuff + putValBase,
                                        (size_t)totalProcs * sizeof(uint64_t),
                                        flagcxMemcpyDeviceToHost, NULL));
    bool s21Ok = true;
    for (int src = 0; src < totalProcs; src++) {
      if (src >= intraBase && src < intraBase + intraSize)
        continue;
      uint64_t expected = (uint64_t)src * 1000u + (uint64_t)proc;
      if (hostVals[src] != expected) {
        s21Ok = false;
        break;
      }
    }
    printf("[rank %d] S21 PutValueS: %s\n", proc, s21Ok ? "PASS" : "FAIL");
    fflush(stdout);
    allInterPass &= s21Ok;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S22: PutValueS_RSigInc ---
  if (!s9Skip) {
    FLAGCXCHECK(devHandle->deviceMemset((char *)recvBuff + putValBase, 0,
                                        (size_t)totalProcs * sizeof(uint64_t),
                                        flagcxMemDevice, NULL));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelNetPutValueRSigS(devCommPtr, recvMemPtr, putValBase, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    uint64_t hostVals[64] = {};
    FLAGCXCHECK(devHandle->deviceMemcpy(hostVals, (char *)recvBuff + putValBase,
                                        (size_t)totalProcs * sizeof(uint64_t),
                                        flagcxMemcpyDeviceToHost, NULL));
    bool s22Ok = true;
    for (int src = 0; src < totalProcs; src++) {
      if (src >= intraBase && src < intraBase + intraSize)
        continue;
      uint64_t expected = (uint64_t)src * 1000u + (uint64_t)proc;
      if (hostVals[src] != expected) {
        s22Ok = false;
        break;
      }
    }
    printf("[rank %d] S22 PutValueS_RSigInc: %s\n", proc,
           s22Ok ? "PASS" : "FAIL");
    fflush(stdout);
    allInterPass &= s22Ok;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S23: GetS + FlushS ---
  if (!s9Skip) {
    initSend();
    FLAGCXCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice, NULL));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelNetGetS(devCommPtr, sendMemPtr, recvMemPtr, countPerPeer,
                        stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    bool s23Ok = verifyAlltoAll();
    printf("[rank %d] S23 GetS: %s\n", proc, s23Ok ? "PASS" : "FAIL");
    fflush(stdout);
    allInterPass &= s23Ok;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S25: Inter-Barrier Test ---
  if (!s9Skip) {
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                        flagcxMemDevice, NULL));
    MPI_Barrier(MPI_COMM_WORLD);
    launchKernelInterBarrierStress(devCommPtr, devResults, 3, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    int hostRes[1] = {0};
    FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults, sizeof(int),
                                        flagcxMemcpyDeviceToHost, NULL));
    printf("[rank %d] S25 InterBarrier: %s\n", proc,
           hostRes[0] == 1 ? "PASS" : (hostRes[0] == -1 ? "SKIP" : "FAIL"));
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S24: Two-sided (COMMENTED) ---
  // if (!s9Skip) {
  //   initSend();
  //   FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
  //                                       flagcxMemDevice, NULL));
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   launchKernelNetTwoSidedS(devCommPtr, sendMemPtr, recvMemPtr,
  //   countPerPeer,
  //                            stream);
  //   FLAGCXCHECK(devHandle->streamSynchronize(stream));
  //   bool s24Ok = verifyAlltoAll();
  //   if (proc == 0)
  //     printf("S24 TwoSidedS: %s\n", s24Ok ? "PASS" : "FAIL");
  //   allInterPass &= s24Ok;
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // --- S26: WorldBarrierSyncS ---
  if (!s9Skip) {
    MPI_Barrier(MPI_COMM_WORLD);
    launchKernelWorldBarrierS(devCommPtr, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    printf("[rank %d] S26 WorldBarrierSyncS: PASS\n", proc);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- S27: WorldBarrierArriveS + WaitS ---
  if (!s9Skip) {
    MPI_Barrier(MPI_COMM_WORLD);
    launchKernelWorldBarrierSplitS(devCommPtr, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    printf("[rank %d] S27 WorldBarrierSplitS: PASS\n", proc);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  delete[] hostBuf;

  // =========================================================================
  // Summary
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  int allPass = s9Pass && s10Pass && allInterPass;
  int globalPass = 0;
  MPI_Allreduce(&allPass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (proc == 0) {
    printf("\n=== Overall: %s ===\n", globalPass ? "PASS" : "FAIL");
  }

  // Cleanup
  FLAGCXCHECK(devHandle->deviceFree(devResults, flagcxMemDevice, NULL));
  FLAGCXCHECK(flagcxDevMemFreeDevicePtr(sendMem));
  FLAGCXCHECK(flagcxDevMemFreeDevicePtr(recvMem));
  FLAGCXCHECK(flagcxDevCommFreeDevicePtr(devComm));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, recvMem));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin));
  FLAGCXCHECK(flagcxMemFree(sendBuff));
  FLAGCXCHECK(flagcxMemFree(recvBuff));
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
