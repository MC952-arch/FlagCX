/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Unified One-Sided Tests — transport-transparent put/get/signal/
 * barrier that dispatch P2P vs Net automatically.
 *
 * Tests S16–S22 (aligned with device_ir.h declarations):
 *   S16: Unified Put — P2P intra-node
 *   S17: Unified Put + Signal + Wait pipeline
 *   S18: Unified Get — P2P intra-node
 *   S19: Unified Barrier — Intra-node sync
 *   S20: Unified Barrier — World sync (intra + inter)
 *   S21: Unified Signal — standalone signal + wait
 *   S22: Team-resolution correctness test
 *
 * Requirements:
 *   - Single-node with 2+ GPUs (P2P path only — no Net contexts needed)
 *   - FLAGCX_USE_HETERO_COMM=1 (for DevComm)
 *
 * Usage: mpirun -np N ./test_device_ir_unified [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 ************************************************************************/

#include "device_ir.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
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
#define RPRINTF(...)                                                           \
  do {                                                                         \
    printf("[rank %d] ", proc);                                                \
    printf(__VA_ARGS__);                                                       \
    fflush(stdout);                                                            \
  } while (0)
  MPI_Comm splitComm;
  uint64_t splitMask = 0;
  int color = 0;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();

  if (stepFactor <= 1) {
    if (proc == 0)
      printf("Error: stepFactor must be > 1, got %d\n", stepFactor);
    MPI_Finalize();
    return 1;
  }

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

  // Create DevComm with signal/counter/barrier slots
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 12; // 12 combinations for S17 and S21
  reqs.interCounterCount = 1;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Allocate send/recv buffers (12x for multi-combination regions in
  // S16/S17/S18)
  size_t bufSize = maxBytes * 12;
  void *sendBuff = nullptr, *recvBuff = nullptr;
#ifdef FLAGCX_COMM_TRAITS_SHMEM
  flagcxMemAllocator_t memAllocator = flagcxMemSHMEM;
#else
  flagcxMemAllocator_t memAllocator = flagcxMemCCL;
#endif
  FLAGCXCHECK(flagcxMemAlloc(&sendBuff, bufSize, memAllocator));
  FLAGCXCHECK(flagcxMemAlloc(&recvBuff, bufSize, memAllocator));

  // Register symmetric windows
  flagcxWindow_t sendWin = nullptr, recvWin = nullptr;
  FLAGCXCHECK(flagcxCommWindowRegister(comm, sendBuff, bufSize, &sendWin,
                                       FLAGCX_WIN_COLL_SYMMETRIC,
                                       memAllocator));
  FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, bufSize, &recvWin,
                                       FLAGCX_WIN_COLL_SYMMETRIC,
                                       memAllocator));

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

  // Host scratch - allocate 12× buffers for 12 combinations
  float *hostSend = new float[bufSize * 12 / sizeof(float)];
  float *hostRecv = new float[bufSize * 12 / sizeof(float)];

  if (proc == 0) {
    printf("=== Device IR Unified One-Sided Tests (P2P path) ===\n");
    printf("Ranks: %d\n\n", totalProcs);
  }

  bool allPass = true;
  int peer = (proc + 1) % totalProcs;
  int prevPeer = (proc + totalProcs - 1) % totalProcs;
  // Team geometry (single-node assumption: intraSize == totalProcs)
  int intraSize = totalProcs; // adjusted if multi-node
  int intraRank = proc;       // worldRank == intraRank on single node
  int intraBase = proc - intraRank;
  int nNodes = totalProcs / intraSize;
  int nodeIdx = proc / intraSize;

  // =========================================================================
  // S19: Unified Barrier — Intra-node sync (size-independent, run once)
  // =========================================================================
  {
    MPI_Barrier(MPI_COMM_WORLD);
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                        flagcxMemDevice, stream));
    launchKernelDevBarrierIntraS(devCommPtr, devResults, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    int hostRes = 0;
    FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    bool s19Pass = (hostRes == 1);
    RPRINTF("S19 DevBarrier(INTRA): %s\n", s19Pass ? "PASS" : "FAIL");
    allPass &= s19Pass;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // =========================================================================
  // S20: Unified Barrier — World sync (size-independent, run once)
  // =========================================================================
  {
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                        flagcxMemDevice, stream));
    launchKernelDevBarrierWorldS(devCommPtr, devResults, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    int hostRes = 0;
    FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    bool s20Pass = (hostRes == 1);
    RPRINTF("S20 DevBarrier(WORLD): %s\n", s20Pass ? "PASS" : "FAIL");
    allPass &= s20Pass;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // =========================================================================
  // S21: Unified Signal — standalone signal + wait — 12 combinations (4 coop ×
  // 3 teams)
  // =========================================================================
  {
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                        flagcxMemDevice, stream));
    launchKernelDevSignalStandaloneS(devCommPtr, devResults, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    int hostRes = 0;
    FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    bool s21Pass = (hostRes == 1);
    RPRINTF("S21 DevSignal(12 combinations): %s%s\n", s21Pass ? "PASS" : "FAIL",
            (!s21Pass) ? " (signal/wait hung)" : "");
    allPass &= s21Pass;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // =========================================================================
  // Main size loop: S16, S17, S18, S22 (data transfer tests)
  // =========================================================================
  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t count = size / sizeof(float);
    if (count == 0)
      count = 1;
    size_t bytes = count * sizeof(float);

    if (proc == 0)
      printf("# Size = %zu bytes, count = %zu\n", bytes, count);

    MPI_Barrier(MPI_COMM_WORLD);

    // --- S16: Unified Put — 12 combinations (4 coop × 3 teams) ---
    {
      // Fill 12 regions: combination i uses offset i*count
      for (size_t i = 0; i < count * 12; i++)
        hostSend[i] = (float)(proc * 1000 + (int)i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, bytes * 12,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, bytes * 12,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      launchKernelDevPutS(devCommPtr, recvMemPtr, sendMemPtr, devResults, bytes,
                          stream);
      if (worldRank == 0)
        printf("[Test] S16 kernel launched\n");

      // Check for immediate kernel launch error
      cudaError_t launchErr = cudaGetLastError();
      if (launchErr != cudaSuccess) {
        fprintf(stderr, "[Rank %d] S16 kernel launch error: %s\n", worldRank,
                cudaGetErrorString(launchErr));
      }

      flagcxResult_t syncErr = devHandle->streamSynchronize(stream);
      if (syncErr != flagcxSuccess) {
        // Get the underlying CUDA error
        cudaError_t cudaErr = cudaGetLastError();
        fprintf(stderr,
                "[Rank %d] S16 streamSync failed: flagcxResult=%d, "
                "cudaError=%s (code %d)\n",
                worldRank, syncErr, cudaGetErrorString(cudaErr), cudaErr);
        FLAGCXCHECK(syncErr);
      }
      if (worldRank == 0)
        printf("[Test] S16 streamSync returned success\n");

      // Check kernel result flag
      int hResult = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hResult, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));
      if (worldRank == 0)
        printf("[Test] S16 kernel result flag = %d\n", hResult);

      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, bytes * 12,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s16Pass = true;
      int s16FailCombo = -1;
      size_t s16FailIdx = 0;
      float s16FailExpected = 0, s16FailActual = 0;

      // Validate all 12 combinations
      int prevIntra = (intraRank + intraSize - 1) % intraSize;
      int prevIntraWorld = intraBase + prevIntra;
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevNodeWorld = (nNodes > 1) ? (prevNode * intraSize + intraRank) : 0;

      for (int combo = 0; combo < 12 && s16Pass; combo++) {
        int teamIdx = combo % 3;
        size_t baseOff = combo * count;

        // Determine expected sender based on team
        int expectedSender = 0;
        if (teamIdx == 0) { // INTRA
          expectedSender = prevIntraWorld;
        } else if (teamIdx == 1) { // WORLD
          expectedSender = prevPeer;
        } else { // INTER
          if (nNodes == 1)
            continue; // Skip INTER on single-node
          expectedSender = prevNodeWorld;
        }

        // Validate data in this region
        for (size_t i = 0; i < count; i++) {
          float expected = (float)(expectedSender * 1000 + (int)(baseOff + i));
          if (hostRecv[baseOff + i] != expected) {
            s16Pass = false;
            s16FailCombo = combo;
            s16FailIdx = i;
            s16FailExpected = expected;
            s16FailActual = hostRecv[baseOff + i];
            break;
          }
        }
      }

      if (!s16Pass) {
        const char *coopNames[] = {"THREAD", "WARP", "BLOCK", "GRID"};
        const char *teamNames[] = {"INTRA", "WORLD", "INTER"};
        int failCoop = s16FailCombo / 3;
        int failTeam = s16FailCombo % 3;
        RPRINTF("S16 DevPut(12 combinations): FAIL combo=%d (%s+%s) idx=%zu "
                "expected=%f actual=%f\n",
                s16FailCombo, coopNames[failCoop], teamNames[failTeam],
                s16FailIdx, s16FailExpected, s16FailActual);
        // Dump first 8 values of failed region
        size_t failBase = s16FailCombo * count;
        RPRINTF("  recv[%zu..%zu]: ", failBase, failBase + 7);
        for (int d = 0; d < 8 && d < (int)count; d++)
          RPRINTF("%f ", hostRecv[failBase + d]);
        RPRINTF("\n");
      } else {
        RPRINTF("S16 DevPut(12 combinations): PASS\n");
      }
      allPass &= s16Pass;
    }

    // --- S17: Unified Put + Signal + Wait pipeline (12 combinations) ---
    {
      for (size_t i = 0; i < count * 12; i++)
        hostSend[i] = (float)(proc * 2000 + (int)i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, bytes * 12,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, bytes * 12,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      if (proc == 0) {
        printf("[Host S17] Rank %d launching kernel\n", proc);
      }
      launchKernelDevPutSignalWaitS(devCommPtr, recvMemPtr, sendMemPtr,
                                    devResults, bytes, stream);
      if (proc == 0) {
        printf("[Host S17] Rank %d calling streamSynchronize\n", proc);
      }
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      if (proc == 0) {
        printf("[Host S17] Rank %d streamSynchronize returned\n", proc);
      }

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, bytes * 12,
                                          flagcxMemcpyDeviceToHost, stream));
      bool dataOk = true;
      int s17FailCombo = -1;
      size_t s17FailIdx = 0;
      float s17FailExpected = 0, s17FailActual = 0;

      // Validate all 12 combinations
      int prevIntra17 = (intraRank + intraSize - 1) % intraSize;
      int prevIntraWorld17 = intraBase + prevIntra17;
      int prevNode17 = (nodeIdx + nNodes - 1) % nNodes;
      int prevNodeWorld17 =
          (nNodes > 1) ? (prevNode17 * intraSize + intraRank) : 0;

      for (int combo = 0; combo < 12 && dataOk; combo++) {
        int teamIdx = combo % 3;
        size_t baseOff = combo * count;

        // Determine expected sender based on team
        int expectedSender = 0;
        if (teamIdx == 0) { // INTRA
          expectedSender = prevIntraWorld17;
        } else if (teamIdx == 1) { // WORLD
          expectedSender = prevPeer;
        } else { // INTER
          if (nNodes == 1)
            continue; // Skip INTER on single-node
          expectedSender = prevNodeWorld17;
        }

        // Validate data in this region
        for (size_t i = 0; i < count; i++) {
          float expected = (float)(expectedSender * 2000 + (int)(baseOff + i));
          if (hostRecv[baseOff + i] != expected) {
            dataOk = false;
            s17FailCombo = combo;
            s17FailIdx = i;
            s17FailExpected = expected;
            s17FailActual = hostRecv[baseOff + i];
            break;
          }
        }
      }

      bool s17Pass = (hostRes == 1) && dataOk;
      if (!s17Pass) {
        if (hostRes != 1) {
          RPRINTF("S17 DevPut+Signal+Wait(12 combinations): FAIL (kernel "
                  "hung/timeout)\n");
        } else {
          const char *coopNames[] = {"THREAD", "WARP", "BLOCK", "GRID"};
          const char *teamNames[] = {"INTRA", "WORLD", "INTER"};
          int failCoop = s17FailCombo / 3;
          int failTeam = s17FailCombo % 3;
          RPRINTF("S17 DevPut+Signal+Wait(12 combinations): FAIL combo=%d "
                  "(%s+%s) idx=%zu "
                  "expected=%f actual=%f\n",
                  s17FailCombo, coopNames[failCoop], teamNames[failTeam],
                  s17FailIdx, s17FailExpected, s17FailActual);
        }
      } else {
        RPRINTF("S17 DevPut+Signal+Wait(12 combinations): PASS\n");
      }
      allPass &= s17Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S18: Unified Get — 12 combinations (4 coop × 3 teams) ---
    {
      // Fill send buffer (source for Get) with 12 regions
      for (size_t i = 0; i < count * 12; i++)
        hostSend[i] = (float)(proc * 3000 + (int)i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, bytes * 12,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, bytes * 12,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      launchKernelDevGetS(devCommPtr, sendMemPtr, recvMemPtr, devResults, bytes,
                          stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, bytes * 12,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s18Pass = true;
      int s18FailCombo = -1;
      size_t s18FailIdx = 0;
      float s18FailExpected = 0, s18FailActual = 0;

      // Validate all 12 combinations
      int nextIntra = (intraRank + 1) % intraSize;
      int nextIntraWorld = intraBase + nextIntra;
      int nextNode = (nodeIdx + 1) % nNodes;
      int nextNodeWorld = (nNodes > 1) ? (nextNode * intraSize + intraRank) : 0;

      for (int combo = 0; combo < 12 && s18Pass; combo++) {
        int teamIdx = combo % 3;
        size_t baseOff = combo * count;

        // Determine expected source based on team
        int expectedSource = 0;
        if (teamIdx == 0) { // INTRA
          expectedSource = nextIntraWorld;
        } else if (teamIdx == 1) { // WORLD
          expectedSource = peer;
        } else { // INTER
          if (nNodes == 1)
            continue; // Skip INTER on single-node
          expectedSource = nextNodeWorld;
        }

        // Validate data in this region
        for (size_t i = 0; i < count; i++) {
          float expected = (float)(expectedSource * 3000 + (int)(baseOff + i));
          if (hostRecv[baseOff + i] != expected) {
            s18Pass = false;
            s18FailCombo = combo;
            s18FailIdx = i;
            s18FailExpected = expected;
            s18FailActual = hostRecv[baseOff + i];
            break;
          }
        }
      }

      if (!s18Pass) {
        const char *coopNames[] = {"THREAD", "WARP", "BLOCK", "GRID"};
        const char *teamNames[] = {"INTRA", "WORLD", "INTER"};
        int failCoop = s18FailCombo / 3;
        int failTeam = s18FailCombo % 3;
        RPRINTF("S18 DevGet(12 combinations): FAIL combo=%d (%s+%s) idx=%zu "
                "expected=%f actual=%f\n",
                s18FailCombo, coopNames[failCoop], teamNames[failTeam],
                s18FailIdx, s18FailExpected, s18FailActual);
      } else {
        RPRINTF("S18 DevGet(12 combinations): PASS\n");
      }
      allPass &= s18Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (proc == 0)
      printf("#\n");
  }

  // =========================================================================
  // S22: Team-resolution correctness test — 12 combinations (4 coop × 3 teams)
  // =========================================================================
  {
    MPI_Barrier(MPI_COMM_WORLD);
    // S22 needs space for 12 regions, each maxRanks in size
    int maxRanks = intraSize;
    if (totalProcs > maxRanks)
      maxRanks = totalProcs;
    if (nNodes > maxRanks)
      maxRanks = nNodes;
    size_t s22Size = 12 * maxRanks * sizeof(float);

    // Pre-fill sendBuff[0] = my world rank as tag
    float myTag = (float)proc;
    FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, &myTag, sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(
        devHandle->deviceMemset(recvBuff, 0, s22Size, flagcxMemDevice, stream));
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelDevTeamResolutionS(devCommPtr, recvMemPtr, sendMemPtr,
                                   devResults, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    int hostRes = 0;
    FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    // Host-side verification: read recvBuff and check tags
    float *s22Recv = new float[12 * maxRanks];
    FLAGCXCHECK(devHandle->deviceMemcpy(s22Recv, recvBuff, s22Size,
                                        flagcxMemcpyDeviceToHost, stream));

    bool s22Pass = (hostRes == 1);
    int s22FailCombo = -1;
    int s22FailSlot = -1;
    float s22FailExpected = 0, s22FailActual = 0;

    int prevIntra22 = (intraRank + intraSize - 1) % intraSize;
    int prevWorld22 = (proc + totalProcs - 1) % totalProcs;
    int prevNode22 = (nodeIdx + nNodes - 1) % nNodes;

    for (int combo = 0; combo < 12 && s22Pass; combo++) {
      int teamIdx = combo % 3;
      size_t baseOff = combo * maxRanks;

      if (teamIdx == 0) { // INTRA
        int expectedIntraWorld = intraBase + prevIntra22;
        float expected = (float)expectedIntraWorld;
        if (s22Recv[baseOff + prevIntra22] != expected) {
          s22Pass = false;
          s22FailCombo = combo;
          s22FailSlot = prevIntra22;
          s22FailExpected = expected;
          s22FailActual = s22Recv[baseOff + prevIntra22];
        }
      } else if (teamIdx == 1) { // WORLD
        float expected = (float)prevWorld22;
        if (s22Recv[baseOff + prevWorld22] != expected) {
          s22Pass = false;
          s22FailCombo = combo;
          s22FailSlot = prevWorld22;
          s22FailExpected = expected;
          s22FailActual = s22Recv[baseOff + prevWorld22];
        }
      } else { // INTER
        if (nNodes == 1)
          continue; // Skip INTER on single-node
        int expectedInterWorld = prevNode22 * intraSize + intraRank;
        float expected = (float)expectedInterWorld;
        if (s22Recv[baseOff + prevNode22] != expected) {
          s22Pass = false;
          s22FailCombo = combo;
          s22FailSlot = prevNode22;
          s22FailExpected = expected;
          s22FailActual = s22Recv[baseOff + prevNode22];
        }
      }
    }

    if (!s22Pass) {
      const char *coopNames[] = {"THREAD", "WARP", "BLOCK", "GRID"};
      const char *teamNames[] = {"INTRA", "WORLD", "INTER"};
      int failCoop = s22FailCombo / 3;
      int failTeam = s22FailCombo % 3;
      RPRINTF(
          "S22 TeamResolution(12 combinations): FAIL combo=%d (%s+%s) slot=%d"
          "expected=%f actual=%f\n",
          s22FailCombo, coopNames[failCoop], teamNames[failTeam], s22FailSlot,
          s22FailExpected, s22FailActual);
    } else {
      RPRINTF("S22 TeamResolution(12 combinations): PASS\n");
    }
    allPass &= s22Pass;
    delete[] s22Recv;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // =========================================================================
  // Summary
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  int pass = allPass ? 1 : 0;
  int globalPass = 0;
  MPI_Allreduce(&pass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  printf("[rank %d] === Overall: %s ===\n", proc, globalPass ? "PASS" : "FAIL");

  // Cleanup
  delete[] hostSend;
  delete[] hostRecv;
  FLAGCXCHECK(devHandle->deviceFree(devResults, flagcxMemDevice, NULL));
  FLAGCXCHECK(flagcxDevMemFreeDevicePtr(sendMem));
  FLAGCXCHECK(flagcxDevMemFreeDevicePtr(recvMem));
  FLAGCXCHECK(flagcxDevCommFreeDevicePtr(devComm));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, recvMem));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin, memAllocator));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin, memAllocator));
  FLAGCXCHECK(flagcxMemFree(sendBuff, memAllocator));
  FLAGCXCHECK(flagcxMemFree(recvBuff, memAllocator));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
