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
 *   S21: Unified Put — Warp-level (fine-grained)
 *   S22: Unified Signal — standalone signal + wait
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
  reqs.interSignalCount = 3;
  reqs.interCounterCount = 1;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Allocate send/recv buffers (3x for multi-team regions in S16/S17/S18/S21)
  size_t bufSize = maxBytes * 3;
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

  // Host scratch
  float *hostSend = new float[bufSize / sizeof(float)];
  float *hostRecv = new float[bufSize / sizeof(float)];

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
  // S22: Unified Signal — standalone signal + wait (size-independent, run once)
  // =========================================================================
  {
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                        flagcxMemDevice, stream));
    launchKernelDevSignalStandaloneS(devCommPtr, devResults, stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    int hostRes = 0;
    FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    bool s22Pass = (hostRes == 1);
    RPRINTF("S22 DevSignal(INTRA+WORLD+INTER): %s%s\n",
            s22Pass ? "PASS" : "FAIL", (!s22Pass) ? " (signal/wait hung)" : "");
    allPass &= s22Pass;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // =========================================================================
  // Main size loop: S16, S17, S18, S21 (data transfer tests)
  // =========================================================================
  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t count = size / sizeof(float);
    if (count == 0)
      count = 1;
    size_t bytes = count * sizeof(float);

    if (proc == 0)
      printf("# Size = %zu bytes, count = %zu\n", bytes, count);

    MPI_Barrier(MPI_COMM_WORLD);

    // --- S16: Unified Put — 3 teams (INTRA / WORLD / INTER) ---
    {
      // Fill 3 regions: [0..count), [count..2*count), [2*count..3*count)
      for (size_t i = 0; i < count * 3; i++)
        hostSend[i] = (float)(proc * 1000 + (int)i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, bytes * 3,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, bytes * 3,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      launchKernelDevPutS(devCommPtr, recvMemPtr, sendMemPtr, devResults, bytes,
                          stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, bytes * 3,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s16Pass = true;
      int s16FailRegion = -1;
      size_t s16FailIdx = 0;
      float s16FailExpected = 0, s16FailActual = 0;
      // Region 0 (INTRA): data from prevPeer (== prevIntraRank on single-node)
      int prevIntra = (intraRank + intraSize - 1) % intraSize;
      int prevIntraWorld = intraBase + prevIntra;
      for (size_t i = 0; i < count; i++) {
        float expected = (float)(prevIntraWorld * 1000 + (int)i);
        if (hostRecv[i] != expected) {
          s16Pass = false;
          s16FailRegion = 0;
          s16FailIdx = i;
          s16FailExpected = expected;
          s16FailActual = hostRecv[i];
          break;
        }
      }
      // Region 1 (WORLD): data from prevPeer (previous world rank)
      if (s16Pass) {
        for (size_t i = 0; i < count; i++) {
          float expected = (float)(prevPeer * 1000 + (int)(count + i));
          if (hostRecv[count + i] != expected) {
            s16Pass = false;
            s16FailRegion = 1;
            s16FailIdx = i;
            s16FailExpected = expected;
            s16FailActual = hostRecv[count + i];
            break;
          }
        }
      }
      // Region 2 (INTER): skip validation on single-node (INTER requires
      // inter-node peers)
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevNodeWorld = (nNodes > 1) ? (prevNode * intraSize + intraRank) : 0;
      if (nNodes > 1) {
        if (s16Pass) {
          for (size_t i = 0; i < count; i++) {
            float expected =
                (float)(prevNodeWorld * 1000 + (int)(2 * count + i));
            if (hostRecv[2 * count + i] != expected) {
              s16Pass = false;
              s16FailRegion = 2;
              s16FailIdx = i;
              s16FailExpected = expected;
              s16FailActual = hostRecv[2 * count + i];
              break;
            }
          }
        }
      }
      if (!s16Pass) {
        RPRINTF("S16 DevPut(INTRA+WORLD+INTER): FAIL region=%d idx=%zu "
                "expected=%f actual=%f (prevIntraWorld=%d prevPeer=%d "
                "prevNodeWorld=%d count=%zu)\n",
                s16FailRegion, s16FailIdx, s16FailExpected, s16FailActual,
                prevIntraWorld, prevPeer, prevNodeWorld, count);
        // Dump first 8 values of each region
        RPRINTF("  INTRA recv[0..7]: ");
        for (int d = 0; d < 8 && d < (int)count; d++)
          RPRINTF("%f ", hostRecv[d]);
        RPRINTF("\n  WORLD recv[0..7]: ");
        for (int d = 0; d < 8 && d < (int)count; d++)
          RPRINTF("%f ", hostRecv[count + d]);
        RPRINTF("\n  INTER recv[0..7]: ");
        for (int d = 0; d < 8 && d < (int)count; d++)
          RPRINTF("%f ", hostRecv[2 * count + d]);
        RPRINTF("\n");
      } else {
        RPRINTF("S16 DevPut(INTRA+WORLD+INTER): PASS\n");
      }
      allPass &= s16Pass;
    }

    // --- S17: Unified Put + Signal + Wait pipeline (3 teams) ---
    {
      for (size_t i = 0; i < count * 3; i++)
        hostSend[i] = (float)(proc * 2000 + (int)i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, bytes * 3,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, bytes * 3,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      launchKernelDevPutSignalWaitS(devCommPtr, recvMemPtr, sendMemPtr,
                                    devResults, bytes, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, bytes * 3,
                                          flagcxMemcpyDeviceToHost, stream));
      bool dataOk = true;
      // Region 0 (INTRA): from previous intra-rank
      int prevIntra17 = (intraRank + intraSize - 1) % intraSize;
      int prevIntraWorld17 = intraBase + prevIntra17;
      for (size_t i = 0; i < count; i++) {
        float expected = (float)(prevIntraWorld17 * 2000 + (int)i);
        if (hostRecv[i] != expected) {
          dataOk = false;
          break;
        }
      }
      // Region 1 (WORLD): from previous world-rank
      for (size_t i = 0; i < count; i++) {
        float expected = (float)(prevPeer * 2000 + (int)(count + i));
        if (hostRecv[count + i] != expected) {
          dataOk = false;
          break;
        }
      }
      // Region 2 (INTER): skip validation on single-node (INTER requires
      // inter-node peers)
      if (nNodes > 1) {
        int prevNode17 = (nodeIdx + nNodes - 1) % nNodes;
        int prevNodeWorld17 = prevNode17 * intraSize + intraRank;
        for (size_t i = 0; i < count; i++) {
          float expected =
              (float)(prevNodeWorld17 * 2000 + (int)(2 * count + i));
          if (hostRecv[2 * count + i] != expected) {
            dataOk = false;
            break;
          }
        }
      }

      bool s17Pass = (hostRes == 1) && dataOk;
      RPRINTF("S17 DevPut+Signal+Wait(INTRA+WORLD+INTER): %s%s\n",
              s17Pass ? "PASS" : "FAIL",
              (!s17Pass && hostRes != 1) ? " (kernel hung/timeout)" : "");
      allPass &= s17Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S18: Unified Get — 3 teams (INTRA / WORLD / INTER) ---
    {
      // Fill send buffer (source for Get) with 3 regions
      for (size_t i = 0; i < count * 3; i++)
        hostSend[i] = (float)(proc * 3000 + (int)i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, bytes * 3,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, bytes * 3,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      launchKernelDevGetS(devCommPtr, sendMemPtr, recvMemPtr, devResults, bytes,
                          stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, bytes * 3,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s18Pass = true;
      // Region 0 (INTRA): got from next intra-rank
      int nextIntra = (intraRank + 1) % intraSize;
      int nextIntraWorld = intraBase + nextIntra;
      for (size_t i = 0; i < count; i++) {
        float expected = (float)(nextIntraWorld * 3000 + (int)i);
        if (hostRecv[i] != expected) {
          s18Pass = false;
          break;
        }
      }
      // Region 1 (WORLD): got from next world-rank
      for (size_t i = 0; i < count; i++) {
        float expected = (float)(peer * 3000 + (int)(count + i));
        if (hostRecv[count + i] != expected) {
          s18Pass = false;
          break;
        }
      }
      // Region 2 (INTER): skip validation on single-node (INTER requires
      // inter-node peers)
      if (nNodes > 1) {
        int nextNode = (nodeIdx + 1) % nNodes;
        int nextNodeWorld = nextNode * intraSize + intraRank;
        for (size_t i = 0; i < count; i++) {
          float expected = (float)(nextNodeWorld * 3000 + (int)(2 * count + i));
          if (hostRecv[2 * count + i] != expected) {
            s18Pass = false;
            break;
          }
        }
      }
      RPRINTF("S18 DevGet(INTRA+WORLD+INTER): %s\n", s18Pass ? "PASS" : "FAIL");
      allPass &= s18Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S21: Unified Put — Warp-level (3 teams, fine-grained) ---
    {
      for (size_t i = 0; i < count * 3; i++)
        hostSend[i] = (float)(proc * 4000 + (int)i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, bytes * 3,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, bytes * 3,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      launchKernelDevPutWarpS(devCommPtr, recvMemPtr, sendMemPtr, devResults,
                              bytes, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      MPI_Barrier(MPI_COMM_WORLD);

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, bytes * 3,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s21Pass = true;
      // Region 0 (INTRA warp): from previous intra-rank
      int prevIntra21 = (intraRank + intraSize - 1) % intraSize;
      int prevIntraWorld21 = intraBase + prevIntra21;
      for (size_t i = 0; i < count; i++) {
        float expected = (float)(prevIntraWorld21 * 4000 + (int)i);
        if (hostRecv[i] != expected) {
          s21Pass = false;
          break;
        }
      }
      // Region 1 (WORLD warp): from previous world-rank
      for (size_t i = 0; i < count; i++) {
        float expected = (float)(prevPeer * 4000 + (int)(count + i));
        if (hostRecv[count + i] != expected) {
          s21Pass = false;
          break;
        }
      }
      // Region 2 (INTER): skip validation on single-node (INTER requires
      // inter-node peers)
      if (nNodes > 1) {
        int prevNode21 = (nodeIdx + nNodes - 1) % nNodes;
        int prevNodeWorld21 = prevNode21 * intraSize + intraRank;
        for (size_t i = 0; i < count; i++) {
          float expected =
              (float)(prevNodeWorld21 * 4000 + (int)(2 * count + i));
          if (hostRecv[2 * count + i] != expected) {
            s21Pass = false;
            break;
          }
        }
      }
      RPRINTF("S21 DevPut(Warp,INTRA+WORLD+INTER): %s\n",
              s21Pass ? "PASS" : "FAIL");
      allPass &= s21Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (proc == 0)
      printf("#\n");
  }

  // =========================================================================
  // S23: Team-resolution correctness test (size-independent, run once)
  // =========================================================================
  {
    MPI_Barrier(MPI_COMM_WORLD);
    // S23 needs space for (intraSize + totalProcs + nNodes) floats in recvBuff
    size_t s23Size = ((size_t)intraSize + (size_t)totalProcs + (size_t)nNodes) *
                     sizeof(float);

    // Pre-fill sendBuff[0] = my world rank as tag
    float myTag = (float)proc;
    FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, &myTag, sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(
        devHandle->deviceMemset(recvBuff, 0, s23Size, flagcxMemDevice, stream));
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
    float *s23Recv = new float[intraSize + totalProcs + nNodes];
    FLAGCXCHECK(devHandle->deviceMemcpy(s23Recv, recvBuff, s23Size,
                                        flagcxMemcpyDeviceToHost, stream));

    bool s23Pass = (hostRes == 1);
    // INTRA region: slot[prevIntraRank] should contain prevIntraRank's world
    // rank
    int prevIntra23 = (intraRank + intraSize - 1) % intraSize;
    float expectedIntra23 = (float)(intraBase + prevIntra23);
    if (s23Recv[prevIntra23] != expectedIntra23) {
      printf("[rank %d] S23 INTRA FAIL: slot[%d] = %.1f, expected %.1f\n", proc,
             prevIntra23, s23Recv[prevIntra23], expectedIntra23);
      s23Pass = false;
    }

    // WORLD region: slot[prevWorldRank] should contain prevWorldRank
    int prevWorld23 = (proc + totalProcs - 1) % totalProcs;
    float expectedWorld23 = (float)prevWorld23;
    if (s23Recv[intraSize + prevWorld23] != expectedWorld23) {
      printf("[rank %d] S23 WORLD FAIL: slot[%d] = %.1f, expected %.1f\n", proc,
             intraSize + prevWorld23, s23Recv[intraSize + prevWorld23],
             expectedWorld23);
      s23Pass = false;
    }

    // INTER region: skip validation on single-node (INTER requires inter-node
    // peers)
    if (nNodes > 1) {
      int prevNode23 = (nodeIdx + nNodes - 1) % nNodes;
      float expectedInter23 = (float)(prevNode23 * intraSize + intraRank);
      if (s23Recv[intraSize + totalProcs + prevNode23] != expectedInter23) {
        printf("[rank %d] S23 INTER FAIL: slot[%d] = %.1f, expected %.1f\n",
               proc, intraSize + totalProcs + prevNode23,
               s23Recv[intraSize + totalProcs + prevNode23], expectedInter23);
        s23Pass = false;
      }
    }

    RPRINTF("S23 TeamResolution(INTRA+WORLD+INTER): %s\n",
            s23Pass ? "PASS" : "FAIL");
    allPass &= s23Pass;
    delete[] s23Recv;
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
