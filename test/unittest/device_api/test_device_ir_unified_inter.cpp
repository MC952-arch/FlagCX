/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Unified Inter Suite Tests — INTER + WORLD teams
 * Tests one-sided operations with inter-node communication.
 *
 * Tests S16–S21 (6 combinations: 3 coop × 2 teams):
 *   S16: DevBarrier — INTER + WORLD (merged)
 *   S17: DevTeamResolution — INTER + WORLD
 *   S18: DevPut — INTER + WORLD
 *   S19: DevGet — INTER + WORLD
 *   S20: DevSignalStandalone — INTER + WORLD
 *   S21: DevPutSignalWait — INTER + WORLD
 *
 * Requirements:
 *   - Multi-node with 2+ nodes (requires INTER team)
 *   - FLAGCX_USE_HETERO_COMM=1 (for DevComm)
 *
 * Usage: mpirun -np N ./test_device_ir_unified_inter [options]
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
  // Inter suite uses 6 combinations (INTER + WORLD) for most tests
  // S18 uses 8 signal slots (includes extra BLOCK single-leader patterns)
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount =
      8; // 8 slots for S18 (6 standard + 2 BLOCK single-leader)
  reqs.interCounterCount = 1;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Allocate send/recv buffers (8x for S18's 8-combination regions)
  size_t bufSize = maxBytes * 8;
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

  // Host scratch - allocate 8× buffers for S18's 8 combinations
  float *hostSend = new float[bufSize * 8 / sizeof(float)];
  float *hostRecv = new float[bufSize * 8 / sizeof(float)];

  // Team geometry
  int intraSize = comm->localRanks;
  int intraRank = comm->localRank;
  int nNodes = totalProcs / intraSize;
  int nodeIdx = proc / intraSize;

  if (proc == 0) {
    printf("=== Device IR Unified Inter Suite (INTER + WORLD teams) ===\n");
    printf("Ranks: %d, Nodes: %d, IntraSize: %d\n\n", totalProcs, nNodes,
           intraSize);
  }

  if (nNodes < 2) {
    if (proc == 0) {
      printf("ERROR: Inter suite requires multi-node setup (nNodes >= 2)\n");
      printf("Current configuration: nNodes = %d\n", nNodes);
    }
    MPI_Finalize();
    return 1;
  }

  bool allPass = true;

  // =========================================================================
  // Main test loop
  // =========================================================================
  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t count = size / sizeof(float);
    if (count == 0)
      count = 1;
    size_t bytes = count * sizeof(float);

    if (proc == 0)
      printf("# Size = %zu bytes, count = %zu\n", bytes, count);

    MPI_Barrier(MPI_COMM_WORLD);

    // =======================================================================
    // S16: DevBarrier — INTER + WORLD (merged)
    // =======================================================================
    {
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0,
                                          FLAGCX_DEVICE_CTA_COUNT * sizeof(int),
                                          flagcxMemDevice, stream));
      launchKernelDevBarrierInterWorldS(devCommPtr, devResults, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      int hostResults[FLAGCX_DEVICE_CTA_COUNT];
      FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                          FLAGCX_DEVICE_CTA_COUNT * sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      bool s16Pass = true;
      for (int i = 0; i < FLAGCX_DEVICE_CTA_COUNT; i++) {
        if (hostResults[i] != 1) {
          s16Pass = false;
          break;
        }
      }
      RPRINTF("S16 DevBarrier(INTER+WORLD): %s\n", s16Pass ? "PASS" : "FAIL");
      allPass &= s16Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // =======================================================================
    // S17: DevTeamResolution — INTER + WORLD (8 combinations)
    // =======================================================================
    {
      int maxRanks = totalProcs;
      if (nNodes > maxRanks)
        maxRanks = nNodes;
      size_t s17Size = 6 * maxRanks * sizeof(float);

      float myTag = (float)proc;
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, &myTag, sizeof(float),
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, s17Size, flagcxMemDevice,
                                          stream));
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevTeamResolutionInterWorldS(devCommPtr, recvMemPtr,
                                               sendMemPtr, devResults, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      float *s17Recv = new float[6 * maxRanks];
      FLAGCXCHECK(devHandle->deviceMemcpy(s17Recv, recvBuff, s17Size,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s17Pass = (hostRes == 1);
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;

      for (int combo = 0; combo < 6 && s17Pass; combo++) {
        int teamIdx = combo % 2; // 0=INTER, 1=WORLD
        size_t baseOff = combo * maxRanks;

        if (teamIdx == 0) { // INTER
          float expected = (float)(prevNode * intraSize + intraRank);
          if (s17Recv[baseOff + prevNode] != expected) {
            s17Pass = false;
          }
        } else { // WORLD
          float expected = (float)prevWorld;
          if (s17Recv[baseOff + prevWorld] != expected) {
            s17Pass = false;
          }
        }
      }

      RPRINTF("S17 DevTeamResolution(INTER+WORLD): %s\n",
              s17Pass ? "PASS" : "FAIL");
      allPass &= s17Pass;
      delete[] s17Recv;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // =======================================================================
    // S18: DevPut — INTER + WORLD
    // =======================================================================
    {
      // Initialize: fill sendBuff with rank pattern
      for (size_t i = 0; i < 6 * count; i++)
        hostSend[i] = (float)(proc * 1000 + i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, 6 * bytes,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 6 * bytes,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutInterWorldS(devCommPtr, recvMemPtr, sendMemPtr,
                                    devResults, bytes, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 6 * bytes,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s18Pass = (hostRes == 1);
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;
      int prevNodeBase = prevNode * intraSize + intraRank;

      for (int combo = 0; combo < 6 && s18Pass; combo++) {
        int teamIdx = combo % 2;
        size_t off = combo * count;
        int senderRank = (teamIdx == 0) ? prevNodeBase : prevWorld;

        for (size_t i = 0; i < count && s18Pass; i++) {
          float expected = (float)(senderRank * 1000 + off + i);
          if (hostRecv[off + i] != expected) {
            s18Pass = false;
          }
        }
      }

      RPRINTF("S18 DevPut(INTER+WORLD): %s\n", s18Pass ? "PASS" : "FAIL");
      allPass &= s18Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // =======================================================================
    // S19: DevGet — INTER + WORLD
    // =======================================================================
    {
      for (size_t i = 0; i < 6 * count; i++)
        hostSend[i] = (float)(proc * 2000 + i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, 6 * bytes,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 6 * bytes,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevGetInterWorldS(devCommPtr, sendMemPtr, recvMemPtr,
                                    devResults, bytes, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 6 * bytes,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s19Pass = (hostRes == 1);
      int nextNode = (nodeIdx + 1) % nNodes;
      int nextWorld = (proc + 1) % totalProcs;
      int nextNodeBase = nextNode * intraSize + intraRank;

      for (int combo = 0; combo < 6 && s19Pass; combo++) {
        int teamIdx = combo % 2;
        size_t off = combo * count;
        int sourceRank = (teamIdx == 0) ? nextNodeBase : nextWorld;

        for (size_t i = 0; i < count && s19Pass; i++) {
          float expected = (float)(sourceRank * 2000 + off + i);
          if (hostRecv[off + i] != expected) {
            s19Pass = false;
          }
        }
      }

      RPRINTF("S19 DevGet(INTER+WORLD): %s\n", s19Pass ? "PASS" : "FAIL");
      allPass &= s19Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // =======================================================================
    // S20: DevSignalStandalone — INTER + WORLD
    // =======================================================================
    {
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevSignalStandaloneInterWorldS(devCommPtr, devResults,
                                                 stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      bool s20Pass = (hostRes == 1);
      RPRINTF("S20 DevSignalStandalone(INTER+WORLD): %s\n",
              s20Pass ? "PASS" : "FAIL");
      allPass &= s20Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // =======================================================================
    // S21: DevPutSignalWait — INTER + WORLD (8 combinations including
    // single-leader)
    // =======================================================================
    {
      for (size_t i = 0; i < 8 * count; i++)
        hostSend[i] = (float)(proc * 3000 + i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, 8 * bytes,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 8 * bytes,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutSignalWaitInterWorldS(
          devCommPtr, recvMemPtr, sendMemPtr, devResults, bytes, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 8 * bytes,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s21Pass = (hostRes == 1);
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;
      int prevNodeBase = prevNode * intraSize + intraRank;

      for (int combo = 0; combo < 8 && s21Pass; combo++) {
        int teamIdx = combo % 2;
        size_t off = combo * count;
        int senderRank = (teamIdx == 0) ? prevNodeBase : prevWorld;

        for (size_t i = 0; i < count && s21Pass; i++) {
          float expected = (float)(senderRank * 3000 + off + i);
          if (hostRecv[off + i] != expected) {
            s21Pass = false;
          }
        }
      }

      RPRINTF("S21 DevPutSignalWait(INTER+WORLD): %s\n",
              s21Pass ? "PASS" : "FAIL");
      allPass &= s21Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (proc == 0)
      printf("\n");
  }

  // Cleanup
  delete[] hostSend;
  delete[] hostRecv;
  FLAGCXCHECK(devHandle->deviceFree(devResults, flagcxMemDevice, NULL));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, recvMem));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin, memAllocator));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin, memAllocator));
  FLAGCXCHECK(flagcxMemFree(sendBuff));
  FLAGCXCHECK(flagcxMemFree(recvBuff));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));

  if (proc == 0) {
    printf("=== Final Result: %s ===\n", allPass ? "ALL PASS" : "SOME FAILED");
  }

  MPI_Finalize();
  return allPass ? 0 : 1;
}
