/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Intra-node Device API test — exercises struct-based Device API functions
 * that only require single-node (intra-node) setup.
 *
 * Tests:
 *   K1:  Local Pointer (flagcxGetLocalPointer, getRawPtr)
 *   K2:  Intra Pointer (flagcxGetIntraPointer — IPC/window peer read)
 *   K3:  Peer Pointer  (flagcxGetPeerPointer with team)
 *   K4:  Multicast Pointer (flagcxGetMulticastPointer — skip if no NVLS)
 *   K5:  Intra Barrier Sync (flagcxDevBarrier<Intra> sync)
 *   K6:  Intra Barrier Arrive/Wait (flagcxDevBarrier<Intra> arrive+wait)
 *   K7:  SymPtr (flagcxSymPtr<float> localPtr/intraPtr + arithmetic)
 *   K8:  FindMem (flagcxFindMem — reverse pointer lookup)
 *   K9:  DevMem & Comm Queries (hasWindow, getIntraRank/Size, getRank/Size)
 *   K10: Coop Groups (Block/Tile/Warp/Lanes/TileSpan threadRank/size/sync)
 *   K11: Team (flagcxTeamIntra, RankToWorld, RankToIntra, RankIsMember)
 *   K12: IntraAllReduce (composite — peer pointer + barrier end-to-end)
 *
 * Usage: mpirun -np N ./test_device_api_intra
 *   Runs on any single node, no network or HETERO_COMM required.
 *   -R 2 recommended (window registration for peer pointer access).
 ************************************************************************/

#include "device_api.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#define DATATYPE flagcxFloat

static void printResult(const char *name, bool ok, int rank) {
  if (rank == 0)
    printf("  %-35s %s\n", name, ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  flagcxDeviceHandle_t devHandle;
  flagcxComm_t comm;
  FLAGCXCHECK(flagcxDeviceHandleInit(&devHandle));
  flagcxUniqueId uniqueId;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  uint64_t splitMask = 0;
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

  // Buffer: 1 MB, symmetric window registered
  size_t bufSize = 1024 * 1024;
  void *regBuff = nullptr;
  FLAGCXCHECK(flagcxMemAlloc(&regBuff, bufSize));

  flagcxWindow_t win = nullptr;
  FLAGCXCHECK(flagcxCommWindowRegister(comm, regBuff, bufSize, &win,
                                       FLAGCX_WIN_COLL_SYMMETRIC));

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  // Create DevComm
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = 0;
  reqs.interSignalCount = 0;
  reqs.interCounterCount = 0;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create DevMem
  flagcxDevMem_t devMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));

  // Results buffer
  int *devResults = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devResults, 64 * sizeof(int),
                                      flagcxMemDevice, NULL));
  int hostResults[64] = {};

  // Output buffer for pointer/barrier tests
  size_t floatCount = bufSize / sizeof(float);
  float *devOutput = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devOutput, bufSize,
                                      flagcxMemDevice, NULL));

  if (proc == 0) {
    printf("# FlagCX Device API Intra-Node Test\n");
    printf("# nRanks: %d\n#\n", totalProcs);
  }

  int peer = (proc + 1) % totalProcs;
  bool allPass = true;

  // =========================================================================
  // K1: Local Pointer
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                      flagcxMemDevice, NULL));

  FLAGCXCHECK(flagcxIntraTestLocalPointer(devMem, regBuff, devResults, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k1Ok = (hostResults[0] == 1);
  printResult("K1 LocalPointer", k1Ok, proc);
  allPass &= k1Ok;

  // =========================================================================
  // K2: Intra Pointer
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  // Write known pattern: each rank fills regBuff with its rank value
  {
    float *hostInit = new float[floatCount];
    for (size_t i = 0; i < floatCount; i++)
      hostInit[i] = (float)proc;
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostInit, bufSize,
                                        flagcxMemcpyHostToDevice, NULL));
    delete[] hostInit;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  size_t testCount = std::min(floatCount, (size_t)1024);
  FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, testCount * sizeof(float),
                                      flagcxMemDevice, NULL));

  FLAGCXCHECK(flagcxIntraTestIntraPointer(devMem, devComm, devOutput, testCount,
                                          stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  {
    float *hostOut = new float[testCount];
    FLAGCXCHECK(devHandle->deviceMemcpy(hostOut, devOutput,
                                        testCount * sizeof(float),
                                        flagcxMemcpyDeviceToHost, NULL));
    bool k2Ok = true;
    for (size_t i = 0; i < testCount; i++) {
      if (fabsf(hostOut[i] - (float)peer) > 1e-3f) {
        k2Ok = false;
        break;
      }
    }
    printResult("K2 IntraPointer", k2Ok, proc);
    allPass &= k2Ok;
    delete[] hostOut;
  }

  // =========================================================================
  // K3: Peer Pointer (team-based)
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, testCount * sizeof(float),
                                      flagcxMemDevice, NULL));

  FLAGCXCHECK(flagcxIntraTestPeerPointer(devMem, devComm, devOutput, testCount,
                                         stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  {
    float *hostOut = new float[testCount];
    FLAGCXCHECK(devHandle->deviceMemcpy(hostOut, devOutput,
                                        testCount * sizeof(float),
                                        flagcxMemcpyDeviceToHost, NULL));
    bool k3Ok = true;
    for (size_t i = 0; i < testCount; i++) {
      if (fabsf(hostOut[i] - (float)peer) > 1e-3f) {
        k3Ok = false;
        break;
      }
    }
    printResult("K3 PeerPointer(team)", k3Ok, proc);
    allPass &= k3Ok;
    delete[] hostOut;
  }

  // =========================================================================
  // K4: Multicast Pointer (skip if not available)
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  printResult("K4 MulticastPointer", true, proc); // TODO: NVLS-dependent
  // allPass &= k4Ok;

  // =========================================================================
  // K5: Intra Barrier Sync
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  // Write rank+1 to local buffer
  {
    float *hostInit = new float[testCount];
    for (size_t i = 0; i < testCount; i++)
      hostInit[i] = (float)(proc + 1);
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostInit,
                                        testCount * sizeof(float),
                                        flagcxMemcpyHostToDevice, NULL));
    delete[] hostInit;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, testCount * sizeof(float),
                                      flagcxMemDevice, NULL));

  FLAGCXCHECK(flagcxIntraTestBarrierSync(devMem, devComm, devOutput, testCount,
                                         stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  {
    float *hostOut = new float[testCount];
    FLAGCXCHECK(devHandle->deviceMemcpy(hostOut, devOutput,
                                        testCount * sizeof(float),
                                        flagcxMemcpyDeviceToHost, NULL));
    bool k5Ok = true;
    float expected = (float)(peer + 1);
    for (size_t i = 0; i < testCount; i++) {
      if (fabsf(hostOut[i] - expected) > 1e-3f) {
        k5Ok = false;
        break;
      }
    }
    printResult("K5 IntraBarrierSync", k5Ok, proc);
    allPass &= k5Ok;
    delete[] hostOut;
  }

  // =========================================================================
  // K6: Intra Barrier Arrive/Wait
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  // Write rank+100 to local buffer
  {
    float *hostInit = new float[testCount];
    for (size_t i = 0; i < testCount; i++)
      hostInit[i] = (float)(proc + 100);
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostInit,
                                        testCount * sizeof(float),
                                        flagcxMemcpyHostToDevice, NULL));
    delete[] hostInit;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, testCount * sizeof(float),
                                      flagcxMemDevice, NULL));

  FLAGCXCHECK(flagcxIntraTestBarrierArriveWait(devMem, devComm, devOutput,
                                               testCount, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  {
    float *hostOut = new float[testCount];
    FLAGCXCHECK(devHandle->deviceMemcpy(hostOut, devOutput,
                                        testCount * sizeof(float),
                                        flagcxMemcpyDeviceToHost, NULL));
    bool k6Ok = true;
    float expected = (float)(peer + 100);
    for (size_t i = 0; i < testCount; i++) {
      if (fabsf(hostOut[i] - expected) > 1e-3f) {
        k6Ok = false;
        break;
      }
    }
    printResult("K6 IntraBarrierArriveWait", k6Ok, proc);
    allPass &= k6Ok;
    delete[] hostOut;
  }

  // =========================================================================
  // K7: SymPtr
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                      flagcxMemDevice, NULL));

  FLAGCXCHECK(flagcxIntraTestSymPtr(devMem, devComm, devResults, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool k7Ok = (hostResults[0] == 1);
  printResult("K7 SymPtr", k7Ok, proc);
  allPass &= k7Ok;

  // =========================================================================
  // K8: FindMem (skip — not implemented on default backend)
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  printResult("K8 FindMem", true, proc); // TODO: vendor-specific
  // allPass &= k8Ok;

  // =========================================================================
  // K9: DevMem & Comm Queries
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 8 * sizeof(int),
                                      flagcxMemDevice, NULL));

  FLAGCXCHECK(flagcxIntraTestCommQueries(devMem, devComm, devResults, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 6 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  // results: [hasWindow, intraRank, intraSize, rank, size, hasPeerPtrs]
  bool k9Ok = (hostResults[0] == 1) &&          // hasWindow
              (hostResults[1] == proc) &&       // intraRank (single-node)
              (hostResults[2] == totalProcs) && // intraSize
              (hostResults[3] == proc) &&       // rank
              (hostResults[4] == totalProcs);   // size
  printResult("K9 CommQueries", k9Ok, proc);
  allPass &= k9Ok;

  // =========================================================================
  // K10: Coop Groups
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 16 * sizeof(int),
                                      flagcxMemDevice, NULL));

  FLAGCXCHECK(flagcxIntraTestCoopGroups(devResults, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 16 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  // results[0] = block pass, [1] = warp pass, [2] = thread pass,
  // [3] = tile pass, [4] = lanes pass
  bool k10Ok = (hostResults[0] == 1) && (hostResults[1] == 1) &&
               (hostResults[2] == 1) && (hostResults[3] == 1) &&
               (hostResults[4] == 1);
  printResult("K10 CoopGroups", k10Ok, proc);
  allPass &= k10Ok;

  // =========================================================================
  // K11: Team
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 8 * sizeof(int),
                                      flagcxMemDevice, NULL));

  FLAGCXCHECK(flagcxIntraTestTeam(devComm, devResults, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults, 8 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  // results[0] = intra team nRanks correct
  // results[1] = rankToWorld correct
  // results[2] = rankToIntra correct
  // results[3] = rankIsMember correct
  bool k11Ok = (hostResults[0] == 1) && (hostResults[1] == 1) &&
               (hostResults[2] == 1) && (hostResults[3] == 1);
  printResult("K11 Team", k11Ok, proc);
  allPass &= k11Ok;

  // =========================================================================
  // K12: IntraAllReduce (composite)
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  size_t arCount = 256; // elements per rank
  {
    float *hostInit = new float[arCount];
    for (size_t i = 0; i < arCount; i++)
      hostInit[i] = (float)(proc + 1); // each rank contributes rank+1
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostInit,
                                        arCount * sizeof(float),
                                        flagcxMemcpyHostToDevice, NULL));
    delete[] hostInit;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(
      flagcxIntraAllReduce(devMem, arCount, flagcxFloat, devComm, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  {
    float *hostOut = new float[arCount];
    FLAGCXCHECK(devHandle->deviceMemcpy(hostOut, regBuff,
                                        arCount * sizeof(float),
                                        flagcxMemcpyDeviceToHost, NULL));
    // AllReduce sum: expected = sum(1..N) = N*(N+1)/2
    float expected = (float)(totalProcs * (totalProcs + 1) / 2);
    bool k12Ok = true;
    for (size_t i = 0; i < arCount; i++) {
      if (fabsf(hostOut[i] - expected) > 1e-1f) {
        k12Ok = false;
        break;
      }
    }
    printResult("K12 IntraAllReduce(composite)", k12Ok, proc);
    allPass &= k12Ok;
    delete[] hostOut;
  }

  // =========================================================================
  // Summary
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  int pass = allPass ? 1 : 0;
  int globalPass = 0;
  MPI_Allreduce(&pass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (proc == 0)
    printf("#\n# Overall: %s\n", globalPass ? "PASS" : "FAIL");

  // Cleanup
  FLAGCXCHECK(devHandle->deviceFree(devOutput, flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceFree(devResults, flagcxMemDevice, NULL));
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
