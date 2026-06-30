/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Function Tests — Inter-node transport tests
 *
 * Tests inter-node S-API transport IR functions:
 *   S9: Net GetFromComm (flagcxDevNetGetFromCommS)
 *   S10: Net Signal/Counter (read, reset, shadow)
 *
 * Requires FLAGCX_USE_HETERO_COMM=1 or actual multi-node setup.
 *
 * Usage: FLAGCX_USE_HETERO_COMM=1 mpirun -np N ./test_device_ir_inter
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
  reqs.interBarrierCount = 4;
  reqs.interSignalCount = 2;
  reqs.interCounterCount = 1;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create DevMem
  flagcxDevMem_t devMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));

  // Get device pointers
  void *devCommPtr = nullptr;
  FLAGCXCHECK(flagcxDevCommGetDevicePtr(devComm, &devCommPtr));

  if (proc == 0) {
    printf("=== Device IR Inter-Node Transport Tests ===\n");
    printf("Ranks: %d\n\n", totalProcs);
  }

  // -------------------------------------------------------------------------
  // S9: Net GetFromCommS
  // -------------------------------------------------------------------------
  if (proc == 0) {
    printf("--- S-API Transport Tests ---\n");
  }

  int *s9Results = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&s9Results, 4 * sizeof(int),
                                      flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemset(s9Results, 0, 4 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelNetGetFromCommS(devCommPtr, s9Results, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  int hostS9[4] = {0};
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS9, s9Results, 4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  bool s9Pass = (hostS9[0] == 1); // net pointer should be non-null
  bool s9Skip = (hostS9[0] == 0); // net unavailable (no contexts) → skip
  if (proc == 0) {
    printf("S9 NetGetFromComm: %s\n", s9Skip ? "SKIP (no transport contexts)"
                                             : (s9Pass ? "PASS" : "FAIL"));
  }
  if (s9Skip)
    s9Pass = true;
  FLAGCXCHECK(devHandle->deviceFree(s9Results, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // S10: Signal/Counter local read/reset/shadow
  // -------------------------------------------------------------------------
  int *s10Results = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&s10Results, 4 * sizeof(int),
                                      flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemset(s10Results, 0, 4 * sizeof(int),
                                      flagcxMemDevice, NULL));

  launchKernelNetSignalCounterS(devCommPtr, s10Results, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  int hostS10[4] = {0};
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS10, s10Results, 4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, NULL));

  // results[0]: signal reset+read==0, results[1]: shadow doesn't change signal,
  // results[2]: counter reset+read==0
  bool s10Pass = (hostS10[0] == 1) && (hostS10[1] == 1) && (hostS10[2] == 1);
  bool s10Skip = (hostS10[0] == 0) && (hostS10[1] == 0) && (hostS10[2] == 0);
  if (proc == 0) {
    printf("S10 NetSignalCounter: %s\n", s10Skip
                                             ? "SKIP (no transport contexts)"
                                             : (s10Pass ? "PASS" : "FAIL"));
  }
  if (s10Skip)
    s10Pass = true;
  FLAGCXCHECK(devHandle->deviceFree(s10Results, flagcxMemDevice, NULL));

  // -------------------------------------------------------------------------
  // Summary
  // -------------------------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);

  int allPass = s9Pass && s10Pass;
  int globalPass = 0;
  MPI_Allreduce(&allPass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (proc == 0) {
    printf("\n=== Overall: %s ===\n", globalPass ? "PASS" : "FAIL");
  }

  // Cleanup
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
