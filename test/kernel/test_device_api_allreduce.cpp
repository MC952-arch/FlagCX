/*************************************************************************
 * Benchmark for FlagCX Intra-node AllReduce using FlagCX Device API.
 *
 * Tests correctness: each rank fills its buffer with (rank+1), then
 * AllReduce(sum) produces nRanks*(nRanks+1)/2 on every element.
 *
 * Tests performance: warmup + timed iterations over multiple message sizes.
 *
 * Usage: mpirun -np <nGPUs> ./test_device_api_allreduce [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -w <warmup>    -n <iters>     -p <printbuffer 0/1>
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"
#include <cmath>
#include <cstring>
#include <iostream>

#define DATATYPE flagcxFloat

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t min_bytes = args.getMinBytes();
  size_t max_bytes = args.getMaxBytes();
  int step_factor = args.getStepFactor();
  int num_warmup_iters = args.getWarmupIters();
  int num_iters = args.getTestIters();
  int print_buffer = args.isPrintBuffer();
  uint64_t split_mask = args.getSplitMask();

  flagcxHandlerGroup_t handler;
  flagcxHandleInit(&handler);
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, split_mask);

  int nGpu;
  devHandle->getDeviceCount(&nGpu);
  devHandle->setDevice(worldRank % nGpu);

  if (proc == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommInitRank(&comm, totalProcs, uniqueId, proc);

  // Create device communicator for custom kernel usage
  flagcxDevComm_t devComm = nullptr;
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 36; // FLAGCX_DEVICE_CTA_COUNT
  reqs.lsaMultimem = 0;
  flagcxDevCommCreate(comm, &reqs, &devComm);

  flagcxStream_t stream;
  devHandle->streamCreate(&stream);

  // Allocate device buffers
  void *sendbuff, *recvbuff;
  devHandle->deviceMalloc(&sendbuff, max_bytes, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, max_bytes, flagcxMemDevice, NULL);

  // Allocate window-registered buffer via flagcxMemAlloc +
  // flagcxCommWindowRegister
  void *windowBuff = nullptr;
  flagcxWindow_t win = nullptr;
  flagcxMemAlloc(&windowBuff, max_bytes, comm);
  flagcxCommWindowRegister(comm, windowBuff, max_bytes, &win, 0);

  // Host buffer for initialization and verification
  void *hostbuff = malloc(max_bytes);

  if (proc == 0 && color == 0) {
    printf("# FlagCX Device API Intra-node AllReduce Benchmark\n");
    printf("# nRanks: %d\n", totalProcs);
    printf("# %-12s %-14s %-14s %-14s %-8s\n", "Size(B)", "Time(us)",
           "AlgBW(GB/s)", "BusBW(GB/s)", "Correct");
  }

  // Warmup with max size
  {
    size_t count = max_bytes / sizeof(float);
    for (int i = 0; i < num_warmup_iters; i++) {
      devHandle->deviceMemcpy(windowBuff, sendbuff, count * sizeof(float),
                              flagcxMemcpyDeviceToDevice, stream);
      flagcxIntraAllReduceDemo(windowBuff, win, count, DATATYPE, devComm,
                               stream);
      devHandle->deviceMemcpy(recvbuff, windowBuff, count * sizeof(float),
                              flagcxMemcpyDeviceToDevice, stream);
    }
    devHandle->streamSynchronize(stream);
  }

  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    size_t count = size / sizeof(float);
    if (count == 0)
      count = 1;
    size_t bytes = count * sizeof(float);

    // Initialize: each rank fills sendbuff with (rank + 1)
    float *hbuf = (float *)hostbuff;
    for (size_t i = 0; i < count; i++) {
      hbuf[i] = (float)(proc + 1);
    }
    devHandle->deviceMemcpy(sendbuff, hostbuff, bytes, flagcxMemcpyHostToDevice,
                            NULL);

    MPI_Barrier(MPI_COMM_WORLD);

    // Timed iterations
    timer tim;
    for (int i = 0; i < num_iters; i++) {
      devHandle->deviceMemcpy(windowBuff, sendbuff, bytes,
                              flagcxMemcpyDeviceToDevice, stream);
      flagcxIntraAllReduceDemo(windowBuff, win, count, DATATYPE, devComm,
                               stream);
      devHandle->deviceMemcpy(recvbuff, windowBuff, bytes,
                              flagcxMemcpyDeviceToDevice, stream);
    }
    devHandle->streamSynchronize(stream);
    double elapsed = tim.elapsed() / num_iters;

    // Reduce elapsed time across ranks for consistent reporting
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsed /= worldSize;

    // Bandwidth calculation
    // AllReduce: 2*(N-1)/N * size for bus bandwidth
    double algBW = (double)size / 1.0e9 / elapsed;
    double busFW = algBW * 2.0 * (totalProcs - 1) / (double)totalProcs;

    // Verify correctness: expected value = sum(1..nRanks) = nRanks*(nRanks+1)/2
    memset(hostbuff, 0, bytes);
    devHandle->deviceMemcpy(hostbuff, recvbuff, bytes, flagcxMemcpyDeviceToHost,
                            NULL);

    float expected = (float)(totalProcs * (totalProcs + 1)) / 2.0f;
    int correct = 1;
    for (size_t i = 0; i < count && correct; i++) {
      if (fabsf(hbuf[i] - expected) > 1e-3f) {
        correct = 0;
        if (print_buffer) {
          printf("rank%d: MISMATCH at [%zu]: got %.4f, expected %.4f\n", proc,
                 i, hbuf[i], expected);
        }
      }
    }
    // Global correctness check
    MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (proc == 0 && color == 0) {
      printf("  %-12zu %-14.2f %-14.4f %-14.4f %-8s\n", size, elapsed * 1e6,
             algBW, busFW, correct ? "PASS" : "FAIL");
    }

    if (print_buffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d result[0..3]:", proc);
      for (size_t i = 0; i < 4 && i < count; i++) {
        printf(" %.2f", hbuf[i]);
      }
      printf(" (expected: %.2f)\n", expected);
    }
  }

  // Cleanup
  flagcxDevCommDestroy(devComm);
  flagcxCommWindowDeregister(comm, win);
  flagcxMemFree(windowBuff, comm);
  devHandle->streamDestroy(stream);
  devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL);
  devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL);
  free(hostbuff);

  flagcxCommDestroy(comm);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}
