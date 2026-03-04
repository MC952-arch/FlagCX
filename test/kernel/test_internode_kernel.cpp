#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"
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
  int local_register = args.getLocalRegister();

  flagcxHandlerGroup_t handler;
  FLAGCXCHECK(flagcxHandleInit(&handler));
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
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, uniqueId, proc));

  flagcxStream_t stream;
  devHandle->streamCreate(&stream);

  void *sendbuff = nullptr, *recvbuff = nullptr, *hello;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  size_t count;
  timer tim;

  if (local_register == 2) {
    // Window mode: VMM alloc with comm (for flagcxCommWindowRegister later)
    FLAGCXCHECK(flagcxMemAlloc(&sendbuff, max_bytes, comm));
    FLAGCXCHECK(flagcxMemAlloc(&recvbuff, max_bytes, comm));
  } else if (local_register == 1) {
    // Zero-copy: alloc + register for NIC RDMA access
    FLAGCXCHECK(flagcxMemAlloc(&sendbuff, max_bytes));
    FLAGCXCHECK(flagcxMemAlloc(&recvbuff, max_bytes));
    FLAGCXCHECK(flagcxCommRegister(comm, sendbuff, max_bytes, &sendHandle));
    FLAGCXCHECK(flagcxCommRegister(comm, recvbuff, max_bytes, &recvHandle));
  } else {
    // Unregistered
    devHandle->deviceMalloc(&sendbuff, max_bytes, flagcxMemDevice, NULL);
    devHandle->deviceMalloc(&recvbuff, max_bytes, flagcxMemDevice, NULL);
  }
  hello = malloc(max_bytes);
  memset(hello, 0, max_bytes);

  // Create device communicator for P2P demo
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create raw device memory handles for send/recv buffers
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(NULL, sendbuff, max_bytes, NULL, &sendMem));
  FLAGCXCHECK(flagcxDevMemCreate(NULL, recvbuff, max_bytes, NULL, &recvMem));

  // Warm-up for large size
  // count is per-peer, total buffer = nRanks * count elements
  for (int i = 0; i < num_warmup_iters; i++) {
    // launch p2p kernel
    FLAGCXCHECK(flagcxInterP2pDemo(sendMem, recvMem,
                                   max_bytes / sizeof(float) / totalProcs,
                                   DATATYPE, devComm, stream));
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < num_warmup_iters; i++) {
    // launch p2p kernel
    FLAGCXCHECK(flagcxInterP2pDemo(sendMem, recvMem,
                                   min_bytes / sizeof(float) / totalProcs,
                                   DATATYPE, devComm, stream));
  }
  devHandle->streamSynchronize(stream);

  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    // count is per-peer elements; total buffer = nRanks * count
    count = size / sizeof(float) / totalProcs;

    // Initialize sendbuff: all elements = proc (my rank)
    // After alltoall, recvbuff[p] should contain p's rank value
    float *helloFloat = (float *)hello;
    size_t totalElements = size / sizeof(float);
    for (size_t i = 0; i < totalElements; i++) {
      helloFloat[i] = (float)proc;
    }

    devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                            NULL);

    // Print sendbuff from rank 0 and last rank
    if (color == 0 && print_buffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d sendbuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", helloFloat[p * count]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < num_iters; i++) {
      // launch p2p kernel
      FLAGCXCHECK(flagcxInterP2pDemo(sendMem, recvMem, count, DATATYPE, devComm,
                                     stream));
    }
    devHandle->streamSynchronize(stream);

    double elapsed_time = tim.elapsed() / num_iters;
    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsed_time /= worldSize;

    double base_bw = (double)(size) / 1.0E9 / elapsed_time;
    double alg_bw = base_bw;
    double factor = 1;
    double bus_bw = base_bw * factor;
    if (proc == 0 && color == 0) {
      printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: %lf "
             "GB/s; Bus bandwidth: %lf GB/s\n",
             size, elapsed_time, alg_bw, bus_bw);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    memset(hello, 0, size);
    devHandle->deviceMemcpy(hello, recvbuff, size, flagcxMemcpyDeviceToHost,
                            NULL);
    // Print recvbuff from rank 0 and last rank
    // Expected: 0 1 2 3 ... nRanks-1 (one value per peer)
    if (color == 0 && print_buffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d recvbuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", helloFloat[p * count]);
      }
      printf("\n");
    }
  }

  // ==========================================================================
  // GIN AlltoAll test (requires -R 2 for window registration)
  // ==========================================================================
  if (local_register == 2) {
    if (proc == 0 && color == 0) {
      printf("\n# GIN AlltoAll test (window-mode devNet)\n");
    }

    // Register windows for GIN
    flagcxWindow_t sendWin = nullptr, recvWin = nullptr;
    FLAGCXCHECK(
        flagcxCommWindowRegister(comm, sendbuff, max_bytes, &sendWin, 0));
    FLAGCXCHECK(
        flagcxCommWindowRegister(comm, recvbuff, max_bytes, &recvWin, 0));

    // Create device communicator with GIN barrier + signal requirements
    flagcxDevCommRequirements ginReqs =
        FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
    ginReqs.fields[2] = FLAGCX_DEVICE_CTA_COUNT; // ginBarrierCount
    ginReqs.fields[3] = 1;                       // ginSignalCount
    flagcxDevComm_t ginDevComm = nullptr;
    FLAGCXCHECK(flagcxDevCommCreate(comm, &ginReqs, &ginDevComm));

    // Create window-mode device memory handles
    flagcxDevMem_t ginSendMem = nullptr, ginRecvMem = nullptr;
    FLAGCXCHECK(
        flagcxDevMemCreate(comm, sendbuff, max_bytes, sendWin, &ginSendMem));
    FLAGCXCHECK(
        flagcxDevMemCreate(comm, recvbuff, max_bytes, recvWin, &ginRecvMem));

    for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
      count = size / sizeof(float) / totalProcs;

      // Initialize sendbuff: sendbuff[r * count + i] = proc * 1000 + r * 100 +
      // i After alltoall: recvbuff[src * count + i] = src * 1000 + proc * 100 +
      // i
      float *helloFloat = (float *)hello;
      for (int r = 0; r < totalProcs; r++) {
        for (size_t i = 0; i < count; i++) {
          helloFloat[r * count + i] = (float)(proc * 1000 + r * 100 + (int)i);
        }
      }
      devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                              NULL);
      memset(hello, 0, size);
      devHandle->deviceMemcpy(recvbuff, hello, size, flagcxMemcpyHostToDevice,
                              NULL);

      MPI_Barrier(MPI_COMM_WORLD);

      tim.reset();
      for (int i = 0; i < num_iters; i++) {
        FLAGCXCHECK(flagcxGinAlltoAllDemo(ginSendMem, ginRecvMem, count,
                                          DATATYPE, ginDevComm, stream));
      }
      devHandle->streamSynchronize(stream);
      double elapsed_time = tim.elapsed() / num_iters;

      // Verify correctness
      memset(hello, 0, size);
      devHandle->deviceMemcpy(hello, recvbuff, size, flagcxMemcpyDeviceToHost,
                              NULL);
      helloFloat = (float *)hello;
      bool correct = true;
      for (int src = 0; src < totalProcs && correct; src++) {
        for (size_t i = 0; i < count && correct; i++) {
          float expected = (float)(src * 1000 + proc * 100 + (int)i);
          if (helloFloat[src * count + i] != expected) {
            correct = false;
            if (proc == 0) {
              printf("  MISMATCH at recvbuff[%d*%zu+%zu]: got %.0f expected "
                     "%.0f\n",
                     src, count, i, helloFloat[src * count + i], expected);
            }
          }
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      elapsed_time /= worldSize;
      double bw = (double)(size) / 1.0E9 / elapsed_time;

      if (proc == 0 && color == 0) {
        printf("GIN AlltoAll %zu bytes; %.3lf us; %.3lf GB/s; %s\n", size,
               elapsed_time * 1e6, bw, correct ? "PASS" : "FAIL");
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Cleanup GIN resources
    FLAGCXCHECK(flagcxDevMemDestroy(comm, ginSendMem));
    FLAGCXCHECK(flagcxDevMemDestroy(comm, ginRecvMem));
    FLAGCXCHECK(flagcxDevCommDestroy(comm, ginDevComm));
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin));
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin));
  }

  // Destroy stream first (sync any pending work)
  devHandle->streamDestroy(stream);

  // Destroy raw device memory handles
  FLAGCXCHECK(flagcxDevMemDestroy(NULL, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(NULL, recvMem));

  // Destroy device communicator before comm destroy
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));

  if (local_register == 1) {
    // deregister buffer (must be done before comm destroy)
    FLAGCXCHECK(flagcxCommDeregister(comm, sendHandle));
    FLAGCXCHECK(flagcxCommDeregister(comm, recvHandle));
  }

  // Free -R 2 (VMM) buffers before comm destroy (flagcxMemFree needs comm
  // alive)
  if (local_register == 2) {
    FLAGCXCHECK(flagcxMemFree(sendbuff, comm));
    FLAGCXCHECK(flagcxMemFree(recvbuff, comm));
  }

  // Destroy comm to stop kernel proxy thread BEFORE freeing device memory
  // The kernel proxy thread holds a CUDA stream that can interfere with
  // deviceFree
  FLAGCXCHECK(flagcxCommDestroy(comm));

  if (local_register == 1) {
    FLAGCXCHECK(flagcxMemFree(sendbuff));
    FLAGCXCHECK(flagcxMemFree(recvbuff));
  } else if (local_register == 0) {
    devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL);
    devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL);
  }
  free(hello);
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return 0;
}
