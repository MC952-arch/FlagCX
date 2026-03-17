#include "flagcx.h"
#include "tools.h"
#include <cstring>
#include <iostream>

#define DATATYPE flagcxFloat

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int numIters = args.getTestIters();
  int printBuffer = args.isPrintBuffer();
  uint64_t splitMask = args.getSplitMask();
  int localRegister = args.getLocalRegister();

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
             splitComm, splitMask);

  int nGpu;
  devHandle->getDeviceCount(&nGpu);
  devHandle->setDevice(worldRank % nGpu);

  if (proc == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommInitRank(&comm, totalProcs, uniqueId, proc);

  flagcxStream_t stream;
  devHandle->streamCreate(&stream);

  void *sendbuff = nullptr;
  void *recvbuff = nullptr;
  void *hello = nullptr;
  void *sendHandle = nullptr;
  void *recvHandle = nullptr;
  size_t *hSendcounts, *hRecvcounts, *hSdispls, *hRdispls;
  size_t count, sdis, rdis;
  timer tim;

  if (localRegister) {
    // allocate buffer
    flagcxMemAlloc(&sendbuff, maxBytes);
    flagcxMemAlloc(&recvbuff, maxBytes);
    // register buffer
    flagcxCommRegister(comm, sendbuff, maxBytes, &sendHandle);
    flagcxCommRegister(comm, recvbuff, maxBytes, &recvHandle);
  } else {
    devHandle->deviceMalloc(&sendbuff, maxBytes, flagcxMemDevice, NULL);
    devHandle->deviceMalloc(&recvbuff, maxBytes, flagcxMemDevice, NULL);
  }
  hello = malloc(maxBytes);
  memset(hello, 0, maxBytes);
  hSendcounts = (size_t *)malloc(totalProcs * sizeof(size_t));
  hRecvcounts = (size_t *)malloc(totalProcs * sizeof(size_t));
  hSdispls = (size_t *)malloc(totalProcs * sizeof(size_t));
  hRdispls = (size_t *)malloc(totalProcs * sizeof(size_t));

  // Warm-up for large size
  sdis = 0;
  rdis = 0;
  count = (maxBytes / sizeof(float)) / totalProcs;
  for (int i = 0; i < totalProcs; i++) {
    if (proc % 2 == 0) {
      if (i % 2 == 0) {
        hSendcounts[i] = 2 * count;
        hRecvcounts[i] = 2 * count;
        hSdispls[i] = sdis;
        hRdispls[i] = rdis;
        if (i == proc) {
          hSendcounts[i] = 0;
          hRecvcounts[i] = 0;
        }
        sdis += 2 * count;
        rdis += 2 * count;
      } else {
        hSendcounts[i] = 0;
        hRecvcounts[i] = 0;
        hSdispls[i] = sdis;
        hRdispls[i] = rdis;
      }
    } else {
      if (i % 2 == 1) {
        hSendcounts[i] = 2 * count;
        hRecvcounts[i] = 2 * count;
        hSdispls[i] = sdis;
        hRdispls[i] = rdis;
        if (i == proc) {
          hSendcounts[i] = 0;
          hRecvcounts[i] = 0;
        }
        sdis += 2 * count;
        rdis += 2 * count;
      } else {
        hSendcounts[i] = 0;
        hRecvcounts[i] = 0;
        hSdispls[i] = sdis;
        hRdispls[i] = rdis;
      }
    }
  }
  for (int i = 0; i < numWarmupIters; i++) {
    flagcxAlltoAllv(sendbuff, hSendcounts, hSdispls, recvbuff, hRecvcounts,
                    hRdispls, DATATYPE, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  sdis = 0;
  rdis = 0;
  count = (minBytes / sizeof(float)) / totalProcs;
  for (int i = 0; i < totalProcs; i++) {
    if (proc % 2 == 0) {
      if (i % 2 == 0) {
        hSendcounts[i] = 2 * count;
        hRecvcounts[i] = 2 * count;
        hSdispls[i] = sdis;
        hRdispls[i] = rdis;
        if (i == proc) {
          hSendcounts[i] = 0;
          hRecvcounts[i] = 0;
        }
        sdis += 2 * count;
        rdis += 2 * count;
      } else {
        hSendcounts[i] = 0;
        hRecvcounts[i] = 0;
        hSdispls[i] = sdis;
        hRdispls[i] = rdis;
      }
    } else {
      if (i % 2 == 1) {
        hSendcounts[i] = 2 * count;
        hRecvcounts[i] = 2 * count;
        hSdispls[i] = sdis;
        hRdispls[i] = rdis;
        if (i == proc) {
          hSendcounts[i] = 0;
          hRecvcounts[i] = 0;
        }
        sdis += 2 * count;
        rdis += 2 * count;
      } else {
        hSendcounts[i] = 0;
        hRecvcounts[i] = 0;
        hSdispls[i] = sdis;
        hRdispls[i] = rdis;
      }
    }
  }
  for (int i = 0; i < numWarmupIters; i++) {
    flagcxAlltoAllv(sendbuff, hSendcounts, hSdispls, recvbuff, hRecvcounts,
                    hRdispls, DATATYPE, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    sdis = 0;
    rdis = 0;
    count = (size / sizeof(float)) / totalProcs;

    for (int i = 0; i < totalProcs; i++) {
      ((float *)hello)[i * count] = 10 * proc + i;
    }

    devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                            NULL);

    if ((proc == 0 || proc == totalProcs - 1) && color == 0 && printBuffer) {
      printf("sendbuff = ");
      for (int i = 0; i < totalProcs; i++) {
        printf("%f ", ((float *)hello)[i * count]);
      }
      printf("\n");
    }

    for (int i = 0; i < totalProcs; i++) {
      if (proc % 2 == 0) {
        if (i % 2 == 0) {
          hSendcounts[i] = 2 * count;
          hRecvcounts[i] = 2 * count;
          hSdispls[i] = sdis;
          hRdispls[i] = rdis;
          if (i == proc) {
            hSendcounts[i] = 0;
            hRecvcounts[i] = 0;
          }
          sdis += 2 * count;
          rdis += 2 * count;
        } else {
          hSendcounts[i] = 0;
          hRecvcounts[i] = 0;
          hSdispls[i] = sdis;
          hRdispls[i] = rdis;
        }
      } else {
        if (i % 2 == 1) {
          hSendcounts[i] = 2 * count;
          hRecvcounts[i] = 2 * count;
          hSdispls[i] = sdis;
          hRdispls[i] = rdis;
          if (i == proc) {
            hSendcounts[i] = 0;
            hRecvcounts[i] = 0;
          }
          sdis += 2 * count;
          rdis += 2 * count;
        } else {
          hSendcounts[i] = 0;
          hRecvcounts[i] = 0;
          hSdispls[i] = sdis;
          hRdispls[i] = rdis;
        }
      }
    }

    if ((proc == 0 || proc == totalProcs - 1) && color == 0 && printBuffer) {
      printf("hSendcounts = ");
      for (int i = 0; i < totalProcs; i++) {
        printf("%ld ", hSendcounts[i]);
      }
      printf("\n");
      printf("hRecvcounts = ");
      for (int i = 0; i < totalProcs; i++) {
        printf("%ld ", hRecvcounts[i]);
      }
      printf("\n");
      printf("hSdispls = ");
      for (int i = 0; i < totalProcs; i++) {
        printf("%ld ", hSdispls[i]);
      }
      printf("\n");
      printf("hRdispls = ");
      for (int i = 0; i < totalProcs; i++) {
        printf("%ld ", hRdispls[i]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; i++) {
      flagcxAlltoAllv(sendbuff, hSendcounts, hSdispls, recvbuff, hRecvcounts,
                      hRdispls, DATATYPE, comm, stream);
    }
    devHandle->streamSynchronize(stream);

    double elapsedTime = tim.elapsed() / numIters;
    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= worldSize;

    double baseBw = (double)(size) / 1.0E9 / elapsedTime;
    double algBw = baseBw;
    double factor = ((double)(totalProcs - 1)) / ((double)(totalProcs));
    double busBw = baseBw * factor;
    if (proc == 0 && color == 0) {
      printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: %lf "
             "GB/s; Bus bandwidth: %lf GB/s\n",
             size, elapsedTime, algBw, busBw);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    memset(hello, 0, size);
    devHandle->deviceMemcpy(hello, recvbuff, size, flagcxMemcpyDeviceToHost,
                            NULL);
    if ((proc == 0 || proc == totalProcs - 1) && color == 0 && printBuffer) {
      printf("recvbuff = ");
      for (int i = 0; i < totalProcs; i++) {
        printf("%f ", ((float *)hello)[i * count]);
      }
      printf("\n");
    }
  }

  if (localRegister) {
    // deregister buffer
    flagcxCommDeregister(comm, sendHandle);
    flagcxCommDeregister(comm, recvHandle);
    // deallocate buffer
    flagcxMemFree(sendbuff);
    flagcxMemFree(recvbuff);
  } else {
    devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL);
    devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL);
  }
  free(hello);
  free(hSendcounts);
  free(hRecvcounts);
  free(hSdispls);
  free(hRdispls);
  flagcxCommDestroy(comm);
  devHandle->streamDestroy(stream);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}