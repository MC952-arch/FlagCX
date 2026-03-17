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
  size_t count;
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

  const char *envLocRed = getenv("FLAGCX_UNIRUNNER_USE_LOCRED");
  const char *envRingAG = getenv("FLAGCX_UNIRUNNER_USE_RINGAG");

  // Warm-up for large size
  for (int i = 0; i < numWarmupIters; i++) {
    flagcxAllReduce(sendbuff, recvbuff, maxBytes / sizeof(float), DATATYPE,
                    flagcxSum, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < numWarmupIters; i++) {
    flagcxAllReduce(sendbuff, recvbuff, minBytes / sizeof(float), DATATYPE,
                    flagcxSum, comm, stream);
  }
  devHandle->streamSynchronize(stream);
  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    count = size / sizeof(float);

    for (size_t i = 0; i < count; i++) {
      ((float *)hello)[i] = i % 10 * (1 << proc);
    }

    devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                            NULL);
    devHandle->deviceMemcpy(recvbuff, hello, size, flagcxMemcpyHostToDevice,
                            NULL);

    if (color == 0 && printBuffer) {
      printf("rank %d sendbuff = ", proc);
      for (size_t i = 0; i < 10; i++) {
        printf("%f ", ((float *)hello)[i]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; i++) {
      flagcxAllReduce(sendbuff, recvbuff, count, DATATYPE, flagcxSum, comm,
                      stream);
    }
    devHandle->streamSynchronize(stream);

    double elapsedTime = tim.elapsed() / numIters;
    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= worldSize;

    double baseBw = (double)(size) / 1.0E9 / elapsedTime;
    double algBw = baseBw;
    double factor = ((double)(2 * (totalProcs - 1))) / ((double)(totalProcs));
    if (envLocRed != NULL && atoi(envLocRed) == 1) {
      factor = 1;
    } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
      factor = ((double)(totalProcs - 1)) / ((double)(totalProcs));
    }
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
    if (color == 0 && printBuffer) {
      printf("rank %d recvbuff = ", proc);
      for (size_t i = 0; i < 10; i++) {
        printf("%f ", ((float *)hello)[i]);
      }
      printf("\n");

      if (envLocRed != NULL && atoi(envLocRed) == 1) {
        /* red correctness check */
        int redCorrect = 1;
        for (size_t i = 0; i < count; i++) {
          if (i * totalProcs / count == (size_t)proc)
            continue;
          if (((float *)hello)[i] != (float)(i % 10 * (1 << proc))) {
            printf("rank %d wrong output at offset %lu, expected %f, got %f\n ",
                   proc, i, (float)(i % 10 * (1 << proc)), ((float *)hello)[i]);
            redCorrect = 0;
            break;
          }
        }
        for (size_t i = proc * count / totalProcs;
             i < (proc + 1) * count / totalProcs; i++) {
          if (((float *)hello)[i] != (float)(i % 10 * (1 << (proc + 1)))) {
            printf("rank %d wrong output at offset %lu, expected %f, got %f\n",
                   proc, i, (float)(i % 10 * (1 << (proc + 1))),
                   ((float *)hello)[i]);
            redCorrect = 0;
            break;
          }
        }
        printf("rank %d reduce correctness = %d\n", proc, redCorrect);
      } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
        /* p2p correctness check */
        int p2pCorrect = 1;
        for (size_t i = 0; i < count; i++) {
          if (((float *)hello)[i] !=
              (float)(i % 10 * (1 << (i * totalProcs / count)))) {
            printf("rank %d wrong output at offset %lu, expected %f, got %f\n",
                   proc, i, (float)(i % 10 * (1 << (i * totalProcs / count))),
                   ((float *)hello)[i]);
            p2pCorrect = 0;
            break;
          }
        }
        printf("rank %d p2p correctness = %d\n", proc, p2pCorrect);
      } else {
        /* all-reduce correctness check */
        int arCorrect = 1;
        for (size_t i = 0; i < count; i++) {
          if ((i % 10 == 0 && ((float *)hello)[i] != 0) ||
              ((float *)hello)[i] / (float)(i % 10 * ((1 << totalProcs) - 1)) >
                  1 + 1e-5 ||
              ((float *)hello)[i] / (float)(i % 10 * ((1 << totalProcs) - 1)) <
                  1 - 1e-5) {
            printf("rank %d wrong output at offset %lu, expected %f, got %f\n",
                   proc, i, (float)(i % 10 * ((1 << totalProcs) - 1)),
                   ((float *)hello)[i]);
            arCorrect = 0;
            break;
          }
        }
        printf("rank %d all-reduce correctness = %d\n", proc, arCorrect);
      }
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
  devHandle->streamDestroy(stream);
  flagcxCommDestroy(comm);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}
