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
  size_t count, recvcount, recvsize;
  timer tim;

  if (localRegister) {
    // allocate buffer
    flagcxMemAlloc(&sendbuff, maxBytes);
    flagcxMemAlloc(&recvbuff, maxBytes / totalProcs);
    // register buffer
    flagcxCommRegister(comm, sendbuff, maxBytes, &sendHandle);
    flagcxCommRegister(comm, recvbuff, maxBytes / totalProcs, &recvHandle);
  } else {
    devHandle->deviceMalloc(&sendbuff, maxBytes, flagcxMemDevice, NULL);
    devHandle->deviceMalloc(&recvbuff, maxBytes / totalProcs, flagcxMemDevice,
                            NULL);
  }
  hello = malloc(maxBytes);
  memset(hello, 0, maxBytes);

  // Warm-up for large size
  for (int i = 0; i < numWarmupIters; i++) {
    flagcxReduceScatter(sendbuff, recvbuff,
                        (maxBytes / sizeof(float)) / totalProcs, DATATYPE,
                        flagcxSum, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < numWarmupIters; i++) {
    flagcxReduceScatter(sendbuff, recvbuff,
                        (minBytes / sizeof(float)) / totalProcs, DATATYPE,
                        flagcxSum, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    count = size / sizeof(float);
    recvcount = count / totalProcs;
    recvsize = size / totalProcs;

    size_t index = 0;
    float value = 0.0;
    for (size_t i = 0; i < count; i++) {
      ((float *)hello)[i] = value;
      if (index == recvcount - 1) {
        index = 0;
        value += 1.0;
      } else {
        index++;
      }
    }

    devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                            stream);
    devHandle->streamSynchronize(stream);

    if (color == 0 && printBuffer) {
      printf("proc %d sendbuff = ", proc);
      for (size_t i = proc * recvcount; i < proc * recvcount + 10; i++) {
        printf("%f ", ((float *)hello)[i]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; i++) {
      flagcxReduceScatter(sendbuff, recvbuff, recvcount, DATATYPE, flagcxSum,
                          comm, stream);
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
    devHandle->deviceMemcpy(hello, recvbuff, recvsize, flagcxMemcpyDeviceToHost,
                            stream);
    devHandle->streamSynchronize(stream);

    if (color == 0 && printBuffer) {
      printf("proc %d recvbuff = ", proc);
      int correct = 1;
      for (size_t i = 0; i < 10; i++) {
        printf("%f ", ((float *)hello)[i]);
      }
      printf("\n");
      for (size_t i = 0; i < recvcount; i++) {
        if (((float *)hello)[i] != (float)(proc)*totalProcs) {
          correct = 0;
          printf("rank %d offset %lu wrong output %f\n", proc, i,
                 ((float *)hello)[i]);
          break;
        }
      }
      printf("rank %d correctness = %d\n", proc, correct);
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
  flagcxCommDestroy(comm);
  devHandle->streamDestroy(stream);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}
