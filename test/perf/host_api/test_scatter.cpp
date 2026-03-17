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
  int root = args.getRootRank();
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
  root = root % totalProcs;

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
    flagcxScatter(sendbuff, recvbuff, (maxBytes / sizeof(float)) / totalProcs,
                  DATATYPE, 0, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < numWarmupIters; i++) {
    flagcxScatter(sendbuff, recvbuff, (minBytes / sizeof(float)) / totalProcs,
                  DATATYPE, 0, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    int beginRoot, endRoot;
    double sumAlgBw = 0;
    double sumBusBw = 0;
    double sumTime = 0;
    int testCount = 0;

    if (root != -1) {
      beginRoot = endRoot = root;
    } else {
      beginRoot = 0;
      endRoot = totalProcs - 1;
    }
    for (int r = beginRoot; r <= endRoot; r++) {
      count = size / sizeof(float);

      for (int v = 0; v < totalProcs; v++) {
        for (size_t i = 0; i < count / totalProcs; i++) {
          ((float *)hello)[v * (count / totalProcs) + i] = v;
        }
      }

      if (proc == r) {
        devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                                NULL);
      }

      if (proc == r && color == 0 && printBuffer) {
        printf("root rank is %d\n", r);
        printf("sendbuff = ");
        for (size_t i = 0; i < 10; i++) {
          printf("%f ", ((float *)hello)[i]);
        }
        printf("\n");
      }

      MPI_Barrier(MPI_COMM_WORLD);

      tim.reset();
      for (int i = 0; i < numIters; i++) {
        flagcxScatter(sendbuff, recvbuff, count / totalProcs, DATATYPE, r, comm,
                      stream);
      }
      devHandle->streamSynchronize(stream);

      MPI_Barrier(MPI_COMM_WORLD);

      double elapsedTime = tim.elapsed() / numIters;
      MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      elapsedTime /= worldSize;

      double baseBw = (double)(size) / 1.0E9 / elapsedTime;
      double algBw = baseBw;
      double factor = ((double)(totalProcs - 1)) / ((double)(totalProcs));
      double busBw = baseBw * factor;
      sumAlgBw += algBw;
      sumBusBw += busBw;
      sumTime += elapsedTime;
      testCount++;

      memset(hello, 0, size);
      devHandle->deviceMemcpy(hello, recvbuff, size / totalProcs,
                              flagcxMemcpyDeviceToHost, NULL);
      if (color == 0 && printBuffer) {
        printf("rank %d recvbuff = %f\n", proc, ((float *)hello)[0]);
      }
    }

    if (proc == 0 && color == 0) {
      double algBw = sumAlgBw / testCount;
      double busBw = sumBusBw / testCount;
      double elapsedTime = sumTime / testCount;
      printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: %lf "
             "GB/s; Bus bandwidth: %lf GB/s\n",
             size, elapsedTime, algBw, busBw);
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