/*************************************************************************
 * Benchmark for FlagCX inter-node one-sided operations using Device API.
 *
 * Tests one-sided put + signal + waitSignal:
 *   Rank 0 (sender): flagcxOnesidedSend (put data + signal to receiver)
 *   Rank 1 (receiver): flagcxOnesidedRecv (waitSignal for expected value)
 *
 * Requires exactly 2 MPI ranks.
 *
 * Usage: mpirun -np 2 ./test_kernel_internode_onesided [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -w <warmup>    -n <iters>     -p <printbuffer 0/1>
 *   -R <regMode>   0=raw(deviceMalloc), 1=IPC(flagcxMemAlloc+CommRegister),
 *                  2=window(flagcxMemAlloc+CommWindowRegister)
 *   One-sided ops require -R 1 or -R 2.
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"
#include <algorithm>
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
  FLAGCXCHECK(flagcxHandleInit(&handler));
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
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, uniqueId, proc));

  if (totalProcs != 2) {
    if (proc == 0)
      printf("test_kernel_internode_onesided requires exactly 2 ranks "
             "(sender=0, receiver=1).\n");
    FLAGCXCHECK(flagcxCommDestroy(comm));
    FLAGCXCHECK(flagcxHandleFree(handler));
    MPI_Finalize();
    return 0;
  }

  const int senderRank = 0;
  const int receiverRank = 1;
  bool isSender = (proc == senderRank);
  bool isReceiver = (proc == receiverRank);

  // Allocate window buffer (GPU memory in all modes)
  size_t maxIterations = std::max(numWarmupIters, numIters);
  size_t windowBytes = maxBytes * maxIterations;

  void *windowBuff = nullptr;
  // -R 0: deviceMalloc (no registration, one-sided ops won't work)
  // -R 1/-R 2: flagcxMemAlloc (GPU memory with SYNC_MEMOPS for GDR)
  if (localRegister == 0) {
    FLAGCXCHECK(devHandle->deviceMalloc(&windowBuff, windowBytes,
                                        flagcxMemDevice, NULL));
  } else {
    FLAGCXCHECK(flagcxMemAlloc(&windowBuff, windowBytes, comm));
  }
  FLAGCXCHECK(devHandle->deviceMemset(windowBuff, 0, windowBytes,
                                      flagcxMemDevice, NULL));

  // Register window buffer
  void *regHandle = nullptr;
  flagcxWindow_t win = nullptr;
  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowRegister(comm, windowBuff, windowBytes, &win,
                                         FLAGCX_WIN_DEFAULT));
  } else if (localRegister == 1) {
    FLAGCXCHECK(flagcxCommRegister(comm, windowBuff, windowBytes, &regHandle));
  }

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  // Host buffer for data preparation and verification
  void *hostBuff = malloc(maxBytes);
  memset(hostBuff, 0, maxBytes);

  // Create device communicator with 1 signal for one-sided wait
  flagcxDevComm_t devComm = nullptr;
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interSignalCount = 1;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Signal semantics: sender does RDMA ATOMIC FETCH_AND_ADD +1 to
  // signalBuffer[0] on each send. Receiver waits for signalBuffer[0] >=
  // expected.
  uint64_t signalExpected = 0;

  if (proc == 0 && color == 0) {
    printf("# FlagCX Device API Inter-node One-sided Benchmark\n");
    printf("# nRanks: %d, regMode: %s\n", totalProcs,
           localRegister == 2   ? "window"
           : localRegister == 1 ? "ipc"
                                : "raw (no registration)");
    printf("# %-12s %-14s %-14s\n", "Size(B)", "Time(s)", "BW(GB/s)");
  }

  // Warm-up iterations
  for (int i = 0; i < numWarmupIters; ++i) {
    size_t sendOff = i * maxBytes;
    size_t recvOff = i * maxBytes;

    if (isSender) {
      uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
      std::memset(hostBuff, value, maxBytes);

      FLAGCXCHECK(devHandle->deviceMemcpy((char *)windowBuff + sendOff,
                                          hostBuff, maxBytes,
                                          flagcxMemcpyHostToDevice, NULL));

      flagcxOnesidedSend(sendOff, recvOff, 0, maxBytes / sizeof(float),
                         DATATYPE, receiverRank, devComm, stream);
    } else if (isReceiver) {
      signalExpected++;
      flagcxOnesidedRecv(0, signalExpected, devComm, stream);
    }
  }
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  // Benchmark loop
  timer tim;
  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    if (size == 0)
      break;

    size_t count = size / sizeof(float);

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; ++i) {
      size_t sendOff = i * size;
      size_t recvOff = i * size;

      if (isSender) {
        // Prepare test pattern in host buffer
        memset(hostBuff, 0, size);
        strcpy((char *)hostBuff, "_0x1234");
        if (size > 32)
          strcpy((char *)hostBuff + size / 3, "_0x5678");
        if (size > 64)
          strcpy((char *)hostBuff + size / 3 * 2, "_0x9abc");

        FLAGCXCHECK(devHandle->deviceMemcpy((char *)windowBuff + sendOff,
                                            hostBuff, size,
                                            flagcxMemcpyHostToDevice, NULL));

        flagcxOnesidedSend(sendOff, recvOff, 0, count, DATATYPE, receiverRank,
                           devComm, stream);
      } else if (isReceiver) {
        signalExpected++;
        flagcxOnesidedRecv(0, signalExpected, devComm, stream);
      }
    }
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    double elapsedTime = tim.elapsed() / numIters;
    MPI_Allreduce(MPI_IN_PLACE, &elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= worldSize;

    double bandwidth = (double)size / 1.0e9 / elapsedTime;
    if (proc == 0 && color == 0) {
      printf("  %-12zu %-14lf %-14lf\n", size, elapsedTime, bandwidth);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (printBuffer) {
      if (isSender && proc == 0 && color == 0) {
        printf("sendbuff = %s\n", (const char *)hostBuff);
      }
      if (isReceiver && numIters > 0) {
        // Read from last iteration's recv offset
        size_t lastRecvOff = (numIters - 1) * size;
        memset(hostBuff, 0, size);
        FLAGCXCHECK(
            devHandle->deviceMemcpy(hostBuff, (char *)windowBuff + lastRecvOff,
                                    size, flagcxMemcpyDeviceToHost, NULL));
        if (color == 0) {
          printf("recvbuff = %s\n", (const char *)hostBuff);
        }
      }
    }
  }

  // Cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);

  if (devComm != nullptr) {
    FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  }

  free(hostBuff);

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, win));
  } else if (localRegister == 1) {
    FLAGCXCHECK(flagcxCommDeregister(comm, regHandle));
  }
  if (localRegister == 0) {
    FLAGCXCHECK(devHandle->deviceFree(windowBuff, flagcxMemDevice, NULL));
  } else {
    FLAGCXCHECK(flagcxMemFree(windowBuff, comm));
  }

  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return 0;
}
