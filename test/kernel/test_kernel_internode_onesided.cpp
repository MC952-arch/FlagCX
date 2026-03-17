/*************************************************************************
 * Benchmark for FlagCX inter-node one-sided operations using Device API.
 *
 * Tests one-sided put + signal + waitSignal:
 *   Rank 0 (sender): flagcxOnesidedSend (put data + signal to receiver)
 *   Rank 1 (receiver): flagcxOnesidedRecv (waitSignal for expected value)
 *
 * Requires exactly 2 MPI ranks and FLAGCX_ENABLE_ONE_SIDE_REGISTER=1.
 *
 * Usage: mpirun -np 2 ./test_kernel_internode_onesided [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -w <warmup>    -n <iters>     -p <printbuffer 0/1>
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
  size_t min_bytes = args.getMinBytes();
  size_t max_bytes = args.getMaxBytes();
  int step_factor = args.getStepFactor();
  int num_warmup_iters = args.getWarmupIters();
  int num_iters = args.getTestIters();
  int print_buffer = args.isPrintBuffer();
  uint64_t split_mask = args.getSplitMask();

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
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  // Enable one-sided register (must be set before communicator initialization)
  setenv("FLAGCX_ENABLE_ONE_SIDE_REGISTER", "1", 1);

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

  // Allocate and register window buffer for one-sided data operations
  size_t max_iterations = std::max(num_warmup_iters, num_iters);
  size_t window_bytes = max_bytes * max_iterations;

  void *window = nullptr;
  if (posix_memalign(&window, 64, window_bytes) != 0 || window == nullptr) {
    fprintf(stderr,
            "[rank %d] posix_memalign failed for host window (size=%zu)\n",
            proc, window_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::memset(window, 0, window_bytes);

  // Register window buffer — sets up one-sided handles
  void *windowHandle = nullptr;
  FLAGCXCHECK(flagcxCommRegister(comm, window, window_bytes, &windowHandle));

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  // Allocate device buffers
  void *srcbuff = nullptr;
  FLAGCXCHECK(
      devHandle->deviceMalloc(&srcbuff, max_bytes, flagcxMemDevice, NULL));

  void *hello = malloc(max_bytes);
  memset(hello, 0, max_bytes);

  // Create device communicator with 1 signal for one-sided wait
  flagcxDevComm_t devComm = nullptr;
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interSignalCount = 1;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Signal semantics: sender does RDMA ATOMIC FETCH_AND_ADD +1 to
  // signalBuffer[0] on each send. Receiver waits for signalBuffer[0] >=
  // expected.
  uint64_t signalExpected = 0;

  // Warm-up iterations
  for (int i = 0; i < num_warmup_iters; ++i) {
    size_t current_send_offset = i * max_bytes;
    size_t current_recv_offset = i * max_bytes;

    if (isSender) {
      uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
      std::memset((char *)window + current_send_offset, value, max_bytes);

      FLAGCXCHECK(
          devHandle->deviceMemcpy(srcbuff, (char *)window + current_send_offset,
                                  max_bytes, flagcxMemcpyHostToDevice, NULL));

      flagcxOnesidedSend(0, current_recv_offset, 0, max_bytes / sizeof(float),
                         DATATYPE, receiverRank, devComm, stream);
    } else if (isReceiver) {
      signalExpected++;
      flagcxOnesidedRecv(0, signalExpected, devComm, stream);
    }
  }
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  if (proc == 0 && color == 0) {
    printf("# FlagCX Device API Inter-node One-sided Benchmark\n");
    printf("# nRanks: %d\n", totalProcs);
    printf("# %-12s %-14s %-14s\n", "Size(B)", "Time(s)", "BW(GB/s)");
  }

  // Benchmark loop
  timer tim;
  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    if (size == 0)
      break;

    size_t count = size / sizeof(float);

    if (isSender) {
      strcpy((char *)hello, "_0x1234");
      if (size > 32)
        strcpy((char *)hello + size / 3, "_0x5678");
      if (size > 64)
        strcpy((char *)hello + size / 3 * 2, "_0x9abc");

      if (proc == 0 && color == 0 && print_buffer) {
        printf("sendbuff = %s\n", (const char *)hello);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < num_iters; ++i) {
      size_t current_send_offset = i * size;
      size_t current_recv_offset = i * size;

      if (isSender) {
        uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
        std::memset((char *)window + current_send_offset, value, size);
        memcpy(hello, (char *)window + current_send_offset, size);

        FLAGCXCHECK(devHandle->deviceMemcpy(srcbuff, hello, size,
                                            flagcxMemcpyHostToDevice, NULL));

        flagcxOnesidedSend(0, current_recv_offset, 0, count, DATATYPE,
                           receiverRank, devComm, stream);
      } else if (isReceiver) {
        signalExpected++;
        flagcxOnesidedRecv(0, signalExpected, devComm, stream);
      }
    }
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    double elapsed_time = tim.elapsed() / num_iters;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsed_time /= worldSize;

    double bandwidth = (double)size / 1.0e9 / elapsed_time;
    if (proc == 0 && color == 0) {
      printf("  %-12zu %-14lf %-14lf\n", size, elapsed_time, bandwidth);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (isReceiver && num_iters > 0 && print_buffer) {
      memset(hello, 0, size);
      memcpy(hello, (char *)window + 0, size);
      if (color == 0) {
        printf("recvbuff = %s\n", (const char *)hello);
      }
    }
  }

  // Cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);

  if (devComm != nullptr) {
    FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  }

  FLAGCXCHECK(devHandle->deviceFree(srcbuff, flagcxMemDevice, NULL));
  free(hello);

  if (windowHandle != nullptr) {
    FLAGCXCHECK(flagcxCommDeregister(comm, windowHandle));
  }
  free(window);

  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return 0;
}
