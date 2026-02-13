#include "comm.h"
#include "flagcx_kernel.h"
#include "global_comm.h"

// P2P kernel with multi-thread support
// All threads concurrently enqueue to test lock-free MPSC FIFO
FLAGCX_GLOBAL_DECORATOR void flagcxP2pKernel(
    const void *sendbuff, void *recvbuff, size_t totalCount,
    flagcxDataType_t datatype, int sendPeer, int recvPeer,
    void *fifoBuffer, size_t elementSize) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  // Each thread handles a chunk of the data
  size_t chunkSize = (totalCount + nthreads - 1) / nthreads;
  size_t myStart = tid * chunkSize;
  size_t myEnd = myStart + chunkSize;
  if (myEnd > totalCount) myEnd = totalCount;
  size_t myCount = (myEnd > myStart) ? (myEnd - myStart) : 0;

  if (myCount > 0) {
    size_t byteOffset = myStart * elementSize;
    const void *sendaddr = static_cast<const char *>(sendbuff) + byteOffset;
    void *recvaddr = static_cast<char *>(recvbuff) + byteOffset;

    // All threads enqueue concurrently (lock-free MPSC FIFO handles this)
    flagcxDeviceSend(sendaddr, myCount, datatype, sendPeer, fifoBuffer);
    flagcxDeviceRecv(recvaddr, myCount, datatype, recvPeer, fifoBuffer);
  }

  // Ensure all threads finish enqueuing before termination
  __syncthreads();

  // Only thread 0 sends termination and waits
  if (tid == 0) {
    flagcxDeviceTerm(fifoBuffer);
    flagcxDeviceWait(fifoBuffer);
  }
}

flagcxResult_t flagcxP2pDemo(const void *sendbuff, void *recvbuff, size_t count,
                             flagcxDataType_t datatype, int sendPeer,
                             int recvPeer, flagcxComm_t comm,
                             flagcxStream_t stream) {
  void *fifo = NULL;
  FLAGCXCHECK(flagcxCommFifoBuffer(comm, &fifo));

  size_t elementSize = 0;
  switch (datatype) {
    case flagcxChar: case flagcxUint8: elementSize = 1; break;
    case flagcxHalf: case flagcxBfloat16: elementSize = 2; break;
    case flagcxInt: case flagcxUint32: case flagcxFloat: elementSize = 4; break;
    case flagcxInt64: case flagcxUint64: case flagcxDouble: elementSize = 8; break;
    default: elementSize = 4;
  }

  // Launch kernel - change nthreads to test single (1) vs multi-thread (>1)
  int nthreads = 1;  // Start with 1 thread to verify basic functionality
  flagcxP2pKernel<<<1, nthreads, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      sendbuff, recvbuff, count, datatype, sendPeer, recvPeer, fifo, elementSize);
  return flagcxSuccess;
}
