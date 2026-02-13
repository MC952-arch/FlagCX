#include "comm.h"
#include "flagcx_kernel.h"
#include "global_comm.h"

#define NBLOCKS 1
#define NTHREADS_PER_BLOCK 32

// P2P kernel with multi-thread support (one thread per peer)
// Each thread handles all communication with its assigned peer
// This preserves send/recv ordering per-peer for correct P2P matching
// Note: Uses single block so __syncthreads() can synchronize all threads
FLAGCX_GLOBAL_DECORATOR void flagcxP2pKernel(
    const void *sendbuff, void *recvbuff, size_t count,
    flagcxDataType_t datatype, int myRank, int nRanks, void *fifoBuffer) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Each thread handles one peer (tid = peer index)
  // Skip if tid >= nRanks or tid == myRank (no self-communication)
  if (tid < nRanks && tid != myRank) {
    int peerRank = tid;

    // Send to peer and receive from peer
    // Each thread's operations are ordered: send then recv
    flagcxDeviceSend(sendbuff, count, datatype, peerRank, fifoBuffer);
    flagcxDeviceRecv(recvbuff, count, datatype, peerRank, fifoBuffer);
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
                             flagcxDataType_t datatype, flagcxComm_t comm,
                             flagcxStream_t stream) {
  void *fifo = NULL;
  FLAGCXCHECK(flagcxCommFifoBuffer(comm, &fifo));

  int myRank, nRanks;
  FLAGCXCHECK(flagcxCommUserRank(comm, &myRank));
  FLAGCXCHECK(flagcxCommCount(comm, &nRanks));

  // Launch kernel with 1 block, nRanks threads (one thread per potential peer)
  // Single block ensures __syncthreads() synchronizes all threads before Term/Wait
  // Each thread handles communication with one peer, preserving ordering
  flagcxP2pKernel<<<NBLOCKS, NTHREADS_PER_BLOCK, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      sendbuff, recvbuff, count, datatype, myRank, nRanks, fifo);
  return flagcxSuccess;
}
