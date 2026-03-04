/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * FlagCX Device API demo kernels.
 *
 * 1. Intra-node AllReduce — peer pointer + barrier based.
 *    Tier 1 (NCCL > 2.28): wraps ncclDevComm + ncclWindow_t + ncclLsaBarrier.
 *    Tier 2 (fallback):    IPC peer pointers + atomics barrier.
 *    Same kernel code compiles for both tiers.
 *
 * 2. Inter-node P2P — FIFO-based device Send/Recv.
 *    Uses flagcxDeviceSend/Recv/Term/Wait via devComm FIFO.
 *
 * Host-side flagcxDevCommCreate/Destroy are in flagcx_device.cc.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "nvidia_adaptor.h"
#include "global_comm.h"
#include "flagcx_kernel.h"
#include <cuda_runtime.h>

// ==========================================================================
// 1. Intra-node AllReduce
// ==========================================================================

// Intra-node AllReduce: each block reads from all peers via team-based
// flagcxGetPeerPointer, reduces (sum), and writes result back to all peers.
template <typename T>
__global__ void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraAllReduceKernel(flagcxDevComm devComm, flagcxDevMem mem,
                               size_t offset, size_t count) {
  flagcxTeam_t intra = flagcxTeamIntra(devComm);

  // Create barrier session using simplified FlagCX API (4 params).
  flagcxIntraBarrierSession<flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, blockIdx.x};

  // Pre-reduce barrier (acquire — ensure peer writes are visible)
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderAcquire);

  const int rank = devComm.getIntraRank();
  const int nRanks = devComm.getIntraSize();
  const int globalTid =
      threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  // Phase 1: Reduce — sum data from all intra-node peers
  // Phase 2: Write — store result to all intra-node peers
  for (size_t o = globalTid; o < count; o += globalNthreads) {
    T v = T(0);
    for (int peer = 0; peer < nRanks; peer++) {
      T* inputPtr = (T*)flagcxGetPeerPointer(mem, offset, intra, peer);
      v += inputPtr[o];
    }
    for (int peer = 0; peer < nRanks; peer++) {
      T* outputPtr = (T*)flagcxGetPeerPointer(mem, offset, intra, peer);
      outputPtr[o] = v;
    }
  }

  // Post-reduce barrier (release ordering — ensure writes are visible)
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelease);
}

// Host-side launcher
template <typename T>
static cudaError_t launchFlagcxIntraAllReduce(flagcxDevComm devComm,
                                              flagcxDevMem mem,
                                              size_t offset, size_t count,
                                              cudaStream_t stream) {
  flagcxIntraAllReduceKernel<T>
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         stream>>>(devComm, mem, offset, count);
  return cudaGetLastError();
}

// Explicit instantiations for common types
template cudaError_t launchFlagcxIntraAllReduce<float>(flagcxDevComm,
                                                       flagcxDevMem, size_t,
                                                       size_t, cudaStream_t);
template cudaError_t launchFlagcxIntraAllReduce<double>(flagcxDevComm,
                                                        flagcxDevMem, size_t,
                                                        size_t, cudaStream_t);

// Host-side demo function — launches the kernel using caller-provided
// registered buffer and device communicator.
flagcxResult_t flagcxIntraAllReduceDemo(void *buff, flagcxDevMem_t devMem,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (devComm == nullptr || devMem == nullptr) {
    return flagcxInternalError;
  }

  cudaStream_t cudaStream = *(cudaStream_t *)stream;

  // Unified constructors — work for both Tier 1 and Tier 2
  flagcxDevComm devCommKernel(*devComm);
  flagcxDevMem devMemKernel(*devMem);

  cudaError_t err;
  switch (datatype) {
  case flagcxFloat32:
    err = launchFlagcxIntraAllReduce<float>(devCommKernel, devMemKernel, 0,
                                            count, cudaStream);
    break;
  case flagcxFloat64:
    err = launchFlagcxIntraAllReduce<double>(devCommKernel, devMemKernel, 0,
                                             count, cudaStream);
    break;
  default:
    return flagcxInvalidArgument;
  }

  // Advance barrier epoch for next launch (2 syncs per kernel invocation)
  devComm->barrierEpoch += 2;

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ==========================================================================
// 2. Inter-node P2P
// ==========================================================================

#define NBLOCKS 1
#define NTHREADS_PER_BLOCK 32

// P2P kernel implementing alltoall pattern (one thread per peer)
// Each thread handles all communication with its assigned peer
// This preserves send/recv ordering per-peer for correct P2P matching
// Note: Uses single block so __syncthreads() can synchronize all threads
// Buffer layout: [rank0_data][rank1_data]...[rankN_data], each of size count
// sendMem: data at offset peerRank * count is sent to peerRank
// recvMem: data from peerRank is stored at offset peerRank * count
FLAGCX_GLOBAL_DECORATOR void flagcxInterP2pKernel(
    flagcxDevMem sendMem, flagcxDevMem recvMem, size_t count,
    flagcxDataType_t datatype, flagcxDevComm devComm) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t elementSize = getFlagcxDataTypeSizeDevice(datatype);
  int nRanks = devComm.getSize();

  // Each thread handles one peer (tid = peer index)
  if (tid < nRanks) {
    int peerRank = tid;

    // Calculate offsets for this peer's send and receive buffers
    size_t offset = peerRank * count * elementSize;
    const void *peerSendBuff = flagcxGetLocalPointer(sendMem, offset);
    void *peerRecvBuff = flagcxGetLocalPointer(recvMem, offset);

    // Trigger P2P operations
    flagcxDeviceSend(peerSendBuff, count, datatype, peerRank, devComm);
    flagcxDeviceRecv(peerRecvBuff, count, datatype, peerRank, devComm);
  }

  // Ensure all threads finish enqueuing before termination
  FLAGCX_DEVICE_SYNC_THREADS();

  // Only thread 0 sends termination and waits
  if (tid == 0) {
    flagcxDeviceTerm(devComm);
    flagcxDeviceWait(devComm);
  }
}

// Alltoall demo: each rank sends different data to each peer and receives from
// all. sendMem/recvMem: size = nRanks * count elements (data for/from peer i
// at offset i * count)
flagcxResult_t flagcxInterP2pDemo(flagcxDevMem_t sendMem, flagcxDevMem_t recvMem,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxDevComm_t devComm,
                                  flagcxStream_t stream) {
  if (devComm == nullptr || sendMem == nullptr || recvMem == nullptr) {
    return flagcxInternalError;
  }

  // Unified constructors — work for both Tier 1 and Tier 2
  flagcxDevComm devCommKernel(*devComm);
  flagcxDevMem sendMemKernel(*sendMem);
  flagcxDevMem recvMemKernel(*recvMem);

  // Launch kernel with (NBLOCKS, NTHREADS_PER_BLOCK) (one thread per potential
  // peer) Single block ensures __syncthreads() synchronizes all threads before
  // Term/Wait Each thread handles communication with one peer, preserving
  // ordering
  flagcxInterP2pKernel<<<NBLOCKS, NTHREADS_PER_BLOCK, 0,
                         *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      sendMemKernel, recvMemKernel, count, datatype, devCommKernel);

  return flagcxSuccess;
}

// ==========================================================================
// 3. GIN AlltoAll (Tier 1 only — requires FLAGCX_DEVICE_API_NCCL)
// ==========================================================================

#ifdef FLAGCX_DEVICE_API_NCCL
template <typename T>
FLAGCX_GLOBAL_DECORATOR void flagcxGinAlltoAllKernel(
    flagcxDevMem sendMem, flagcxDevMem recvMem, size_t count,
    flagcxDevComm devComm) {
  flagcxDevNet net(devComm, 0);
  uint64_t signalValue = net.readSignal(0);

  flagcxBarrierSession<flagcxCoopBlock> bar(flagcxCoopBlock(),
                                            flagcxTeamTagWorld{}, net,
                                            blockIdx.x);
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  int myRank = devComm.getRank();
  int nRanks = devComm.getSize();
  size_t size = count * sizeof(T);

  for (int r = tid; r < nRanks; r += nthreads) {
    net.put(flagcxTeamWorld(devComm), r, recvMem, myRank * size, sendMem,
            r * size, size, flagcxDevNet_SignalInc{0});
  }

  net.waitSignal(flagcxCoopBlock(), 0, signalValue + nRanks);
  net.flush(flagcxCoopBlock());
}
#endif // FLAGCX_DEVICE_API_NCCL

flagcxResult_t flagcxGinAlltoAllDemo(flagcxDevMem_t sendMem,
                                     flagcxDevMem_t recvMem, size_t count,
                                     flagcxDataType_t datatype,
                                     flagcxDevComm_t devComm,
                                     flagcxStream_t stream) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm == nullptr || sendMem == nullptr || recvMem == nullptr) {
    return flagcxInternalError;
  }

  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);
  flagcxGinAlltoAllKernel<float>
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, dc);
  return flagcxSuccess;
#else
  return flagcxInternalError; // GIN not available on Tier 2
#endif
}
