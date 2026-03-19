/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * FlagCX Device API kernels.
 *
 * 1. Intra-node AllReduce — peer pointer + barrier based.
 *    Vendor (NCCL > 2.28): wraps ncclDevComm + ncclWindow_t + ncclLsaBarrier.
 *    Fallback:    IPC peer pointers + atomics barrier.
 *    Same kernel code compiles for both paths.
 *
 * 2. Inter-node AlltoAll — two separate kernels:
 *    a) One-sided (put): thread-stride loop, put + waitSignal + flush.
 *    b) Two-sided (send/recv): thread-0 block-stride loop, FIFO + term/wait.
 *    Both paths wrapped by bar.sync() pre/post barriers.
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
  // AllReduce requires peer pointer access (window or IPC)
  if (!mem._hasWindow && mem.peerPtrs == nullptr) {
    if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
      printf("flagcxIntraAllReduceKernel: no peer access (no window, no IPC), "
             "skipping\n");
    }
    return;
  }

  flagcxTeam_t intra = flagcxTeamIntra(devComm);

  // Create barrier session using simplified FlagCX API (4 params).
  flagcxIntraBarrierSession<flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, FLAGCX_BLOCK_IDX_X};

  // Pre-reduce barrier (acquire — ensure peer writes are visible)
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderAcquire);

  const int rank = devComm.getIntraRank();
  const int nRanks = devComm.getIntraSize();
  const int globalTid =
      FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_DIM_X * (rank + FLAGCX_BLOCK_IDX_X * nRanks);
  const int globalNthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X * nRanks;

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

// Host-side function — launches the kernel using caller-provided
// registered buffer and device communicator.
flagcxResult_t flagcxIntraAllReduce(flagcxDevMem_t devMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (devComm == nullptr || devMem == nullptr) {
    return flagcxInternalError;
  }

  cudaStream_t cudaStream = *(cudaStream_t *)stream;

  // Unified constructors — work for both Vendor and Fallback
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

  // Advance barrier epoch for next launch (2 syncs, each += nLocalRanks-1)
  devComm->barrierEpoch += 2 * (devComm->intraSize - 1);

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ==========================================================================
// 2a. Inter-node One-sided AlltoAll
//
// Thread-stride loop: each thread dispatches put ops to different peers.
// put() posts FIFO descriptor (Fallback) or GIN descriptor (Vendor).
// After all puts, waitSignal + flush ensure completion.
//
// Buffer layout: [rank0_data][rank1_data]...[rankN_data], each of size `count`
// sendMem: data at offset peerRank * count * elementSize is sent to peerRank
// recvMem: data from peerRank is stored at offset peerRank * count * elementSize
// ==========================================================================

FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterOneSidedAlltoAllKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxDevComm devComm) {

  // GIN one-sided: all CTAs share context 0 (matches NCCL GIN pattern).
  // Per-CTA contexts are for two-sided FIFO channels, not GIN.
  flagcxDevNet net(devComm, 0);
  // Inter-only barrier (matches NCCL GIN's ncclGinBarrierSession — no intra).
  // Uses flagcxInterBarrierSession directly to avoid ncclBarrierSession(Rail) copy issue.
  flagcxInterBarrierSession<flagcxCoopBlock> bar(
      flagcxCoopBlock(), net, flagcxTeamWorld(devComm), FLAGCX_BLOCK_IDX_X);

  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  // Pre-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

  // One-sided put/waitSignal/flush — works for both Vendor (GIN) and Fallback (FIFO proxy).
  uint64_t signalValue = net.readSignal(0);

  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    net.put(flagcxTeamWorld(devComm), peer, recvMem, myRank * size,
            sendMem, peer * size, size, flagcxDevNet_SignalInc{0},
            flagcxDevNet_None{}, flagcxCoopThread{});
  }

  net.waitSignal(flagcxCoopBlock{}, 0, signalValue + nRanks);
  net.flush(flagcxCoopBlock{});

  // Post-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
}

// ==========================================================================
// 2b. Inter-node Two-sided AlltoAll
//
// Thread-0 block-stride loop dispatches send/recv via FIFO.
// term() + wait() for group semantic completion.
//
// Buffer layout: same as one-sided.
// ==========================================================================

FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTwoSidedAlltoAllKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxDevComm devComm) {

  flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
  // Inter-only barrier: matches one-sided kernel pattern.
  // Same-node (nInterPeers == 0): no-op (FIFO term/wait handles completion).
  // Multi-node: GIN barrier (Vendor) or FIFO Signal relay (Fallback).
  flagcxInterBarrierSession<flagcxCoopBlock> bar(
      flagcxCoopBlock(), net, flagcxTeamWorld(devComm), FLAGCX_BLOCK_IDX_X);

  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  // Pre-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

  // Thread 0 dispatches all communication ops (block-stride over peers).
  if (FLAGCX_THREAD_IDX_X == 0) {
    for (int peer = FLAGCX_BLOCK_IDX_X; peer < nRanks;
         peer += FLAGCX_GRID_DIM_X) {
      size_t offset = peer * size;
      net.send(sendMem, offset, count, datatype, peer);
      net.recv(recvMem, offset, count, datatype, peer);
    }

    // Two-sided: all CTAs must enqueue term/wait, even idle ones
    // (blockIdx >= nRanks), because the proxy counts term entries from all
    // CTA_COUNT channels to trigger groupEnd. Idle CTAs' term/wait are
    // benign — no data was enqueued so wait completes instantly.
    net.term();
    net.wait();
  }

  // Post-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
}

// Host-side one-sided AlltoAll function.
flagcxResult_t flagcxInterOneSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream) {
  if (devComm == nullptr || sendMem == nullptr || recvMem == nullptr) {
    return flagcxInternalError;
  }

  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);

  flagcxInterOneSidedAlltoAllKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);

  cudaError_t err = cudaGetLastError();

  // Inter-only barrier: only advance interBarrierEpoch (no intra barriers used)
  // 2 syncs per kernel, each += nInterPeers
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// Host-side two-sided AlltoAll function.
flagcxResult_t flagcxInterTwoSidedAlltoAll(flagcxDevMem_t sendMem,
                                            flagcxDevMem_t recvMem, size_t count,
                                            flagcxDataType_t datatype,
                                            flagcxDevComm_t devComm,
                                            flagcxStream_t stream) {
  if (devComm == nullptr || sendMem == nullptr || recvMem == nullptr) {
    return flagcxInternalError;
  }

  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);

  flagcxInterTwoSidedAlltoAllKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);

  cudaError_t err = cudaGetLastError();

  // Advance inter-node barrier epoch only (no intra barrier in this kernel).
  // 2 syncs per kernel, each += nInterPeers.
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

