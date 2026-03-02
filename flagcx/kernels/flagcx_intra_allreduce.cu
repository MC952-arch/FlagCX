/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * FlagCX Intra-node AllReduce kernel using FlagCX Device API.
 * Functionally equivalent to the NCCL reference inPlaceAllReduceKernel,
 * but uses exclusively FlagCX abstractions (zero direct NCCL references).
 *
 * Host-side flagcxDevCommCreate/Destroy are in flagcx_device.cc.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "nvidia_adaptor.h"
#include "global_comm.h"
#include "flagcx_kernel.h"
#include <cuda_runtime.h>

#ifdef FLAGCX_DEVICE_API_NCCL

// Intra-node AllReduce: each block reads from all peers via team-based
// flagcxGetPeerPointer, reduces (sum), and writes result back to all peers.
template <typename T>
__global__ void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraAllReduceKernel(flagcxDeviceComm devComm, flagcxDeviceWindow win,
                               size_t offset, size_t count) {
  flagcxTeam_t intra = flagcxTeamIntra(devComm);

  // Create barrier session using simplified FlagCX API (4 params).
  // Internally maps to NCCL's 6-param ncclLsaBarrierSession constructor:
  //   {ncclCoopCta(), devComm._base, ncclTeamLsa(devComm._base),
  //    devComm._base.lsaBarrier, blockIdx.x, false}
  flagcxIntraBarrierSession<flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, blockIdx.x};

  // Pre-reduce barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

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
      T* inputPtr = (T*)flagcxGetPeerPointer(win, offset, intra, peer);
      v += inputPtr[o];
    }
    for (int peer = 0; peer < nRanks; peer++) {
      T* outputPtr = (T*)flagcxGetPeerPointer(win, offset, intra, peer);
      outputPtr[o] = v;
    }
  }

  // Post-reduce barrier (release ordering — ensure writes are visible)
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelease);
}

// Host-side launcher
template <typename T>
static cudaError_t launchFlagcxIntraAllReduce(flagcxDeviceComm devComm,
                                              flagcxDeviceWindow win,
                                              size_t offset, size_t count,
                                              cudaStream_t stream) {
  flagcxIntraAllReduceKernel<T>
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         stream>>>(devComm, win, offset, count);
  return cudaGetLastError();
}

// Explicit instantiations for common types
template cudaError_t launchFlagcxIntraAllReduce<float>(flagcxDeviceComm,
                                                       flagcxDeviceWindow, size_t,
                                                       size_t, cudaStream_t);
template cudaError_t launchFlagcxIntraAllReduce<double>(flagcxDeviceComm,
                                                        flagcxDeviceWindow, size_t,
                                                        size_t, cudaStream_t);

// ==========================================================================
// Host-side demo function — launches the kernel using caller-provided
// window-registered buffer and device communicator.
// The caller is responsible for:
//   1. flagcxDevCommCreate to create devComm
//   2. flagcxMemAlloc / flagcxCommWindowRegister to create windowBuff + win
//   3. Copying sendbuff into windowBuff before calling this function
//   4. Copying the result out of windowBuff after this function returns
//   5. flagcxCommWindowDeregister / flagcxMemFree for cleanup
//   6. flagcxDevCommDestroy to destroy devComm
// ==========================================================================
flagcxResult_t flagcxIntraAllReduceDemo(void *windowBuff,
                                        flagcxWindow_t win, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (devComm == nullptr) {
    return flagcxInternalError;
  }

  cudaStream_t cudaStream = *(cudaStream_t *)stream;

  // Construct FlagCX device API types from the opaque handle
  flagcxDeviceComm devCommKernel(devComm->ncclDev);
  flagcxDeviceWindow devWin(win->base);

  cudaError_t err;
  switch (datatype) {
  case flagcxFloat32:
    err =
        launchFlagcxIntraAllReduce<float>(devCommKernel, devWin, 0, count, cudaStream);
    break;
  case flagcxFloat64:
    err = launchFlagcxIntraAllReduce<double>(devCommKernel, devWin, 0, count,
                                             cudaStream);
    break;
  default:
    return flagcxInvalidArgument;
  }
  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

#else // !FLAGCX_DEVICE_API_NCCL

flagcxResult_t flagcxIntraAllReduceDemo(void *windowBuff,
                                        flagcxWindow_t win, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  return flagcxNotSupported;
}

#endif // FLAGCX_DEVICE_API_NCCL
