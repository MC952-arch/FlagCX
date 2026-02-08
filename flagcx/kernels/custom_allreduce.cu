/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "nvidia_adaptor.h"
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

// Helper structures and functions
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T, int N = 16>
struct packed_t {
  using P = array_t<T, N / sizeof(T)>;
  using A = array_t<float, N / sizeof(float)>;
};

// Type aliases for compatibility
typedef __half half;
typedef __nv_bfloat16 nv_bfloat16;

template<typename T>
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t pack32(T lo, T hi) {
  uint32_t v;
  uint16_t lo16 = *reinterpret_cast<uint16_t*>(&lo);
  uint16_t hi16 = *reinterpret_cast<uint16_t*>(&hi);
  v = uint32_t(lo16) | (uint32_t(hi16) << 16);
  return v;
}

template<typename T>
FLAGCX_DEVICE_INLINE_DECORATOR void unpack32(uint32_t v, T& lo, T& hi) {
  uint16_t lo16 = v & 0xffff;
  uint16_t hi16 = v >> 16;
  lo = *reinterpret_cast<T*>(&lo16);
  hi = *reinterpret_cast<T*>(&hi16);
}

// Multimem operations
template <typename T, int N>
FLAGCX_DEVICE_INLINE_DECORATOR array_t<T, N> multimem_sum(T *addr) {
    array_t<T, N> ret;
    if constexpr (std::is_same<T, nv_bfloat16>::value) {
      uint32_t h;
      asm volatile (
          "multimem.ld_reduce.global.add.bf16x2 %0, [%1];"
          : "=r"(h)
          : "l"(addr)
          : "memory"
      );
      unpack32<nv_bfloat16>(h, ret.data[0], ret.data[1]);
    } else if constexpr (std::is_same<T, half>::value){
      uint32_t h;
      asm volatile (
          "multimem.ld_reduce.global.add.f16x2 %0, [%1];"
          : "=r"(h)
          : "l"(addr)
          : "memory"
      );
      unpack32<half>(h, ret.data[0], ret.data[1]);
    } else if constexpr (std::is_same<T, float>::value){
      asm volatile (
          "multimem.ld_reduce.global.add.f32 %0, [%1];"
          : "=f"(ret.data[0])
          : "l"(addr)
          : "memory"
      );
    }
    return ret;
}

template <typename T, int N>
FLAGCX_DEVICE_INLINE_DECORATOR void multimem_st(T *addr, array_t<T,N> val) {
    if constexpr (std::is_same<T, nv_bfloat16>::value) {
      uint32_t h = pack32<nv_bfloat16>(val.data[0], val.data[1]);
      asm volatile (
          "multimem.st.global.bf16x2 [%0], %1;"
          :
          : "l"(addr), "r"(h)
          : "memory"
      );
    } else if constexpr (std::is_same<T, half>::value){
      uint32_t h = pack32<half>(val.data[0], val.data[1]);
      asm volatile (
          "multimem.st.global.f16x2 [%0], %1;"
          :
          : "l"(addr), "r"(h)
          : "memory"
      );
    } else if constexpr (std::is_same<T, float>::value){
      asm volatile (
          "multimem.st.global.f32 [%0], %1;"
          :
          : "l"(addr), "f"(val.data[0])
          : "memory"
      );
    }
}

template <typename T, int N>
FLAGCX_DEVICE_INLINE_DECORATOR void lsa_ldst(T *srcAddr, T *dstAddr) {
    if constexpr (std::is_same<T, nv_bfloat16>::value) {
      dstAddr[0] = srcAddr[0];
      dstAddr[1] = srcAddr[1];
    } else if constexpr (std::is_same<T, half>::value){
      dstAddr[0] = srcAddr[0];
      dstAddr[1] = srcAddr[1];
    } else if constexpr (std::is_same<T, float>::value){
      dstAddr[0] = srcAddr[0];
    }
}

template <typename T, int N>
FLAGCX_DEVICE_INLINE_DECORATOR void lsa_st(T *addr, array_t<T,N> val) {
    if constexpr (std::is_same<T, nv_bfloat16>::value) {
      addr[0] = val.data[0];
      addr[1] = val.data[1];
    } else if constexpr (std::is_same<T, half>::value){
      addr[0] = val.data[0];
      addr[1] = val.data[1];
    } else if constexpr (std::is_same<T, float>::value){
      addr[0] = val.data[0];
    }
}

template <typename T>
__global__ void localAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
                                     void *recvbuffer, size_t count, int root,
                                     struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm,
                                         ncclTeamLsa(devComm),
                                         devComm.lsaBarrier, blockIdx.x, true};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const int globalTid = threadIdx.x + blockDim.x * blockIdx.x;
  const int globalNthreads = blockDim.x * gridDim.x;
  using P = typename packed_t<T, 4>::P;
  size_t pSize = packed_t<T, 4>::P::size;
  count /= pSize;

  T* mmSendPtr = (T*)ncclGetLsaMultimemPointer(sendwin, sendoffset, devComm);
  T* lsaRecvPtr = (T*)recvbuffer;

#pragma unroll
  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    array_t<T, 2> v = multimem_sum<T, 2>(mmSendPtr+pSize*offset);
    lsa_st<T, 2>(lsaRecvPtr + pSize*offset, v);
  }
}

template <typename T>
__global__ void interleavedAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
                                           ncclWindow_t recvwin, size_t recvoffset,
                                           void *recvbuffer, size_t count, int root,
                                           struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm,
                                         ncclTeamLsa(devComm),
                                         devComm.lsaBarrier, blockIdx.x, true};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;
  using P = typename packed_t<T, 4>::P;
  size_t pSize = packed_t<T, 4>::P::size;
  count /= pSize;

  T* mmSendPtr = (T*)ncclGetLsaMultimemPointer(sendwin, sendoffset, devComm);
  T* mmRecvPtr = (T*)ncclGetLsaMultimemPointer(recvwin, recvoffset, devComm);

#pragma unroll
  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    array_t<T, 2> v = multimem_sum<T, 2>(mmSendPtr+pSize*offset);
    multimem_st<T, 2>(mmRecvPtr+pSize*offset, v);
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// template <typename T>
// __global__ void twoStageAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
//                                         ncclWindow_t recvwin, size_t recvoffset,
//                                         void *recvbuffer, size_t count, int root,
//                                         struct ncclDevComm devComm) {
//   ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm,
//                                          ncclTeamLsa(devComm),
//                                          devComm.lsaBarrier, blockIdx.x, true};
//   bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

//   const int rank = devComm.rank, nRanks = devComm.nRanks;
//   const int globalTid = threadIdx.x + blockDim.x * blockIdx.x;
//   const int globalNthreads = blockDim.x * gridDim.x;
//   using P = typename packed_t<T, 4>::P;
//   size_t pSize = packed_t<T, 4>::P::size;
//   count /= pSize;
//   size_t part = count / nRanks;
//   size_t largestPart = part + count % nRanks;
//   size_t start = rank * part;
//   size_t end = (rank == nRanks - 1) ? count : start + part;
//   T* mmSendPtr = (T*)ncclGetLsaMultimemPointer(sendwin, sendoffset, devComm);
//   T* mmRecvPtr = (T*)ncclGetLsaMultimemPointer(recvwin, recvoffset, devComm);
//   T* lsaRecvPtr = (T*)recvbuffer;

//   // stage 1: reduce scatter
// #pragma unroll
//   for (size_t offset = start + globalTid; offset < end; offset += globalNthreads) {
//     array_t<T, 2> v = multimem_sum<T, 2>(mmSendPtr+pSize*offset);
//     multimem_st<T, 2>(mmRecvPtr+pSize*offset, v);
//   }
//   bar.sync(ncclCoopCta(), cuda::memory_order_release);

//   // stage 2: allgather
//   for (int idx = globalTid; idx < largestPart; idx += globalNthreads) {
// #pragma unroll
//     for (int i = 0; i < nRanks; i++) {
//       T* lsaTmpRecvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, i);
//       int gatherFromRank = ((rank + i) % nRanks);
//       if (gatherFromRank == nRanks - 1 || idx < part) {
//         int dstIdx = gatherFromRank * part + idx;
//         lsa_ldst<T, 2>(lsaTmpRecvPtr + pSize*dstIdx, lsaRecvPtr + pSize*dstIdx);
//       }
//     }
//   }
// }

// Helper function to launch appropriate kernel
template <typename T>
void launchLocalAllReduceKernel(ncclWindow_t send_win, void *recvbuffer,
                                size_t count, ncclDevComm& devComm,
                                cudaStream_t stream) {
  localAllReduceKernel<T><<<NCCL_ADAPTOR_DEVICE_CTA_COUNT, NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA, 0,
                            stream>>>(send_win, 0, recvbuffer, count, 0, devComm);
}

template <typename T>
void launchInterleavedAllReduceKernel(ncclWindow_t send_win, ncclWindow_t recv_win, void *recvbuffer,
                                      size_t count, ncclDevComm& devComm,
                                      cudaStream_t stream) {
  interleavedAllReduceKernel<T><<<NCCL_ADAPTOR_DEVICE_CTA_COUNT, NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA, 0,
                                  stream>>>(send_win, 0, recv_win, 0, recvbuffer, count, 0, devComm);
  // twoStageAllReduceKernel<T><<<NCCL_ADAPTOR_DEVICE_CTA_COUNT, NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA, 0,
  //                             stream>>>(send_win, 0, recv_win, 0, recvbuffer, count, 0, devComm);
}

extern "C"
ncclResult_t ncclAdaptorLocalAllReduce(const void *sendbuff,
                                       void *recvbuff,
                                       ncclWindow_t send_win,
                                       ncclWindow_t recv_win,
                                       size_t count,
                                       ncclDataType_t datatype,
                                       ncclRedOp_t op,
                                       ncclDevComm &devComm,
                                       cudaStream_t stream) {
  // Launch kernel based on datatype
  switch (datatype) {
    case ncclFloat32:
      launchLocalAllReduceKernel<float>(send_win, recvbuff, count, devComm, stream);
      break;
    case ncclFloat16:
      launchLocalAllReduceKernel<half>(send_win, recvbuff, count, devComm, stream);
      break;
    case ncclBfloat16:
      launchLocalAllReduceKernel<nv_bfloat16>(send_win, recvbuff, count, devComm, stream);
      break;
    default:
      return ncclInvalidArgument;
  }
  return ncclSuccess;

}

extern "C"
ncclResult_t ncclAdaptorInterleavedAllReduce(const void *sendbuff,
                                             void *recvbuff,
                                             ncclWindow_t send_win,
                                             ncclWindow_t recv_win,
                                             size_t count,
                                             ncclDataType_t datatype,
                                             ncclRedOp_t op,
                                             ncclDevComm &devComm,
                                             cudaStream_t stream) {
  // Launch kernel based on datatype
  switch (datatype) {
    case ncclFloat32:
      launchInterleavedAllReduceKernel<float>(send_win, recv_win, recvbuff, count, devComm, stream);
      break;
    case ncclFloat16:
      launchInterleavedAllReduceKernel<half>(send_win, recv_win, recvbuff, count, devComm, stream);
      break;
    case ncclBfloat16:
      launchInterleavedAllReduceKernel<nv_bfloat16>(send_win, recv_win, recvbuff, count, devComm, stream);
      break;
    default:
      return ncclInvalidArgument;
  }
  return ncclSuccess;
}
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)