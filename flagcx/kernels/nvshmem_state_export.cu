/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM device-state export helper.
 * Compiled by nvcc, device-linked into libflagcx.so alongside
 * libnvshmem_device.a.  Provides flagcxNvshmemGetDeviceState() which
 * reads the library's initialized __constant__ nvshmemi_device_state_d
 * and copies it to a host buffer.
 *
 * External consumers (test binaries) that device-link their own copy of
 * libnvshmem_device.a call this to get the initialized state and then
 * cudaMemcpyToSymbol it into their own __constant__ symbol.
 ************************************************************************/

#ifdef FLAGCX_COMM_TRAITS_SHMEM

#include <cuda_runtime.h>
#include <nvshmem.h>
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

// nvshmemi_device_state_d is the __constant__ symbol defined in
// libnvshmem_device.a, resolved at device-link time to the library's copy.

extern "C" void flagcxNvshmemGetDeviceState(void *buf, size_t bufSize) {
    size_t copySize = bufSize < sizeof(nvshmemi_device_host_state_t)
                          ? bufSize
                          : sizeof(nvshmemi_device_host_state_t);
    cudaMemcpyFromSymbol(buf, nvshmemi_device_state_d, copySize);
}

extern "C" void flagcxNvshmemGetHeapState(void **heapBase, size_t *heapSize,
                                          void ***peerHeapBaseP2P) {
    nvshmemi_device_host_state_t state;
    cudaError_t err = cudaMemcpyFromSymbol(&state, nvshmemi_device_state_d, sizeof(state));
    printf("[flagcxNvshmemGetHeapState] cudaMemcpyFromSymbol err=%d, "
           "heap_base=%p, heap_size=%zu, peer_heap_base_p2p=%p\n",
           (int)err, state.heap_base, state.heap_size, state.peer_heap_base_p2p);
    *heapBase = state.heap_base;
    *heapSize = state.heap_size;
    *peerHeapBaseP2P = state.peer_heap_base_p2p;
}

#endif // FLAGCX_COMM_TRAITS_SHMEM
