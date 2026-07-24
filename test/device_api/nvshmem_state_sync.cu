/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Syncs the NVSHMEM device state into the consumer binary's own
 * __constant__ nvshmemi_device_state_d symbol.
 *
 * Problem: nvshmem_init() inside libflagcx.so populates the library's
 * device-link copy of nvshmemi_device_state_d. But consumers that
 * device-link libnvshmem_device.a separately get their own uninitialized
 * copy. This file — compiled with RDC and device-linked in the consumer's
 * scope — provides a function to copy the host-side mirror into the
 * consumer's __constant__ symbol.
 *
 * Must be called AFTER flagcxCommInitRank (which triggers nvshmem_init)
 * and BEFORE any kernel launch that uses NVSHMEM device functions.
 ************************************************************************/

#include <cuda_runtime.h>
#include <nvshmem.h>
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

// Host-side mirror of device state (defined in libnvshmem_host, populated
// by nvshmem_init). Accessible because we link against libnvshmem_host.
extern nvshmemi_device_host_state_t nvshmemi_device_state;

extern "C" void flagcxNvshmemSyncDeviceState() {
    cudaMemcpyToSymbol(nvshmemi_device_state_d, &nvshmemi_device_state,
                       sizeof(nvshmemi_device_host_state_t));
}
