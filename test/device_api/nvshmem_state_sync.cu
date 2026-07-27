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
 * scope — copies the initialized state from libflagcx.so (via
 * flagcxNvshmemGetDeviceState) into the consumer's __constant__ symbol.
 *
 * Must be called AFTER flagcxCommInitRank (which triggers nvshmem_init)
 * and BEFORE any kernel launch that uses NVSHMEM device functions.
 ************************************************************************/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <nvshmem.h>
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

// Provided by libflagcx.so (nvshmem_state_export.cu).
// Reads the library's initialized nvshmemi_device_state_d into buf.
extern "C" void flagcxNvshmemGetDeviceState(void *buf, size_t bufSize);

extern "C" void flagcxNvshmemSyncDeviceState() {
    nvshmemi_device_host_state_t hostState;
    flagcxNvshmemGetDeviceState(&hostState, sizeof(hostState));
    fprintf(stderr, "[SyncDeviceState] src: mype=%d, npes=%d, heap_base=%p, "
            "heap_size=%zu, peer_heap_base_p2p=%p, job_connectivity=%d\n",
            hostState.mype, hostState.npes, hostState.heap_base,
            hostState.heap_size, (void *)hostState.peer_heap_base_p2p,
            hostState.job_connectivity);

    cudaError_t err = cudaMemcpyToSymbol(nvshmemi_device_state_d, &hostState,
                                         sizeof(nvshmemi_device_host_state_t));
    fprintf(stderr, "[SyncDeviceState] cudaMemcpyToSymbol err=%d\n", (int)err);

    // Verify: read back and confirm non-zero
    nvshmemi_device_host_state_t verify;
    memset(&verify, 0, sizeof(verify));
    cudaError_t err2 = cudaMemcpyFromSymbol(&verify, nvshmemi_device_state_d,
                                            sizeof(verify));
    fprintf(stderr, "[SyncDeviceState] verify: err=%d, mype=%d, npes=%d, "
            "heap_base=%p, job_connectivity=%d\n",
            (int)err2, verify.mype, verify.npes, verify.heap_base,
            verify.job_connectivity);
}
