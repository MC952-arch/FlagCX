/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Test-only Device IR kernel declarations.
 * These kernels exercise the FlagCX Device API IR wrapper functions
 * (extern "C" wrappers compiled via nvcc here, or via LLVM bitcode for
 * Triton). Compiled from device_ir.cu in test/device_api/.
 ************************************************************************/

#ifndef TEST_KERNEL_DEVICE_IR_H_
#define TEST_KERNEL_DEVICE_IR_H_

#include "flagcx.h"

// K1: Comm Queries — writes rank, size, intraRank, intraSize to results[0..3]
void launchKernelCommQueries(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream);

// K2: Cooperative Group — writes threadRank, coopSize per thread
void launchKernelCoopGroup(const void *devCommPtr, int *devResults, int nBlocks,
                           int nThreads, flagcxStream_t stream);

// K3: Team Queries — writes intraRank, worldRank to results[0..1]
void launchKernelTeamQueries(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream);

// K4: Local Pointer — verifies localPtr == rawBuff
void launchKernelLocalPointer(const void *devMemPtr, void *rawBuff,
                              int *devResults, flagcxStream_t stream);

// K5: Intra Pointer — reads peer's data via LSA
void launchKernelIntraPointer(const void *devCommPtr, const void *devMemPtr,
                              float *devOutput, int nBlocks, int nThreads,
                              flagcxStream_t stream);

// K6: Data Type Size — writes sizeof for 5 types to results[0..4]
void launchKernelDataTypeSize(int *devResults, flagcxStream_t stream);

// K7: Intra Barrier Sync — write buffer, barrier, read peer
void launchKernelIntraBarrierSync(const void *devCommPtr, const void *devMemPtr,
                                  float *buffer, float *output, int N,
                                  flagcxStream_t stream);

// K8: Intra Barrier Arrive/Wait — write buffer, arrive, wait, read peer
void launchKernelIntraBarrierArriveWait(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream);

#endif // TEST_KERNEL_DEVICE_IR_H_
