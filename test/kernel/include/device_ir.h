/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Test-only Device IR kernel declarations.
 * These kernels exercise S-suffixed (scalar) IR functions.
 *
 * Intra-node tests (S1–S10) are aligned with device_api_intra K1–K10.
 * Inter-node tests (S1–S15) are aligned with device_api_inter K1–K15.
 *
 * Compiled from device_ir.cu in test/kernel/[platform]/.
 ************************************************************************/

#ifndef TEST_KERNEL_DEVICE_IR_H_
#define TEST_KERNEL_DEVICE_IR_H_

#include "flagcx.h"

// =========================================================================
// Intra-Node Scalar IR kernel launchers (S1–S10)
// =========================================================================

// S1: Comm Queries — rank, size, intraRank, intraSize
void launchKernelCommQueriesS(const void *devCommPtr, int *devResults,
                              flagcxStream_t stream);

// S2: Coop Groups — block, tile_span, lanes (results[0..2] = pass flags)
void launchKernelCoopGroupsS(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream);

// S3: Team Queries — writes intraRank, worldRank to results[0..1]
void launchKernelTeamQueriesS(const void *devCommPtr, int *devResults,
                              flagcxStream_t stream);

// S4: Local Pointer — verifies localPtr == rawBuff
void launchKernelLocalPointerS(const void *devMemPtr, void *rawBuff,
                               int *devResults, flagcxStream_t stream);

// S5: Intra Pointer — reads peer's data via intra pointer
void launchKernelIntraPointerS(const void *devCommPtr, const void *devMemPtr,
                               float *devOutput, int count,
                               flagcxStream_t stream);

// S6: Peer Pointer — team-based peer memory access
void launchKernelPeerPointerS(const void *devCommPtr, const void *devMemPtr,
                              float *devOutput, int count,
                              flagcxStream_t stream);

// S7: Multicast Pointer — NVLS-dependent, commented out
// void launchKernelMulticastPointerS(const void *devCommPtr,
//                                    const void *devMemPtr, float *devOutput,
//                                    int nBlocks, int nThreads,
//                                    flagcxStream_t stream);

// S8: Intra Barrier Sync — write buffer, barrier, read peer
void launchKernelIntraBarrierSyncS(const void *devCommPtr,
                                   const void *devMemPtr, float *buffer,
                                   float *output, int N, flagcxStream_t stream);

// S9: Intra Barrier Arrive/Wait — SyncS(Release) + read + SyncS(Acquire)
void launchKernelIntraBarrierArriveWaitS(const void *devCommPtr,
                                         const void *devMemPtr, float *buffer,
                                         float *output, int N,
                                         flagcxStream_t stream);

// S10: Intra AllReduce — composite using barriers + pointers
void launchKernelIntraAllReduceS(const void *devCommPtr, const void *devMemPtr,
                                 float *buffer, int count,
                                 flagcxStream_t stream);

// =========================================================================
// Inter-Node Transport Tests (S1–S15, aligned with device_api_inter K1–K15)
// =========================================================================

// S1: Transport Handle — verify NetGetFromCommS non-null
void launchKernelNetGetFromCommS(const void *devCommPtr, int *devResults,
                                 flagcxStream_t stream);

// S2: Signal/Counter Reset — read/reset/shadow
void launchKernelNetResetS(const void *devCommPtr, int *devResults,
                           flagcxStream_t stream);

// S3: Put + SigInc — PutS_RSigInc + WaitSignalS + FlushS
void launchKernelNetPutSignalIncS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream);

// S4: Put + SigAdd — PutS_RSigAdd + WaitSignalS + FlushS
void launchKernelNetPutSignalAddS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream);

// S5: Put + SigInc + CtrInc — PutS_RSigInc_LCtrInc + WaitSignalS + WaitCounterS
// + FlushS
void launchKernelNetCounterPipelineS(const void *devCommPtr,
                                     const void *sendMemPtr,
                                     const void *recvMemPtr,
                                     size_t countPerPeer,
                                     flagcxStream_t stream);

// S6: Put(None) + Flush + Signal (FlushDecouple) — PutS + FlushS +
// SignalSigIncS + WaitSignalS + FlushS
void launchKernelNetFlushDecoupleS(const void *devCommPtr,
                                   const void *sendMemPtr,
                                   const void *recvMemPtr, size_t countPerPeer,
                                   flagcxStream_t stream);

// S7: PutValue — PutValueS(None)+Signal then PutValueS_RSigInc (both in one
// kernel)
void launchKernelNetPutValueS(const void *devCommPtr, const void *recvMemPtr,
                              size_t putValBase, flagcxStream_t stream);

// S8: Get — GetS + FlushS
void launchKernelNetGetS(const void *devCommPtr, const void *sendMemPtr,
                         const void *recvMemPtr, size_t countPerPeer,
                         flagcxStream_t stream);

// S9: Signal — SignalSigIncS + SignalSigAddS + WaitSignalS (both in one kernel)
void launchKernelNetSignalS(const void *devCommPtr, flagcxStream_t stream);

// S10: Shadow (commented — MeetShadowS)
void launchKernelNetWaitSignalMeetShadowS(const void *devCommPtr,
                                          flagcxStream_t stream);

// S11: WaitSignal + Flush (standalone)
void launchKernelNetWaitSignalFlushS(const void *devCommPtr,
                                     flagcxStream_t stream);

// S12: Inter Barrier — stress test
void launchKernelInterBarrierS(const void *devCommPtr, int *devResults,
                               int nIters, flagcxStream_t stream);

// S13: World Barrier — sync + arrive/wait split
void launchKernelWorldBarrierS(const void *devCommPtr, flagcxStream_t stream);

// S14: AlltoAll (one-sided composite) — put + signal + wait + flush + world
// barrier
void launchKernelNetOneSidedAlltoAllS(const void *devCommPtr,
                                      const void *sendMemPtr,
                                      const void *recvMemPtr,
                                      size_t countPerPeer,
                                      flagcxStream_t stream);

// S15: AlltoAll (two-sided, commented)
// void launchKernelNetTwoSidedS(const void *devCommPtr, const void *sendMemPtr,
//                               const void *recvMemPtr, size_t countPerPeer,
//                               flagcxStream_t stream);

#endif // TEST_KERNEL_DEVICE_IR_H_
