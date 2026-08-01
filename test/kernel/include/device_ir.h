/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Test-only Device IR kernel declarations.
 * These kernels exercise both API paths:
 *   - Struct-based IR wrappers: K1–K8
 *   - S-suffixed (scalar) IR functions:      S1–S10
 *
 * Compiled from device_ir.cu in test/kernel/[platform]/.
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

// =========================================================================
// Scalar IR (S-suffixed) kernel launchers
// =========================================================================

// S1: Cooperative Group (Scalar) — writes threadRank, coopSize per thread
void launchKernelCoopGroupS(const void *devCommPtr, int *devResults,
                            int nBlocks, int nThreads, flagcxStream_t stream);

// S2: Team Queries (Scalar) — writes intraRank, worldRank to results[0..1]
void launchKernelTeamQueriesS(const void *devCommPtr, int *devResults,
                              flagcxStream_t stream);

// S3: Local Pointer (Scalar) — verifies localPtr == rawBuff
void launchKernelLocalPointerS(const void *devMemPtr, void *rawBuff,
                               int *devResults, flagcxStream_t stream);

// S4: Intra Pointer (Scalar) — reads peer's data via LSA
void launchKernelIntraPointerS(const void *devCommPtr, const void *devMemPtr,
                               float *devOutput, int nBlocks, int nThreads,
                               flagcxStream_t stream);

// S5: Intra Barrier Sync (Scalar) — write buffer, barrier, read peer
void launchKernelIntraBarrierSyncS(const void *devCommPtr,
                                   const void *devMemPtr, float *buffer,
                                   float *output, int N, flagcxStream_t stream);

// S6: SyncS(Release) + read + SyncS(Acquire)
void launchKernelIntraBarrierSyncSplitS(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream);

// =========================================================================
// Extended Coop Kinds (S-suffixed)
// =========================================================================

// S7: TILE_SPAN coop — threadRankEx, sizeEx, syncEx
void launchKernelCoopTileSpanS(int *devResults, int nBlocks, int nThreads,
                               flagcxStream_t stream);

// S8: LANES coop — threadRankEx, sizeEx, syncEx (full warp mask)
void launchKernelCoopLanesS(int *devResults, flagcxStream_t stream);

// =========================================================================
// S-API Transport Tests
// =========================================================================

// S9: GetFromCommS — verify transport handle non-null
void launchKernelNetGetFromCommS(const void *devCommPtr, int *devResults,
                                 flagcxStream_t stream);

// S10: Signal/Counter local read/reset/shadow
void launchKernelNetSignalCounterS(const void *devCommPtr, int *devResults,
                                   flagcxStream_t stream);

// =========================================================================
// S-API Inter-Node Transport Tests (S11-S27)
// =========================================================================

// S11: WaitSignalS + FlushS — signal peer, peer waits + flushes (hang-free =
// PASS)
void launchKernelNetWaitSignalFlushS(const void *devCommPtr,
                                     flagcxStream_t stream);

// S12: WaitCounterS (COMMENTED — standalone counter signal not supported by
// GIN) void launchKernelNetWaitCounterS(const void *devCommPtr, flagcxStream_t
// stream);

// S13: WaitSignalMeetShadowS — increaseSignalShadow + signal + waitMeetShadow
void launchKernelNetWaitSignalMeetShadowS(const void *devCommPtr,
                                          flagcxStream_t stream);

// S14: PutS(None,None) + SignalSigIncS + WaitSignalS + FlushS — alltoall
void launchKernelNetPutS(const void *devCommPtr, const void *sendMemPtr,
                         const void *recvMemPtr, size_t countPerPeer,
                         flagcxStream_t stream);

// S15: PutS_RSigInc + WaitSignalS + FlushS — alltoall
void launchKernelNetPutRSigIncS(const void *devCommPtr, const void *sendMemPtr,
                                const void *recvMemPtr, size_t countPerPeer,
                                flagcxStream_t stream);

// S16: PutS_RSigAdd + WaitSignalS + FlushS — alltoall
void launchKernelNetPutRSigAddS(const void *devCommPtr, const void *sendMemPtr,
                                const void *recvMemPtr, size_t countPerPeer,
                                flagcxStream_t stream);

// S17: PutS_RSigInc_LCtrInc + WaitSignalS + WaitCounterS + FlushS — alltoall
void launchKernelNetPutRSigLCtrS(const void *devCommPtr, const void *sendMemPtr,
                                 const void *recvMemPtr, size_t countPerPeer,
                                 flagcxStream_t stream);

// S18: SignalSigIncS + WaitSignalS — signal round-trip (hang-free = PASS)
void launchKernelNetSignalSigIncS(const void *devCommPtr,
                                  flagcxStream_t stream);

// S19: SignalSigAddS + WaitSignalS — signal round-trip (hang-free = PASS)
void launchKernelNetSignalSigAddS(const void *devCommPtr,
                                  flagcxStream_t stream);

// S21: PutValueS(None) + SignalSigIncS + WaitSignalS — uint64 value transfer
void launchKernelNetPutValueS(const void *devCommPtr, const void *recvMemPtr,
                              size_t putValBase, flagcxStream_t stream);

// S22: PutValueS_RSigInc + WaitSignalS — uint64 value transfer with signal
void launchKernelNetPutValueRSigS(const void *devCommPtr,
                                  const void *recvMemPtr, size_t putValBase,
                                  flagcxStream_t stream);

// S23: GetS + FlushS — alltoall via one-sided get
void launchKernelNetGetS(const void *devCommPtr, const void *sendMemPtr,
                         const void *recvMemPtr, size_t countPerPeer,
                         flagcxStream_t stream);

// S24: SendS + RecvS + TermS + WaitS — two-sided alltoall (COMMENTED)
// void launchKernelNetTwoSidedS(const void *devCommPtr, const void *sendMemPtr,
//                               const void *recvMemPtr, size_t countPerPeer,
//                               flagcxStream_t stream);

// S25: Inter-Barrier Test
void launchKernelInterBarrierStress(const void *devCommPtr, int *devResults,
                                    int nIters, flagcxStream_t stream);

// S26: WorldBarrierSyncS — world barrier (hang-free = PASS)
void launchKernelWorldBarrierS(const void *devCommPtr, flagcxStream_t stream);

// S27: WorldBarrierArriveS + WorldBarrierWaitS — split world barrier
void launchKernelWorldBarrierSplitS(const void *devCommPtr,
                                    flagcxStream_t stream);

#endif // TEST_KERNEL_DEVICE_IR_H_
