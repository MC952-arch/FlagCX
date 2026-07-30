/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Test-only Device API kernel declarations.
 * These kernels are compiled from device_api.cu in test/kernel/[platform]/
 * NOT part of libflagcx.so.
 ************************************************************************/

#ifndef TEST_KERNEL_DEVICE_API_H_
#define TEST_KERNEL_DEVICE_API_H_

#include "flagcx_kernel.h"

// Intra-node AllReduce using FlagCX Device API.
flagcxResult_t flagcxIntraAllReduce(flagcxDevMem_t devMem, size_t count,
                                    flagcxDataType_t datatype,
                                    flagcxDevComm_t devComm,
                                    flagcxStream_t stream);

// Inter-node one-sided AlltoAll (put + waitSignal + flush).
flagcxResult_t flagcxInterOneSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

// Inter-node two-sided AlltoAll (send/recv + term/wait via FIFO).
flagcxResult_t flagcxInterTwoSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

// Inter-node Device API test kernels.
flagcxResult_t flagcxInterTestPutSignalInc(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

flagcxResult_t flagcxInterTestPutSignalAddDecoupled(
    flagcxDevMem_t sendMem, flagcxDevMem_t recvMem, size_t count,
    flagcxDataType_t datatype, flagcxDevComm_t devComm, flagcxStream_t stream);

flagcxResult_t
flagcxInterTestCounterPipeline(flagcxDevMem_t sendMem, flagcxDevMem_t recvMem,
                               size_t count, flagcxDataType_t datatype,
                               flagcxDevComm_t devComm, flagcxStream_t stream,
                               uint64_t *resultBuf);

flagcxResult_t flagcxInterTestPutValue(flagcxDevMem_t recvMem,
                                       flagcxDevComm_t devComm,
                                       flagcxStream_t stream,
                                       size_t putValBase);

flagcxResult_t flagcxInterTestSignal(flagcxDevComm_t devComm,
                                     flagcxStream_t stream);

flagcxResult_t
flagcxInterTestFlushDecouple(flagcxDevMem_t sendMem, flagcxDevMem_t recvMem,
                             size_t count, flagcxDataType_t datatype,
                             flagcxDevComm_t devComm, flagcxStream_t stream);

flagcxResult_t flagcxInterTestFollowShadow(flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

flagcxResult_t flagcxInterTestMeetShadow(flagcxDevComm_t devComm,
                                         flagcxStream_t stream);

flagcxResult_t flagcxInterTestReset(flagcxDevComm_t devComm,
                                    flagcxStream_t stream, uint64_t *resultBuf);

flagcxResult_t flagcxInterTestGet(flagcxDevMem_t sendMem,
                                  flagcxDevMem_t recvMem, size_t count,
                                  flagcxDataType_t datatype,
                                  flagcxDevComm_t devComm,
                                  flagcxStream_t stream);

// =========================================================================
// Intra-node Device API test kernels (test_device_api_intra)
// =========================================================================

// K1: Local Pointer — verify flagcxGetLocalPointer returns rawPtr
// results[0] = (localPtr == rawPtr) ? 1 : 0
flagcxResult_t flagcxIntraTestLocalPointer(flagcxDevMem_t devMem, void *rawPtr,
                                           int *results, flagcxStream_t stream);

// K2: Intra Pointer — read peer's buffer via flagcxGetIntraPointer
// Writes peer's data into output buffer
flagcxResult_t flagcxIntraTestIntraPointer(flagcxDevMem_t devMem,
                                           flagcxDevComm_t devComm,
                                           float *output, size_t count,
                                           flagcxStream_t stream);

// K3: Peer Pointer (team) — read peer via flagcxGetPeerPointer(mem, off, team,
// peer)
flagcxResult_t flagcxIntraTestPeerPointer(flagcxDevMem_t devMem,
                                          flagcxDevComm_t devComm,
                                          float *output, size_t count,
                                          flagcxStream_t stream);

// K5: Intra Barrier Sync — write local, barrier, read peer
flagcxResult_t flagcxIntraTestBarrierSync(flagcxDevMem_t devMem,
                                          flagcxDevComm_t devComm,
                                          float *output, size_t count,
                                          flagcxStream_t stream);

// K6: Intra Barrier Arrive/Wait — write local, arrive, wait, read peer
flagcxResult_t flagcxIntraTestBarrierArriveWait(flagcxDevMem_t devMem,
                                                flagcxDevComm_t devComm,
                                                float *output, size_t count,
                                                flagcxStream_t stream);

// K7: SymPtr — test flagcxSymPtr<float> localPtr/intraPtr + arithmetic
// results[0] = all checks pass ? 1 : 0
flagcxResult_t flagcxIntraTestSymPtr(flagcxDevMem_t devMem,
                                     flagcxDevComm_t devComm, int *results,
                                     flagcxStream_t stream);

// K9: DevMem & Comm Queries — hasWindow, getIntraRank/Size, getRank/Size
// results[0..5] = hasWindow, intraRank, intraSize, rank, size, hasPeerPtrs
flagcxResult_t flagcxIntraTestCommQueries(flagcxDevMem_t devMem,
                                          flagcxDevComm_t devComm, int *results,
                                          flagcxStream_t stream);

// K10: Coop Groups — test all coop types threadRank/size/sync
// results[0..N] = pass/fail for each coop type
flagcxResult_t flagcxIntraTestCoopGroups(int *results, flagcxStream_t stream);

// K11: Team — flagcxTeamIntra, RankToWorld, RankToIntra, RankIsMember
// results[0..N] = pass/fail for each team query
flagcxResult_t flagcxIntraTestTeam(flagcxDevComm_t devComm, int *results,
                                   flagcxStream_t stream);

// K12: Intra AllReduce (composite) — end-to-end peer pointer + barrier
// Already declared above as flagcxIntraAllReduce

#endif // TEST_KERNEL_DEVICE_API_H_
