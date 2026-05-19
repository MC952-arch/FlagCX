/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API C-style wrapper functions for LLVM IR generation.
 *
 * This header declares extern "C" device functions that wrap the C++
 * template-based FlagCX Device API. When compiled to LLVM bitcode,
 * these functions can be linked by LLVM-based languages (e.g. Triton).
 ************************************************************************/
#ifndef FLAGCX_DEVICE_WRAPPER_H_
#define FLAGCX_DEVICE_WRAPPER_H_

#include "flagcx_device_core.h"

/* ================================================================
 * C-compatible wrapper structs
 * ================================================================ */

struct flagcxIntraBarrierSession_C {
  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopAny> bar;
};

struct flagcxInterBarrierSession_C {
  flagcxDevBarrier<flagcxTeamTagInter, flagcxCoopAny> bar;
};

struct flagcxBarrierSession_C {
  flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopAny> bar;
};

/* ================================================================
 * Category 1: Comm Queries (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ int flagcxDevCommGetRank(const void *comm);
FLAGCX_IR_EXTERN_C __device__ int flagcxDevCommGetSize(const void *comm);
FLAGCX_IR_EXTERN_C __device__ int flagcxDevCommGetIntraRank(const void *comm);
FLAGCX_IR_EXTERN_C __device__ int flagcxDevCommGetIntraSize(const void *comm);

/* ================================================================
 * Category 2: Cooperative Group — Init / Query / Sync (8)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void flagcxCoopAnyInitBlock(void *coop);
FLAGCX_IR_EXTERN_C __device__ void flagcxCoopAnyInitWarp(void *coop);
FLAGCX_IR_EXTERN_C __device__ void flagcxCoopAnyInitThread(void *coop);
FLAGCX_IR_EXTERN_C __device__ void
flagcxCoopAnyInitTileSpan(void *coop, int t0, int nTiles, int id);
FLAGCX_IR_EXTERN_C __device__ void flagcxCoopAnyInitLanes(void *coop,
                                                          uint32_t laneMask);

FLAGCX_IR_EXTERN_C __device__ int flagcxCoopThreadRankC(const void *coop);
FLAGCX_IR_EXTERN_C __device__ int flagcxCoopSizeC(const void *coop);
FLAGCX_IR_EXTERN_C __device__ void flagcxCoopSyncC(void *coop);

/* ================================================================
 * Category 3: Team Functions (5)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void flagcxGetTeamIntra(const void *comm,
                                                      void *team);
FLAGCX_IR_EXTERN_C __device__ void flagcxGetTeamWorld(const void *comm,
                                                      void *team);
FLAGCX_IR_EXTERN_C __device__ void flagcxGetTeamInter(const void *comm,
                                                      void *team);
FLAGCX_IR_EXTERN_C __device__ int
flagcxTeamRankToWorldC(const void *comm, const void *team, int rank);
FLAGCX_IR_EXTERN_C __device__ int
flagcxTeamRankToIntraC(const void *comm, const void *team, int rank);

/* ================================================================
 * Category 4: Pointer Access (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void *flagcxGetPeerPointerC(const void *mem,
                                                          size_t offset,
                                                          const void *team,
                                                          int peer);
FLAGCX_IR_EXTERN_C __device__ void *flagcxGetLocalPointerC(const void *mem,
                                                           size_t offset);
FLAGCX_IR_EXTERN_C __device__ void *
flagcxGetIntraPointerC(const void *mem, size_t offset, int peer);
FLAGCX_IR_EXTERN_C __device__ void *
flagcxGetMulticastPointerC(const void *mem, size_t offset, const void *comm);

/* ================================================================
 * Category 5: Utility (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ size_t
flagcxDataTypeSizeDevice(flagcxDataType_t dt);

/* ================================================================
 * Category 6: Intra-Node Barrier Session (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxIntraBarrierSessionInit(void *session, const void *coop, const void *comm,
                              const void *team, uint32_t index, bool multimem);
FLAGCX_IR_EXTERN_C __device__ void
flagcxIntraBarrierSessionArrive(void *session, flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void
flagcxIntraBarrierSessionWait(void *session, flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void
flagcxIntraBarrierSessionSync(void *session, flagcxDeviceMemoryOrder_t order);

/* ================================================================
 * Category 7: Inter-Node Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxInterBarrierSessionInit(void *session, const void *coop,
                              const void *trans, const void *team,
                              uint32_t index);
FLAGCX_IR_EXTERN_C __device__ void
flagcxInterBarrierSessionSync(void *session, flagcxDeviceMemoryOrder_t order,
                              flagcxDevNetFenceLevel fence);

/* ================================================================
 * Category 8: World Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxWorldBarrierSessionInit(void *session, const void *coop,
                              flagcxTeamTagWorld tag, const void *trans,
                              uint32_t index, bool multimem);
FLAGCX_IR_EXTERN_C __device__ void
flagcxWorldBarrierSessionSync(void *session, flagcxDeviceMemoryOrder_t order,
                              flagcxDevNetFenceLevel fence);

/* ================================================================
 * Category 9: Transport — Init / Signal Read / Wait / Counter / Flush (7)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetInitC(void *trans,
                                                     const void *comm, int idx);
FLAGCX_IR_EXTERN_C __device__ uint64_t
flagcxDevNetReadSignal(const void *trans, flagcxDevNetSignal_t signalId,
                       int bits, flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetWaitSignal(const void *trans, const void *coop,
                       flagcxDevNetSignal_t signalId, uint64_t least, int bits,
                       flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetWaitSignalMeetShadow(const void *trans, const void *coop,
                                 flagcxDevNetSignal_t signalId, int bits,
                                 flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ uint64_t
flagcxDevNetReadCounter(const void *trans, flagcxDevNetCounter_t counterId,
                        int bits, flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetWaitCounter(const void *trans, const void *coop,
                        flagcxDevNetCounter_t counterId, uint64_t least,
                        int bits, flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetFlush(const void *trans, const void *coop,
                  flagcxDeviceMemoryOrder_t order);

/* ================================================================
 * Category 9b: Net — Reset / Shadow (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetResetSignal(const void *net, flagcxDevNetSignal_t slot);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetResetCounter(const void *net, flagcxDevNetCounter_t slot);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetIncreaseSignalShadow(const void *net, flagcxDevNetSignal_t slot,
                                 uint64_t delta);
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetWaitSignalFollowShadow(
    const void *net, const void *coop, flagcxDevNetSignal_t slot,
    uint64_t leastDelta, uint64_t *before, uint64_t *delta, int bits,
    flagcxDeviceMemoryOrder_t order);

/* ================================================================
 * Category 10: Transport — Two-Sided (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ int
flagcxDevNetSend(const void *trans, const void *coop, const void *mem,
                 size_t offset, size_t count, flagcxDataType_t datatype,
                 int peer);
FLAGCX_IR_EXTERN_C __device__ int
flagcxDevNetRecv(const void *trans, const void *coop, const void *mem,
                 size_t offset, size_t count, flagcxDataType_t datatype,
                 int peer);
FLAGCX_IR_EXTERN_C __device__ int flagcxDevNetWait(const void *trans,
                                                   const void *coop);
FLAGCX_IR_EXTERN_C __device__ int flagcxDevNetTerm(const void *trans,
                                                   const void *coop);

/* ================================================================
 * Category 11: Transport — One-Sided put (16)
 *
 * Naming: flagcxDevNetPut[_R<remote>][_L<local>]
 * Actions: None, SigInc, SigAdd, CtrInc
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetPut(const void *trans, const void *team, int peer, const void *dst,
                size_t dstOffset, const void *src, size_t srcOffset,
                size_t bytes, const void *coop);

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetPut_RSigInc(const void *trans, const void *team, int peer,
                        const void *dst, size_t dstOffset, const void *src,
                        size_t srcOffset, size_t bytes, const void *coop,
                        flagcxDevNetSignal_t remoteSignal);

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RSigAdd(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue);

/* (CtrInc, None) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetPut_RCtrInc(const void *trans, const void *team, int peer,
                        const void *dst, size_t dstOffset, const void *src,
                        size_t srcOffset, size_t bytes, const void *coop,
                        flagcxDevNetCounter_t remoteCounter);

/* (None, SigInc) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetPut_LSigInc(const void *trans, const void *team, int peer,
                        const void *dst, size_t dstOffset, const void *src,
                        size_t srcOffset, size_t bytes, const void *coop,
                        flagcxDevNetSignal_t localSignal);

/* (SigInc, SigInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RSigInc_LSigInc(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetSignal_t remoteSignal,
    flagcxDevNetSignal_t localSignal);

/* (SigAdd, SigInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RSigAdd_LSigInc(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevNetSignal_t localSignal);

/* (CtrInc, SigInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RCtrInc_LSigInc(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetCounter_t remoteCounter,
    flagcxDevNetSignal_t localSignal);

/* (None, SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetPut_LSigAdd(const void *trans, const void *team, int peer,
                        const void *dst, size_t dstOffset, const void *src,
                        size_t srcOffset, size_t bytes, const void *coop,
                        flagcxDevNetSignal_t localSignal, uint64_t localValue);

/* (SigInc, SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RSigInc_LSigAdd(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetSignal_t remoteSignal,
    flagcxDevNetSignal_t localSignal, uint64_t localValue);

/* (SigAdd, SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RSigAdd_LSigAdd(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevNetSignal_t localSignal, uint64_t localValue);

/* (CtrInc, SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RCtrInc_LSigAdd(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetCounter_t remoteCounter,
    flagcxDevNetSignal_t localSignal, uint64_t localValue);

/* (None, CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetPut_LCtrInc(const void *trans, const void *team, int peer,
                        const void *dst, size_t dstOffset, const void *src,
                        size_t srcOffset, size_t bytes, const void *coop,
                        flagcxDevNetCounter_t localCounter);

/* (SigInc, CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RSigInc_LCtrInc(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetSignal_t remoteSignal,
    flagcxDevNetCounter_t localCounter);

/* (SigAdd, CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RSigAdd_LCtrInc(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevNetCounter_t localCounter);

/* (CtrInc, CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPut_RCtrInc_LCtrInc(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetCounter_t remoteCounter,
    flagcxDevNetCounter_t localCounter);

/* ================================================================
 * Category 12: Transport — One-Sided signal (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetSignalSigInc(const void *trans, const void *team, int peer,
                         const void *coop, flagcxDevNetSignal_t signal);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetSignalSigAdd(const void *trans, const void *team, int peer,
                         const void *coop, flagcxDevNetSignal_t signal,
                         uint64_t value);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetSignalCtrInc(const void *trans, const void *team, int peer,
                         const void *coop, flagcxDevNetCounter_t counter);

/* ================================================================
 * Category 13: Transport — One-Sided putValue<uint64_t> (4)
 *
 * C++ putValue only supports RemoteAction (no LocalAction).
 * ================================================================ */

/* (None) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetPutValue(const void *trans, const void *team, int peer,
                     const void *dst, size_t dstOffset, uint64_t value,
                     const void *coop);

/* (SigInc) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetPutValue_RSigInc(const void *trans, const void *team, int peer,
                             const void *dst, size_t dstOffset, uint64_t value,
                             const void *coop,
                             flagcxDevNetSignal_t remoteSignal);

/* (SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevNetPutValue_RSigAdd(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, uint64_t value, const void *coop,
    flagcxDevNetSignal_t remoteSignal, uint64_t remoteAddValue);

/* (CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetPutValue_RCtrInc(const void *trans, const void *team, int peer,
                             const void *dst, size_t dstOffset, uint64_t value,
                             const void *coop,
                             flagcxDevNetCounter_t remoteCounter);

/* ================================================================
 * Category 14: Transport — One-Sided get (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxDevNetGet(const void *trans, const void *team, int peer, const void *src,
                size_t srcOffset, const void *dst, size_t dstOffset,
                size_t bytes, const void *coop);

#endif /* FLAGCX_DEVICE_WRAPPER_H_ */
