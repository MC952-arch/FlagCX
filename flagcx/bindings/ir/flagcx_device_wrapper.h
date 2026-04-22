/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API C-style wrapper functions for LLVM IR generation.
 *
 * This header declares extern "C" device functions that wrap the C++
 * template-based FlagCX Device API. When compiled to LLVM bitcode,
 * these functions can be linked by LLVM-based languages (e.g. Triton).
 *
 * Typed load/store uses NVSHMEM-style X-macro generation.
 ************************************************************************/
#ifndef FLAGCX_DEVICE_WRAPPER_H_
#define FLAGCX_DEVICE_WRAPPER_H_

#include "flagcx_device.h"

/* ================================================================
 * X-macro: typed load/store generation (NVSHMEM pattern)
 * Platform-specific — defines FLAGCX_REPT_FOR_DEVICE_TYPES(FN).
 * ================================================================ */
#include "flagcx_device_types.h"

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

FLAGCX_IR_EXTERN_C __device__ int
flagcxDevCommGetRank(const flagcxDevComm *comm);
FLAGCX_IR_EXTERN_C __device__ int
flagcxDevCommGetSize(const flagcxDevComm *comm);
FLAGCX_IR_EXTERN_C __device__ int
flagcxDevCommGetIntraRank(const flagcxDevComm *comm);
FLAGCX_IR_EXTERN_C __device__ int
flagcxDevCommGetIntraSize(const flagcxDevComm *comm);

/* ================================================================
 * Category 2: Cooperative Group — Init / Query / Sync (8)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void flagcxCoopAnyInitBlock(flagcxCoopAny *coop);
FLAGCX_IR_EXTERN_C __device__ void flagcxCoopAnyInitWarp(flagcxCoopAny *coop);
FLAGCX_IR_EXTERN_C __device__ void flagcxCoopAnyInitThread(flagcxCoopAny *coop);
FLAGCX_IR_EXTERN_C __device__ void
flagcxCoopAnyInitTileSpan(flagcxCoopAny *coop, int t0, int nTiles, int id);
FLAGCX_IR_EXTERN_C __device__ void flagcxCoopAnyInitLanes(flagcxCoopAny *coop,
                                                          uint32_t laneMask);

FLAGCX_IR_EXTERN_C __device__ int
flagcxCoopThreadRankC(const flagcxCoopAny *coop);
FLAGCX_IR_EXTERN_C __device__ int flagcxCoopSizeC(const flagcxCoopAny *coop);
FLAGCX_IR_EXTERN_C __device__ void flagcxCoopSyncC(flagcxCoopAny *coop);

/* ================================================================
 * Category 3: Team Functions (5)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ flagcxTeam
flagcxGetTeamIntra(const flagcxDevComm *comm);
FLAGCX_IR_EXTERN_C __device__ flagcxTeam
flagcxGetTeamWorld(const flagcxDevComm *comm);
FLAGCX_IR_EXTERN_C __device__ flagcxTeam
flagcxGetTeamInter(const flagcxDevComm *comm);
FLAGCX_IR_EXTERN_C __device__ int
flagcxTeamRankToWorldC(const flagcxDevComm *comm, flagcxTeam team, int rank);
FLAGCX_IR_EXTERN_C __device__ int
flagcxTeamRankToIntraC(const flagcxDevComm *comm, flagcxTeam team, int rank);

/* ================================================================
 * Category 4: Pointer Access (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void *
flagcxGetPeerPointerTeam(const flagcxDevMem *mem, size_t offset,
                         flagcxTeam team, int peer);
FLAGCX_IR_EXTERN_C __device__ void *
flagcxGetLocalPointerC(const flagcxDevMem *mem, size_t offset);
FLAGCX_IR_EXTERN_C __device__ void *
flagcxGetIntraPointerC(const flagcxDevMem *mem, size_t offset, int peer);
FLAGCX_IR_EXTERN_C __device__ void *
flagcxGetMulticastPointerC(const flagcxDevMem *mem, size_t offset,
                           const flagcxDevComm *comm);

/* ================================================================
 * Category 5: Utility (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ size_t
flagcxDataTypeSizeDevice(flagcxDataType_t dt);

/* ================================================================
 * Category 6: Intra-Node Barrier Session (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxIntraBarrierSessionInit(flagcxIntraBarrierSession_C *session,
                              flagcxCoopAny coop, const flagcxDevComm *comm,
                              flagcxTeam team, uint32_t index, bool multimem);
FLAGCX_IR_EXTERN_C __device__ void
flagcxIntraBarrierSessionArrive(flagcxIntraBarrierSession_C *session,
                                flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void
flagcxIntraBarrierSessionWait(flagcxIntraBarrierSession_C *session,
                              flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void
flagcxIntraBarrierSessionSync(flagcxIntraBarrierSession_C *session,
                              flagcxDeviceMemoryOrder_t order);

/* ================================================================
 * Category 7: Inter-Node Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void flagcxInterBarrierSessionInit(
    flagcxInterBarrierSession_C *session, flagcxCoopAny coop,
    const flagcxDevTransport *trans, flagcxTeam team, uint32_t index);
FLAGCX_IR_EXTERN_C __device__ void
flagcxInterBarrierSessionSync(flagcxInterBarrierSession_C *session,
                              flagcxDeviceMemoryOrder_t order,
                              flagcxTransportFenceLevel fence);

/* ================================================================
 * Category 8: World Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void flagcxWorldBarrierSessionInit(
    flagcxBarrierSession_C *session, flagcxCoopAny coop, flagcxTeamTagWorld tag,
    const flagcxDevTransport *trans, uint32_t index, bool multimem);
FLAGCX_IR_EXTERN_C __device__ void
flagcxWorldBarrierSessionSync(flagcxBarrierSession_C *session,
                              flagcxDeviceMemoryOrder_t order,
                              flagcxTransportFenceLevel fence);

/* ================================================================
 * Category 9: Transport — Init / Signal Read / Wait / Counter / Flush (7)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxDevTransportInitC(flagcxDevTransport *trans, const flagcxDevComm *comm,
                        int idx);
FLAGCX_IR_EXTERN_C __device__ uint64_t flagcxDevTransportReadSignal(
    const flagcxDevTransport *trans, flagcxDevTransportSignal_t signalId,
    int bits, flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportWaitSignal(
    const flagcxDevTransport *trans, flagcxCoopAny coop,
    flagcxDevTransportSignal_t signalId, uint64_t least, int bits,
    flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportWaitSignalMeetShadow(
    const flagcxDevTransport *trans, flagcxCoopAny coop,
    flagcxDevTransportSignal_t signalId, int bits,
    flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ uint64_t flagcxDevTransportReadCounter(
    const flagcxDevTransport *trans, flagcxDevTransportCounter_t counterId,
    int bits, flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportWaitCounter(
    const flagcxDevTransport *trans, flagcxCoopAny coop,
    flagcxDevTransportCounter_t counterId, uint64_t least, int bits,
    flagcxDeviceMemoryOrder_t order);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevTransportFlush(const flagcxDevTransport *trans, flagcxCoopAny coop,
                        flagcxDeviceMemoryOrder_t order);

/* ================================================================
 * Category 10: Transport — Two-Sided (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ int
flagcxDevTransportSend(const flagcxDevTransport *trans, flagcxCoopAny coop,
                       const flagcxDevMem *mem, size_t offset, size_t count,
                       flagcxDataType_t datatype, int peer);
FLAGCX_IR_EXTERN_C __device__ int
flagcxDevTransportRecv(const flagcxDevTransport *trans, flagcxCoopAny coop,
                       const flagcxDevMem *mem, size_t offset, size_t count,
                       flagcxDataType_t datatype, int peer);
FLAGCX_IR_EXTERN_C __device__ int
flagcxDevTransportWait(const flagcxDevTransport *trans, flagcxCoopAny coop);
FLAGCX_IR_EXTERN_C __device__ int
flagcxDevTransportTerm(const flagcxDevTransport *trans, flagcxCoopAny coop);

/* ================================================================
 * Category 11: Transport — One-Sided put (16)
 *
 * Naming: flagcxDevTransportPut[_R<remote>][_L<local>]
 * Actions: None, SigInc, SigAdd, CtrInc
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevTransportPut(const flagcxDevTransport *trans, flagcxTeam team,
                      int peer, const flagcxDevMem *dst, size_t dstOffset,
                      const flagcxDevMem *src, size_t srcOffset, size_t bytes,
                      flagcxCoopAny coop);

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RSigInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal);

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RSigAdd(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal, uint64_t remoteValue);

/* (CtrInc, None) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RCtrInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportCounter_t remoteCounter);

/* (None, SigInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_LSigInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t localSignal);

/* (SigInc, SigInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RSigInc_LSigInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal,
    flagcxDevTransportSignal_t localSignal);

/* (SigAdd, SigInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RSigAdd_LSigInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevTransportSignal_t localSignal);

/* (CtrInc, SigInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RCtrInc_LSigInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportCounter_t remoteCounter,
    flagcxDevTransportSignal_t localSignal);

/* (None, SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_LSigAdd(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t localSignal, uint64_t localValue);

/* (SigInc, SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RSigInc_LSigAdd(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal,
    flagcxDevTransportSignal_t localSignal, uint64_t localValue);

/* (SigAdd, SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RSigAdd_LSigAdd(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevTransportSignal_t localSignal, uint64_t localValue);

/* (CtrInc, SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RCtrInc_LSigAdd(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportCounter_t remoteCounter,
    flagcxDevTransportSignal_t localSignal, uint64_t localValue);

/* (None, CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_LCtrInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportCounter_t localCounter);

/* (SigInc, CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RSigInc_LCtrInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal,
    flagcxDevTransportCounter_t localCounter);

/* (SigAdd, CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RSigAdd_LCtrInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevTransportCounter_t localCounter);

/* (CtrInc, CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPut_RCtrInc_LCtrInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportCounter_t remoteCounter,
    flagcxDevTransportCounter_t localCounter);

/* ================================================================
 * Category 12: Transport — One-Sided signal (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxDevTransportSignalSigInc(const flagcxDevTransport *trans, flagcxTeam team,
                               int peer, flagcxCoopAny coop,
                               flagcxDevTransportSignal_t signal);
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportSignalSigAdd(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    flagcxCoopAny coop, flagcxDevTransportSignal_t signal, uint64_t value);
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevTransportSignalCtrInc(const flagcxDevTransport *trans, flagcxTeam team,
                               int peer, flagcxCoopAny coop,
                               flagcxDevTransportCounter_t counter);

/* ================================================================
 * Category 13: Transport — One-Sided putValue<uint64_t> (4)
 *
 * C++ putValue only supports RemoteAction (no LocalAction).
 * ================================================================ */

/* (None) */
FLAGCX_IR_EXTERN_C __device__ void
flagcxDevTransportPutValue(const flagcxDevTransport *trans, flagcxTeam team,
                           int peer, const flagcxDevMem *dst, size_t dstOffset,
                           uint64_t value, flagcxCoopAny coop);

/* (SigInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPutValue_RSigInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, uint64_t value,
    flagcxCoopAny coop, flagcxDevTransportSignal_t remoteSignal);

/* (SigAdd) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPutValue_RSigAdd(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, uint64_t value,
    flagcxCoopAny coop, flagcxDevTransportSignal_t remoteSignal,
    uint64_t remoteAddValue);

/* (CtrInc) */
FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportPutValue_RCtrInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, uint64_t value,
    flagcxCoopAny coop, flagcxDevTransportCounter_t remoteCounter);

/* ================================================================
 * Category 14: Transport — One-Sided get (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C __device__ void
flagcxDevTransportGet(const flagcxDevTransport *trans, flagcxTeam team,
                      int peer, const flagcxDevMem *src, size_t srcOffset,
                      const flagcxDevMem *dst, size_t dstOffset, size_t bytes,
                      flagcxCoopAny coop);

/* ================================================================
 * Category 15: Transport — Typed Load / Store (9×2 = 18, X-macro)
 * ================================================================ */

#define FLAGCX_DECL_TRANSPORT_LOAD(SUFFIX, TYPE)                               \
  FLAGCX_IR_EXTERN_C __device__ TYPE flagcxDevTransportLoad##SUFFIX(           \
      const flagcxDevTransport *trans, const flagcxDevMem *mem,                \
      size_t byteOffset, int peer);

#define FLAGCX_DECL_TRANSPORT_STORE(SUFFIX, TYPE)                              \
  FLAGCX_IR_EXTERN_C __device__ void flagcxDevTransportStore##SUFFIX(          \
      const flagcxDevTransport *trans, const flagcxDevMem *mem,                \
      size_t byteOffset, int peer, TYPE value);

FLAGCX_REPT_FOR_DEVICE_TYPES(FLAGCX_DECL_TRANSPORT_LOAD)
FLAGCX_REPT_FOR_DEVICE_TYPES(FLAGCX_DECL_TRANSPORT_STORE)
#undef FLAGCX_DECL_TRANSPORT_LOAD
#undef FLAGCX_DECL_TRANSPORT_STORE

#endif /* FLAGCX_DEVICE_WRAPPER_H_ */
