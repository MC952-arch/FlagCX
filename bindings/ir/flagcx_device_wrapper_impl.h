/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API — C wrapper implementations for LLVM IR generation.
 *
 * This file is compiled by clang with -emit-llvm to produce LLVM bitcode.
 * All functions use FLAGCX_IR_EXTERN_C (= extern "C" under
 * __clang_llvm_bitcode_lib__) to ensure stable, unmangled symbol names.
 ************************************************************************/
#ifndef FLAGCX_DEVICE_WRAPPER_IMPL_H_
#define FLAGCX_DEVICE_WRAPPER_IMPL_H_

#include "flagcx_device_wrapper.h"
#include <new>

#if FLAGCX_CHECK_DEVICE_CC

/* ================================================================
 * Category 1: Comm Queries (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetRank(const void *comm_opaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  return comm->getRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetSize(const void *comm_opaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  return comm->getSize();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetIntraRank(const void *comm_opaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  return comm->getIntraRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetIntraSize(const void *comm_opaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  return comm->getIntraSize();
}

/* ================================================================
 * Category 2: Cooperative Group — Init / Query / Sync (8)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitBlock(void *coop_opaque) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coop_opaque;
  ::new (coop) flagcxCoopAny(flagcxCoopBlock());
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitWarp(void *coop_opaque) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coop_opaque;
  ::new (coop) flagcxCoopAny(flagcxCoopWarp());
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitThread(void *coop_opaque) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coop_opaque;
  ::new (coop) flagcxCoopAny(flagcxCoopThread());
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitTileSpan(void *coop_opaque, int t0, int nTiles, int id) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coop_opaque;
  ::new (coop) flagcxCoopAny(flagcxCoopTileSpan(t0, nTiles, id));
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitLanes(void *coop_opaque, uint32_t laneMask) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coop_opaque;
  ::new (coop) flagcxCoopAny(flagcxCoopLanes(laneMask));
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopThreadRankC(const void *coop_opaque) {
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  return coop->threadRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopSizeC(const void *coop_opaque) {
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  return coop->size();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopSyncC(void *coop_opaque) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coop_opaque;
  coop->sync();
}

/* ================================================================
 * Category 3: Team Functions (5)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxGetTeamIntra(const void *comm_opaque, void *out_opaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  flagcxTeam *out = (flagcxTeam *)out_opaque;
  *out = flagcxTeamIntra(*comm);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxGetTeamWorld(const void *comm_opaque, void *out_opaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  flagcxTeam *out = (flagcxTeam *)out_opaque;
  *out = flagcxTeamWorld(*comm);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxGetTeamInter(const void *comm_opaque, void *out_opaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  flagcxTeam *out = (flagcxTeam *)out_opaque;
  *out = flagcxTeamInter(*comm);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToWorldC(const void *comm_opaque, const void *team_opaque,
                       int rank) {
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  return flagcxTeamRankToWorld(*comm, *team, rank);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToIntraC(const void *comm_opaque, const void *team_opaque,
                       int rank) {
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  return flagcxTeamRankToIntra(*comm, *team, rank);
}

/* ================================================================
 * Category 4: Pointer Access (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointerTeam(const void *mem_opaque, size_t offset,
                         const void *team_opaque, int peer) {
  const flagcxDevMem *mem = (const flagcxDevMem *)mem_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  return flagcxGetPeerPointer(*mem, offset, *team, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointerC(const void *mem_opaque, size_t offset) {
  const flagcxDevMem *mem = (const flagcxDevMem *)mem_opaque;
  return flagcxGetLocalPointer(*mem, offset);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetIntraPointerC(const void *mem_opaque, size_t offset, int peer) {
  const flagcxDevMem *mem = (const flagcxDevMem *)mem_opaque;
  return flagcxGetIntraPointer(*mem, offset, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointerC(const void *mem_opaque, size_t offset,
                           const void *comm_opaque) {
  const flagcxDevMem *mem = (const flagcxDevMem *)mem_opaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  return flagcxGetMulticastPointer(*mem, offset, *comm);
}

/* ================================================================
 * Category 5: Utility (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR size_t
flagcxDataTypeSizeDevice(flagcxDataType_t dt) {
  switch (dt) {
    case flagcxChar:
      return sizeof(char);
    case flagcxUint8:
      return sizeof(unsigned char);
    case flagcxInt:
      return sizeof(int);
    case flagcxUint32:
      return sizeof(unsigned int);
    case flagcxInt64:
      return sizeof(long long);
    case flagcxUint64:
      return sizeof(unsigned long long);
    case flagcxHalf:
      return 2;
    case flagcxFloat:
      return sizeof(float);
    case flagcxDouble:
      return sizeof(double);
    case flagcxBfloat16:
      return 2;
    default:
      return 0;
  }
}

/* ================================================================
 * Category 6: Intra-Node Barrier Session (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionInit(void *session_opaque, const void *coop_opaque,
                              const void *comm_opaque, const void *team_opaque,
                              uint32_t index, bool multimem) {
  flagcxIntraBarrierSession_C *session =
      (flagcxIntraBarrierSession_C *)session_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  ::new (&(session->bar)) flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopAny>(
      *coop, *comm, *team, index, multimem);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionArrive(void *session_opaque,
                                flagcxDeviceMemoryOrder_t order) {
  flagcxIntraBarrierSession_C *session =
      (flagcxIntraBarrierSession_C *)session_opaque;
  session->bar.arrive(order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionWait(void *session_opaque,
                              flagcxDeviceMemoryOrder_t order) {
  flagcxIntraBarrierSession_C *session =
      (flagcxIntraBarrierSession_C *)session_opaque;
  session->bar.wait(order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionSync(void *session_opaque,
                              flagcxDeviceMemoryOrder_t order) {
  flagcxIntraBarrierSession_C *session =
      (flagcxIntraBarrierSession_C *)session_opaque;
  session->bar.sync(order);
}

/* ================================================================
 * Category 7: Inter-Node Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxInterBarrierSessionInit(void *session_opaque, const void *coop_opaque,
                              const void *trans_opaque, const void *team_opaque,
                              uint32_t index) {
  flagcxInterBarrierSession_C *session =
      (flagcxInterBarrierSession_C *)session_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  ::new (&(session->bar)) flagcxDevBarrier<flagcxTeamTagInter, flagcxCoopAny>(
      *coop, *trans, *team, index);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxInterBarrierSessionSync(void *session_opaque,
                              flagcxDeviceMemoryOrder_t order,
                              flagcxDevNetFenceLevel fence) {
  flagcxInterBarrierSession_C *session =
      (flagcxInterBarrierSession_C *)session_opaque;
  session->bar.sync(order, fence);
}

/* ================================================================
 * Category 8: World Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxWorldBarrierSessionInit(void *session_opaque, const void *coop_opaque,
                              flagcxTeamTagWorld tag, const void *trans_opaque,
                              uint32_t index, bool multimem) {
  flagcxBarrierSession_C *session = (flagcxBarrierSession_C *)session_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  ::new (&(session->bar)) flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopAny>(
      *coop, tag, *trans, index, multimem);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxWorldBarrierSessionSync(void *session_opaque,
                              flagcxDeviceMemoryOrder_t order,
                              flagcxDevNetFenceLevel fence) {
  flagcxBarrierSession_C *session = (flagcxBarrierSession_C *)session_opaque;
  session->bar.sync(order, fence);
}

/* ================================================================
 * Category 9: Transport — Init / Signal Read / Wait / Counter / Flush (7)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetInitC(void *trans_opaque, const void *comm_opaque, int idx) {
  flagcxDevNet *trans = (flagcxDevNet *)trans_opaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)comm_opaque;
  ::new (trans) flagcxDevNet(*comm, idx);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevNetReadSignal(const void *trans_opaque, flagcxDevNetSignal_t signalId,
                       int bits, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  return trans->readSignal(signalId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitSignal(const void *trans_opaque, const void *coop_opaque,
                       flagcxDevNetSignal_t signalId, uint64_t least, int bits,
                       flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->waitSignal(*coop, signalId, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitSignalMeetShadow(const void *trans_opaque,
                                 const void *coop_opaque,
                                 flagcxDevNetSignal_t signalId, int bits,
                                 flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->waitSignalMeetShadow(*coop, signalId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevNetReadCounter(const void *trans_opaque,
                        flagcxDevNetCounter_t counterId, int bits,
                        flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  return trans->readCounter(counterId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitCounter(const void *trans_opaque, const void *coop_opaque,
                        flagcxDevNetCounter_t counterId, uint64_t least,
                        int bits, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->waitCounter(*coop, counterId, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetFlush(const void *trans_opaque, const void *coop_opaque,
                  flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->flush(*coop, order);
}

/* ================================================================
 * Category 9b: Net — Reset / Shadow (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetResetSignal(const void *net_opaque, flagcxDevNetSignal_t slot) {
  const flagcxDevNet *net = (const flagcxDevNet *)net_opaque;
  net->resetSignal(slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetResetCounter(const void *net_opaque, flagcxDevNetCounter_t slot) {
  const flagcxDevNet *net = (const flagcxDevNet *)net_opaque;
  net->resetCounter(slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetIncreaseSignalShadow(const void *net_opaque,
                                 flagcxDevNetSignal_t slot, uint64_t delta) {
  const flagcxDevNet *net = (const flagcxDevNet *)net_opaque;
  net->increaseSignalShadow(slot, delta);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitSignalFollowShadow(const void *net_opaque,
                                   const void *coop_opaque,
                                   flagcxDevNetSignal_t slot,
                                   uint64_t leastDelta, uint64_t *before,
                                   uint64_t *delta, int bits,
                                   flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *net = (const flagcxDevNet *)net_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  net->waitSignalFollowShadow(*coop, slot, leastDelta, before, delta, bits,
                              order);
}

/* ================================================================
 * Category 10: Transport — Two-Sided (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetSend(const void *trans_opaque, const void *coop_opaque,
                 const void *mem_opaque, size_t offset, size_t count,
                 flagcxDataType_t datatype, int peer) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  const flagcxDevMem *mem = (const flagcxDevMem *)mem_opaque;
  return (int)trans->send(*coop, *mem, offset, count, datatype, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetRecv(const void *trans_opaque, const void *coop_opaque,
                 const void *mem_opaque, size_t offset, size_t count,
                 flagcxDataType_t datatype, int peer) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  const flagcxDevMem *mem = (const flagcxDevMem *)mem_opaque;
  return (int)trans->recv(*coop, *mem, offset, count, datatype, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetWait(const void *trans_opaque, const void *coop_opaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  return (int)trans->wait(*coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetTerm(const void *trans_opaque, const void *coop_opaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  return (int)trans->term(*coop);
}

/* ================================================================
 * Category 11: Transport — One-Sided put (16)
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut(const void *trans_opaque, const void *team_opaque, int peer,
                const void *dst_opaque, size_t dstOffset,
                const void *src_opaque, size_t srcOffset, size_t bytes,
                const void *coop_opaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_None{}, flagcxDevNet_None{}, *coop);
}

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigInc(const void *trans_opaque, const void *team_opaque,
                        int peer, const void *dst_opaque, size_t dstOffset,
                        const void *src_opaque, size_t srcOffset, size_t bytes,
                        const void *coop_opaque,
                        flagcxDevNetSignal_t remoteSignal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalInc{remoteSignal}, flagcxDevNet_None{}, *coop);
}

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevNetPut_RSigAdd(
    const void *trans_opaque, const void *team_opaque, int peer,
    const void *dst_opaque, size_t dstOffset, const void *src_opaque,
    size_t srcOffset, size_t bytes, const void *coop_opaque,
    flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
             flagcxDevNet_None{}, *coop);
}

/* (CtrInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RCtrInc(const void *trans_opaque, const void *team_opaque,
                        int peer, const void *dst_opaque, size_t dstOffset,
                        const void *src_opaque, size_t srcOffset, size_t bytes,
                        const void *coop_opaque,
                        flagcxDevNetCounter_t remoteCounter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_CounterInc{remoteCounter}, flagcxDevNet_None{},
             *coop);
}

/* (None, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_LSigInc(const void *trans_opaque, const void *team_opaque,
                        int peer, const void *dst_opaque, size_t dstOffset,
                        const void *src_opaque, size_t srcOffset, size_t bytes,
                        const void *coop_opaque,
                        flagcxDevNetSignal_t localSignal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_None{}, flagcxDevNet_SignalInc{localSignal}, *coop);
}

/* (SigInc, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigInc_LSigInc(const void *trans_opaque,
                                const void *team_opaque, int peer,
                                const void *dst_opaque, size_t dstOffset,
                                const void *src_opaque, size_t srcOffset,
                                size_t bytes, const void *coop_opaque,
                                flagcxDevNetSignal_t remoteSignal,
                                flagcxDevNetSignal_t localSignal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalInc{remoteSignal},
             flagcxDevNet_SignalInc{localSignal}, *coop);
}

/* (SigAdd, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigAdd_LSigInc(const void *trans_opaque,
                                const void *team_opaque, int peer,
                                const void *dst_opaque, size_t dstOffset,
                                const void *src_opaque, size_t srcOffset,
                                size_t bytes, const void *coop_opaque,
                                flagcxDevNetSignal_t remoteSignal,
                                uint64_t remoteValue,
                                flagcxDevNetSignal_t localSignal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
             flagcxDevNet_SignalInc{localSignal}, *coop);
}

/* (CtrInc, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RCtrInc_LSigInc(const void *trans_opaque,
                                const void *team_opaque, int peer,
                                const void *dst_opaque, size_t dstOffset,
                                const void *src_opaque, size_t srcOffset,
                                size_t bytes, const void *coop_opaque,
                                flagcxDevNetCounter_t remoteCounter,
                                flagcxDevNetSignal_t localSignal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_CounterInc{remoteCounter},
             flagcxDevNet_SignalInc{localSignal}, *coop);
}

/* (None, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_LSigAdd(const void *trans_opaque, const void *team_opaque,
                        int peer, const void *dst_opaque, size_t dstOffset,
                        const void *src_opaque, size_t srcOffset, size_t bytes,
                        const void *coop_opaque,
                        flagcxDevNetSignal_t localSignal, uint64_t localValue) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_None{},
             flagcxDevNet_SignalAdd{localSignal, localValue}, *coop);
}

/* (SigInc, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigInc_LSigAdd(const void *trans_opaque,
                                const void *team_opaque, int peer,
                                const void *dst_opaque, size_t dstOffset,
                                const void *src_opaque, size_t srcOffset,
                                size_t bytes, const void *coop_opaque,
                                flagcxDevNetSignal_t remoteSignal,
                                flagcxDevNetSignal_t localSignal,
                                uint64_t localValue) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalInc{remoteSignal},
             flagcxDevNet_SignalAdd{localSignal, localValue}, *coop);
}

/* (SigAdd, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigAdd_LSigAdd(
    const void *trans_opaque, const void *team_opaque, int peer,
    const void *dst_opaque, size_t dstOffset, const void *src_opaque,
    size_t srcOffset, size_t bytes, const void *coop_opaque,
    flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevNetSignal_t localSignal, uint64_t localValue) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
             flagcxDevNet_SignalAdd{localSignal, localValue}, *coop);
}

/* (CtrInc, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RCtrInc_LSigAdd(const void *trans_opaque,
                                const void *team_opaque, int peer,
                                const void *dst_opaque, size_t dstOffset,
                                const void *src_opaque, size_t srcOffset,
                                size_t bytes, const void *coop_opaque,
                                flagcxDevNetCounter_t remoteCounter,
                                flagcxDevNetSignal_t localSignal,
                                uint64_t localValue) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_CounterInc{remoteCounter},
             flagcxDevNet_SignalAdd{localSignal, localValue}, *coop);
}

/* (None, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_LCtrInc(const void *trans_opaque, const void *team_opaque,
                        int peer, const void *dst_opaque, size_t dstOffset,
                        const void *src_opaque, size_t srcOffset, size_t bytes,
                        const void *coop_opaque,
                        flagcxDevNetCounter_t localCounter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_None{}, flagcxDevNet_CounterInc{localCounter}, *coop);
}

/* (SigInc, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigInc_LCtrInc(const void *trans_opaque,
                                const void *team_opaque, int peer,
                                const void *dst_opaque, size_t dstOffset,
                                const void *src_opaque, size_t srcOffset,
                                size_t bytes, const void *coop_opaque,
                                flagcxDevNetSignal_t remoteSignal,
                                flagcxDevNetCounter_t localCounter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalInc{remoteSignal},
             flagcxDevNet_CounterInc{localCounter}, *coop);
}

/* (SigAdd, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigAdd_LCtrInc(const void *trans_opaque,
                                const void *team_opaque, int peer,
                                const void *dst_opaque, size_t dstOffset,
                                const void *src_opaque, size_t srcOffset,
                                size_t bytes, const void *coop_opaque,
                                flagcxDevNetSignal_t remoteSignal,
                                uint64_t remoteValue,
                                flagcxDevNetCounter_t localCounter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
             flagcxDevNet_CounterInc{localCounter}, *coop);
}

/* (CtrInc, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RCtrInc_LCtrInc(const void *trans_opaque,
                                const void *team_opaque, int peer,
                                const void *dst_opaque, size_t dstOffset,
                                const void *src_opaque, size_t srcOffset,
                                size_t bytes, const void *coop_opaque,
                                flagcxDevNetCounter_t remoteCounter,
                                flagcxDevNetCounter_t localCounter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_CounterInc{remoteCounter},
             flagcxDevNet_CounterInc{localCounter}, *coop);
}

/* ================================================================
 * Category 12: Transport — One-Sided signal (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalSigInc(const void *trans_opaque, const void *team_opaque,
                         int peer, const void *coop_opaque,
                         flagcxDevNetSignal_t signal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->signal(*team, peer, flagcxDevNet_SignalInc{signal}, *coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalSigAdd(const void *trans_opaque, const void *team_opaque,
                         int peer, const void *coop_opaque,
                         flagcxDevNetSignal_t signal, uint64_t value) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->signal(*team, peer, flagcxDevNet_SignalAdd{signal, value}, *coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalCtrInc(const void *trans_opaque, const void *team_opaque,
                         int peer, const void *coop_opaque,
                         flagcxDevNetCounter_t counter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->signal(*team, peer, flagcxDevNet_CounterInc{counter}, *coop);
}

/* ================================================================
 * Category 13: Transport — One-Sided putValue<uint64_t> (16)
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValue(const void *trans_opaque, const void *team_opaque,
                     int peer, const void *dst_opaque, size_t dstOffset,
                     uint64_t value, const void *coop_opaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->putValue(*team, peer, *dst, dstOffset, value, flagcxDevNet_None{},
                  *coop);
}

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValue_RSigInc(const void *trans_opaque, const void *team_opaque,
                             int peer, const void *dst_opaque, size_t dstOffset,
                             uint64_t value, const void *coop_opaque,
                             flagcxDevNetSignal_t remoteSignal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->putValue(*team, peer, *dst, dstOffset, value,
                  flagcxDevNet_SignalInc{remoteSignal}, *coop);
}

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValue_RSigAdd(const void *trans_opaque, const void *team_opaque,
                             int peer, const void *dst_opaque, size_t dstOffset,
                             uint64_t value, const void *coop_opaque,
                             flagcxDevNetSignal_t remoteSignal,
                             uint64_t remoteAddValue) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->putValue(*team, peer, *dst, dstOffset, value,
                  flagcxDevNet_SignalAdd{remoteSignal, remoteAddValue}, *coop);
}

/* (CtrInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValue_RCtrInc(const void *trans_opaque, const void *team_opaque,
                             int peer, const void *dst_opaque, size_t dstOffset,
                             uint64_t value, const void *coop_opaque,
                             flagcxDevNetCounter_t remoteCounter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->putValue(*team, peer, *dst, dstOffset, value,
                  flagcxDevNet_CounterInc{remoteCounter}, *coop);
}

/* ================================================================
 * Category 14: Transport — One-Sided get (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetGet(const void *trans_opaque, const void *team_opaque, int peer,
                const void *src_opaque, size_t srcOffset,
                const void *dst_opaque, size_t dstOffset, size_t bytes,
                const void *coop_opaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)trans_opaque;
  const flagcxTeam *team = (const flagcxTeam *)team_opaque;
  const flagcxDevMem *src = (const flagcxDevMem *)src_opaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dst_opaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coop_opaque;
  trans->get(*team, peer, *src, srcOffset, *dst, dstOffset, bytes, *coop);
}

#endif /* FLAGCX_CHECK_DEVICE_CC */
#endif /* FLAGCX_DEVICE_WRAPPER_IMPL_H_ */
