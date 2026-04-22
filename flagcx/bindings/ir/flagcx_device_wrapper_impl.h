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
flagcxDevCommGetRank(const flagcxDevComm *comm) {
  return comm->getRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetSize(const flagcxDevComm *comm) {
  return comm->getSize();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetIntraRank(const flagcxDevComm *comm) {
  return comm->getIntraRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetIntraSize(const flagcxDevComm *comm) {
  return comm->getIntraSize();
}

/* ================================================================
 * Category 2: Cooperative Group — Init / Query / Sync (8)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitBlock(flagcxCoopAny *coop) {
  ::new (coop) flagcxCoopAny(flagcxCoopBlock());
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitWarp(flagcxCoopAny *coop) {
  ::new (coop) flagcxCoopAny(flagcxCoopWarp());
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitThread(flagcxCoopAny *coop) {
  ::new (coop) flagcxCoopAny(flagcxCoopThread());
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitTileSpan(flagcxCoopAny *coop, int t0, int nTiles, int id) {
  ::new (coop) flagcxCoopAny(flagcxCoopTileSpan(t0, nTiles, id));
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitLanes(flagcxCoopAny *coop, uint32_t laneMask) {
  ::new (coop) flagcxCoopAny(flagcxCoopLanes(laneMask));
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopThreadRankC(const flagcxCoopAny *coop) {
  return coop->threadRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopSizeC(const flagcxCoopAny *coop) {
  return coop->size();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopSyncC(flagcxCoopAny *coop) {
  coop->sync();
}

/* ================================================================
 * Category 3: Team Functions (5)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR flagcxTeam
flagcxGetTeamIntra(const flagcxDevComm *comm) {
  return flagcxTeamIntra(*comm);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR flagcxTeam
flagcxGetTeamWorld(const flagcxDevComm *comm) {
  return flagcxTeamWorld(*comm);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR flagcxTeam
flagcxGetTeamInter(const flagcxDevComm *comm) {
  return flagcxTeamInter(*comm);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToWorldC(const flagcxDevComm *comm, flagcxTeam team, int rank) {
  return flagcxTeamRankToWorld(*comm, team, rank);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToIntraC(const flagcxDevComm *comm, flagcxTeam team, int rank) {
  return flagcxTeamRankToIntra(*comm, team, rank);
}

/* ================================================================
 * Category 4: Pointer Access (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointerTeam(const flagcxDevMem *mem, size_t offset,
                         flagcxTeam team, int peer) {
  return flagcxGetPeerPointer(*mem, offset, team, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointerC(const flagcxDevMem *mem, size_t offset) {
  return flagcxGetLocalPointer(*mem, offset);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetIntraPointerC(const flagcxDevMem *mem, size_t offset, int peer) {
  return flagcxGetIntraPointer(*mem, offset, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointerC(const flagcxDevMem *mem, size_t offset,
                           const flagcxDevComm *comm) {
  return flagcxGetMulticastPointer(*mem, offset, *comm);
}

/* ================================================================
 * Category 5: Utility (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR size_t
flagcxDataTypeSizeDevice(flagcxDataType_t dt) {
  return getFlagcxDataTypeSizeDevice(dt);
}

/* ================================================================
 * Category 6: Intra-Node Barrier Session (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionInit(flagcxIntraBarrierSession_C *session,
                              flagcxCoopAny coop, const flagcxDevComm *comm,
                              flagcxTeam team, uint32_t index, bool multimem) {
  ::new (&(session->bar)) flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopAny>(
      coop, *comm, team, index, multimem);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionArrive(flagcxIntraBarrierSession_C *session,
                                flagcxDeviceMemoryOrder_t order) {
  session->bar.arrive(order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionWait(flagcxIntraBarrierSession_C *session,
                              flagcxDeviceMemoryOrder_t order) {
  session->bar.wait(order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionSync(flagcxIntraBarrierSession_C *session,
                              flagcxDeviceMemoryOrder_t order) {
  session->bar.sync(order);
}

/* ================================================================
 * Category 7: Inter-Node Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxInterBarrierSessionInit(flagcxInterBarrierSession_C *session,
                              flagcxCoopAny coop,
                              const flagcxDevTransport *trans, flagcxTeam team,
                              uint32_t index) {
  ::new (&(session->bar)) flagcxDevBarrier<flagcxTeamTagInter, flagcxCoopAny>(
      coop, *trans, team, index);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxInterBarrierSessionSync(flagcxInterBarrierSession_C *session,
                              flagcxDeviceMemoryOrder_t order,
                              flagcxTransportFenceLevel fence) {
  session->bar.sync(order, fence);
}

/* ================================================================
 * Category 8: World Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxWorldBarrierSessionInit(flagcxBarrierSession_C *session,
                              flagcxCoopAny coop, flagcxTeamTagWorld tag,
                              const flagcxDevTransport *trans, uint32_t index,
                              bool multimem) {
  ::new (&(session->bar)) flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopAny>(
      coop, tag, *trans, index, multimem);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxWorldBarrierSessionSync(flagcxBarrierSession_C *session,
                              flagcxDeviceMemoryOrder_t order,
                              flagcxTransportFenceLevel fence) {
  session->bar.sync(order, fence);
}

/* ================================================================
 * Category 9: Transport — Init / Signal Read / Wait / Counter / Flush (7)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportInitC(flagcxDevTransport *trans, const flagcxDevComm *comm,
                        int idx) {
  ::new (trans) flagcxDevTransport(*comm, idx);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevTransportReadSignal(const flagcxDevTransport *trans,
                             flagcxDevTransportSignal_t signalId, int bits,
                             flagcxDeviceMemoryOrder_t order) {
  return trans->readSignal(signalId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportWaitSignal(const flagcxDevTransport *trans,
                             flagcxCoopAny coop,
                             flagcxDevTransportSignal_t signalId,
                             uint64_t least, int bits,
                             flagcxDeviceMemoryOrder_t order) {
  trans->waitSignal(coop, signalId, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportWaitSignalMeetShadow(const flagcxDevTransport *trans,
                                       flagcxCoopAny coop,
                                       flagcxDevTransportSignal_t signalId,
                                       int bits,
                                       flagcxDeviceMemoryOrder_t order) {
  trans->waitSignalMeetShadow(coop, signalId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevTransportReadCounter(const flagcxDevTransport *trans,
                              flagcxDevTransportCounter_t counterId, int bits,
                              flagcxDeviceMemoryOrder_t order) {
  return trans->readCounter(counterId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportWaitCounter(const flagcxDevTransport *trans,
                              flagcxCoopAny coop,
                              flagcxDevTransportCounter_t counterId,
                              uint64_t least, int bits,
                              flagcxDeviceMemoryOrder_t order) {
  trans->waitCounter(coop, counterId, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportFlush(const flagcxDevTransport *trans, flagcxCoopAny coop,
                        flagcxDeviceMemoryOrder_t order) {
  trans->flush(coop, order);
}

/* ================================================================
 * Category 10: Transport — Two-Sided (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevTransportSend(const flagcxDevTransport *trans, flagcxCoopAny coop,
                       const flagcxDevMem *mem, size_t offset, size_t count,
                       flagcxDataType_t datatype, int peer) {
  return (int)trans->send(coop, *mem, offset, count, datatype, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevTransportRecv(const flagcxDevTransport *trans, flagcxCoopAny coop,
                       const flagcxDevMem *mem, size_t offset, size_t count,
                       flagcxDataType_t datatype, int peer) {
  return (int)trans->recv(coop, *mem, offset, count, datatype, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevTransportWait(const flagcxDevTransport *trans, flagcxCoopAny coop) {
  return (int)trans->wait(coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevTransportTerm(const flagcxDevTransport *trans, flagcxCoopAny coop) {
  return (int)trans->term(coop);
}

/* ================================================================
 * Category 11: Transport — One-Sided put (16)
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut(const flagcxDevTransport *trans, flagcxTeam team,
                      int peer, const flagcxDevMem *dst, size_t dstOffset,
                      const flagcxDevMem *src, size_t srcOffset, size_t bytes,
                      flagcxCoopAny coop) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_None{}, flagcxDevTransport_None{}, coop);
}

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RSigInc(const flagcxDevTransport *trans, flagcxTeam team,
                              int peer, const flagcxDevMem *dst,
                              size_t dstOffset, const flagcxDevMem *src,
                              size_t srcOffset, size_t bytes,
                              flagcxCoopAny coop,
                              flagcxDevTransportSignal_t remoteSignal) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_SignalInc{remoteSignal},
             flagcxDevTransport_None{}, coop);
}

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RSigAdd(const flagcxDevTransport *trans, flagcxTeam team,
                              int peer, const flagcxDevMem *dst,
                              size_t dstOffset, const flagcxDevMem *src,
                              size_t srcOffset, size_t bytes,
                              flagcxCoopAny coop,
                              flagcxDevTransportSignal_t remoteSignal,
                              uint64_t remoteValue) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_SignalAdd{remoteSignal, remoteValue},
             flagcxDevTransport_None{}, coop);
}

/* (CtrInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RCtrInc(const flagcxDevTransport *trans, flagcxTeam team,
                              int peer, const flagcxDevMem *dst,
                              size_t dstOffset, const flagcxDevMem *src,
                              size_t srcOffset, size_t bytes,
                              flagcxCoopAny coop,
                              flagcxDevTransportCounter_t remoteCounter) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_CounterInc{remoteCounter},
             flagcxDevTransport_None{}, coop);
}

/* (None, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_LSigInc(const flagcxDevTransport *trans, flagcxTeam team,
                              int peer, const flagcxDevMem *dst,
                              size_t dstOffset, const flagcxDevMem *src,
                              size_t srcOffset, size_t bytes,
                              flagcxCoopAny coop,
                              flagcxDevTransportSignal_t localSignal) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_None{},
             flagcxDevTransport_SignalInc{localSignal}, coop);
}

/* (SigInc, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RSigInc_LSigInc(const flagcxDevTransport *trans,
                                      flagcxTeam team, int peer,
                                      const flagcxDevMem *dst, size_t dstOffset,
                                      const flagcxDevMem *src, size_t srcOffset,
                                      size_t bytes, flagcxCoopAny coop,
                                      flagcxDevTransportSignal_t remoteSignal,
                                      flagcxDevTransportSignal_t localSignal) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_SignalInc{remoteSignal},
             flagcxDevTransport_SignalInc{localSignal}, coop);
}

/* (SigAdd, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RSigAdd_LSigInc(const flagcxDevTransport *trans,
                                      flagcxTeam team, int peer,
                                      const flagcxDevMem *dst, size_t dstOffset,
                                      const flagcxDevMem *src, size_t srcOffset,
                                      size_t bytes, flagcxCoopAny coop,
                                      flagcxDevTransportSignal_t remoteSignal,
                                      uint64_t remoteValue,
                                      flagcxDevTransportSignal_t localSignal) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_SignalAdd{remoteSignal, remoteValue},
             flagcxDevTransport_SignalInc{localSignal}, coop);
}

/* (CtrInc, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RCtrInc_LSigInc(const flagcxDevTransport *trans,
                                      flagcxTeam team, int peer,
                                      const flagcxDevMem *dst, size_t dstOffset,
                                      const flagcxDevMem *src, size_t srcOffset,
                                      size_t bytes, flagcxCoopAny coop,
                                      flagcxDevTransportCounter_t remoteCounter,
                                      flagcxDevTransportSignal_t localSignal) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_CounterInc{remoteCounter},
             flagcxDevTransport_SignalInc{localSignal}, coop);
}

/* (None, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_LSigAdd(const flagcxDevTransport *trans, flagcxTeam team,
                              int peer, const flagcxDevMem *dst,
                              size_t dstOffset, const flagcxDevMem *src,
                              size_t srcOffset, size_t bytes,
                              flagcxCoopAny coop,
                              flagcxDevTransportSignal_t localSignal,
                              uint64_t localValue) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_None{},
             flagcxDevTransport_SignalAdd{localSignal, localValue}, coop);
}

/* (SigInc, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RSigInc_LSigAdd(const flagcxDevTransport *trans,
                                      flagcxTeam team, int peer,
                                      const flagcxDevMem *dst, size_t dstOffset,
                                      const flagcxDevMem *src, size_t srcOffset,
                                      size_t bytes, flagcxCoopAny coop,
                                      flagcxDevTransportSignal_t remoteSignal,
                                      flagcxDevTransportSignal_t localSignal,
                                      uint64_t localValue) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_SignalInc{remoteSignal},
             flagcxDevTransport_SignalAdd{localSignal, localValue}, coop);
}

/* (SigAdd, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RSigAdd_LSigAdd(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevTransportSignal_t localSignal, uint64_t localValue) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_SignalAdd{remoteSignal, remoteValue},
             flagcxDevTransport_SignalAdd{localSignal, localValue}, coop);
}

/* (CtrInc, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RCtrInc_LSigAdd(const flagcxDevTransport *trans,
                                      flagcxTeam team, int peer,
                                      const flagcxDevMem *dst, size_t dstOffset,
                                      const flagcxDevMem *src, size_t srcOffset,
                                      size_t bytes, flagcxCoopAny coop,
                                      flagcxDevTransportCounter_t remoteCounter,
                                      flagcxDevTransportSignal_t localSignal,
                                      uint64_t localValue) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_CounterInc{remoteCounter},
             flagcxDevTransport_SignalAdd{localSignal, localValue}, coop);
}

/* (None, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_LCtrInc(const flagcxDevTransport *trans, flagcxTeam team,
                              int peer, const flagcxDevMem *dst,
                              size_t dstOffset, const flagcxDevMem *src,
                              size_t srcOffset, size_t bytes,
                              flagcxCoopAny coop,
                              flagcxDevTransportCounter_t localCounter) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_None{},
             flagcxDevTransport_CounterInc{localCounter}, coop);
}

/* (SigInc, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RSigInc_LCtrInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal,
    flagcxDevTransportCounter_t localCounter) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_SignalInc{remoteSignal},
             flagcxDevTransport_CounterInc{localCounter}, coop);
}

/* (SigAdd, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RSigAdd_LCtrInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevTransportCounter_t localCounter) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_SignalAdd{remoteSignal, remoteValue},
             flagcxDevTransport_CounterInc{localCounter}, coop);
}

/* (CtrInc, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPut_RCtrInc_LCtrInc(
    const flagcxDevTransport *trans, flagcxTeam team, int peer,
    const flagcxDevMem *dst, size_t dstOffset, const flagcxDevMem *src,
    size_t srcOffset, size_t bytes, flagcxCoopAny coop,
    flagcxDevTransportCounter_t remoteCounter,
    flagcxDevTransportCounter_t localCounter) {
  trans->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevTransport_CounterInc{remoteCounter},
             flagcxDevTransport_CounterInc{localCounter}, coop);
}

/* ================================================================
 * Category 12: Transport — One-Sided signal (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportSignalSigInc(const flagcxDevTransport *trans, flagcxTeam team,
                               int peer, flagcxCoopAny coop,
                               flagcxDevTransportSignal_t signal) {
  trans->signal(team, peer, flagcxDevTransport_SignalInc{signal}, coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportSignalSigAdd(const flagcxDevTransport *trans, flagcxTeam team,
                               int peer, flagcxCoopAny coop,
                               flagcxDevTransportSignal_t signal,
                               uint64_t value) {
  trans->signal(team, peer, flagcxDevTransport_SignalAdd{signal, value}, coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportSignalCtrInc(const flagcxDevTransport *trans, flagcxTeam team,
                               int peer, flagcxCoopAny coop,
                               flagcxDevTransportCounter_t counter) {
  trans->signal(team, peer, flagcxDevTransport_CounterInc{counter}, coop);
}

/* ================================================================
 * Category 13: Transport — One-Sided putValue<uint64_t> (16)
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPutValue(const flagcxDevTransport *trans, flagcxTeam team,
                           int peer, const flagcxDevMem *dst, size_t dstOffset,
                           uint64_t value, flagcxCoopAny coop) {
  trans->putValue(team, peer, *dst, dstOffset, value, flagcxDevTransport_None{},
                  coop);
}

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPutValue_RSigInc(const flagcxDevTransport *trans,
                                   flagcxTeam team, int peer,
                                   const flagcxDevMem *dst, size_t dstOffset,
                                   uint64_t value, flagcxCoopAny coop,
                                   flagcxDevTransportSignal_t remoteSignal) {
  trans->putValue(team, peer, *dst, dstOffset, value,
                  flagcxDevTransport_SignalInc{remoteSignal}, coop);
}

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPutValue_RSigAdd(const flagcxDevTransport *trans,
                                   flagcxTeam team, int peer,
                                   const flagcxDevMem *dst, size_t dstOffset,
                                   uint64_t value, flagcxCoopAny coop,
                                   flagcxDevTransportSignal_t remoteSignal,
                                   uint64_t remoteAddValue) {
  trans->putValue(team, peer, *dst, dstOffset, value,
                  flagcxDevTransport_SignalAdd{remoteSignal, remoteAddValue},
                  coop);
}

/* (CtrInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportPutValue_RCtrInc(const flagcxDevTransport *trans,
                                   flagcxTeam team, int peer,
                                   const flagcxDevMem *dst, size_t dstOffset,
                                   uint64_t value, flagcxCoopAny coop,
                                   flagcxDevTransportCounter_t remoteCounter) {
  trans->putValue(team, peer, *dst, dstOffset, value,
                  flagcxDevTransport_CounterInc{remoteCounter}, coop);
}

/* ================================================================
 * Category 14: Transport — One-Sided get (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevTransportGet(const flagcxDevTransport *trans, flagcxTeam team,
                      int peer, const flagcxDevMem *src, size_t srcOffset,
                      const flagcxDevMem *dst, size_t dstOffset, size_t bytes,
                      flagcxCoopAny coop) {
  trans->get(team, peer, *src, srcOffset, *dst, dstOffset, bytes, coop);
}

/* ================================================================
 * Category 15: Transport — Typed Load / Store (9×2 = 18, X-macro)
 * ================================================================ */

#define FLAGCX_IMPL_TRANSPORT_LOAD(SUFFIX, TYPE)                               \
  FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR TYPE                       \
      flagcxDevTransportLoad##SUFFIX(const flagcxDevTransport *trans,          \
                                     const flagcxDevMem *mem,                  \
                                     size_t byteOffset, int peer) {            \
    return trans->load<TYPE>(*mem, byteOffset, peer);                          \
  }

#define FLAGCX_IMPL_TRANSPORT_STORE(SUFFIX, TYPE)                              \
  FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void                       \
      flagcxDevTransportStore##SUFFIX(                                         \
          const flagcxDevTransport *trans, const flagcxDevMem *mem,            \
          size_t byteOffset, int peer, TYPE value) {                           \
    trans->store<TYPE>(*mem, byteOffset, peer, value);                         \
  }

FLAGCX_REPT_FOR_DEVICE_TYPES(FLAGCX_IMPL_TRANSPORT_LOAD)
FLAGCX_REPT_FOR_DEVICE_TYPES(FLAGCX_IMPL_TRANSPORT_STORE)
#undef FLAGCX_IMPL_TRANSPORT_LOAD
#undef FLAGCX_IMPL_TRANSPORT_STORE

#endif /* FLAGCX_CHECK_DEVICE_CC */
#endif /* FLAGCX_DEVICE_WRAPPER_IMPL_H_ */
