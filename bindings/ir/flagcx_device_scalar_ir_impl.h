/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device Scalar IR API — Implementation (Step 2: Coop + Team).
 *
 * This file implements the scalar cooperative group and team functions.
 * It is included by the bitcode compilation unit alongside the existing
 * flagcx_device_wrapper_impl.h.
 *
 * Remaining categories (barrier, net) will be added in subsequent steps.
 ************************************************************************/
#ifndef FLAGCX_DEVICE_SCALAR_IR_IMPL_H_
#define FLAGCX_DEVICE_SCALAR_IR_IMPL_H_

#include "flagcx_device_scalar_ir.h"

// flagcx_device_core.h is already included by flagcx_device_wrapper_impl.h
// which includes us. We rely on its types (flagcxDevComm, flagcxCoopAny, etc.).

/* ================================================================
 * Internal helper: construct flagcxCoopAny from kind enum
 *
 * Used by all scalar functions that need a cooperative group.
 * Constructs the type-erased CoopAny on the stack transiently.
 * ================================================================ */

static FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny
flagcxMakeCoopFromKind(flagcxCoopKind_t kind) {
  switch (kind) {
    case FLAGCX_COOP_BLOCK:
      return flagcxCoopAny(flagcxCoopBlock());
    case FLAGCX_COOP_WARP:
      return flagcxCoopAny(flagcxCoopWarp());
    case FLAGCX_COOP_THREAD:
      return flagcxCoopAny(flagcxCoopThread());
    default:
      return flagcxCoopAny(flagcxCoopBlock());
  }
}

static FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny flagcxMakeCoopFromKindEx(
    flagcxCoopKind_t kind, uint32_t param0, uint32_t param1, uint32_t param2) {
  switch (kind) {
    case FLAGCX_COOP_BLOCK:
      return flagcxCoopAny(flagcxCoopBlock());
    case FLAGCX_COOP_WARP:
      return flagcxCoopAny(flagcxCoopWarp());
    case FLAGCX_COOP_THREAD:
      return flagcxCoopAny(flagcxCoopThread());
    case FLAGCX_COOP_TILE_SPAN:
      return flagcxCoopAny(
          flagcxCoopTileSpan((int)param0, (int)param1, (int)param2));
    case FLAGCX_COOP_LANES:
      return flagcxCoopAny(flagcxCoopLanes(param0));
    default:
      return flagcxCoopAny(flagcxCoopBlock());
  }
}

/* ================================================================
 * Internal helper: construct flagcxTeam from kind enum + comm
 * ================================================================ */

static FLAGCX_DEVICE_INLINE_DECORATOR flagcxTeam
flagcxMakeTeamFromKind(const flagcxDevComm &comm, flagcxTeamKind_t kind) {
  switch (kind) {
    case FLAGCX_TEAM_INTRA:
      return flagcxTeamIntra(comm);
    case FLAGCX_TEAM_INTER:
      return flagcxTeamInter(comm);
    case FLAGCX_TEAM_WORLD:
      return flagcxTeamWorld(comm);
    default:
      return flagcxTeamIntra(comm);
  }
}

/* ================================================================
 * Category 2: Scalar Cooperative Group (6)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopThreadRankS(flagcxCoopKind_t kind) {
  flagcxCoopAny coop = flagcxMakeCoopFromKind(kind);
  return coop.threadRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopSizeS(flagcxCoopKind_t kind) {
  flagcxCoopAny coop = flagcxMakeCoopFromKind(kind);
  return coop.size();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopSyncS(flagcxCoopKind_t kind) {
  flagcxCoopAny coop = flagcxMakeCoopFromKind(kind);
  coop.sync();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopThreadRankExS(flagcxCoopKind_t kind, uint32_t param0, uint32_t param1,
                        uint32_t param2) {
  flagcxCoopAny coop = flagcxMakeCoopFromKindEx(kind, param0, param1, param2);
  return coop.threadRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopSizeExS(flagcxCoopKind_t kind, uint32_t param0, uint32_t param1,
                  uint32_t param2) {
  flagcxCoopAny coop = flagcxMakeCoopFromKindEx(kind, param0, param1, param2);
  return coop.size();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopSyncExS(flagcxCoopKind_t kind, uint32_t param0, uint32_t param1,
                  uint32_t param2) {
  flagcxCoopAny coop = flagcxMakeCoopFromKindEx(kind, param0, param1, param2);
  coop.sync();
}

/* ================================================================
 * Category 3: Scalar Team (2)
 *
 * Suffix 'S' = scalar-style (enum-based), vs 'C' = C-wrapper style
 * (opaque struct pointer) in flagcx_device_wrapper_impl.h.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToWorldS(const void *commOpaque, flagcxTeamKind_t teamKind,
                       int rank) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  return flagcxTeamRankToWorld(*comm, team, rank);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToIntraS(const void *commOpaque, flagcxTeamKind_t teamKind,
                       int rank) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  return flagcxTeamRankToIntra(*comm, team, rank);
}

/* ================================================================
 * Category 4: Pointer Access (scalar team) (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointerS(const void *memOpaque, size_t offset,
                      const void *commOpaque, flagcxTeamKind_t teamKind,
                      int peer) {
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  return flagcxGetPeerPointer(*mem, offset, team, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointerS(const void *memOpaque, size_t offset) {
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  return flagcxGetLocalPointer(*mem, offset);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetIntraPointerS(const void *memOpaque, size_t offset, int peer) {
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  return flagcxGetIntraPointer(*mem, offset, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointerS(const void *memOpaque, size_t offset,
                           const void *commOpaque) {
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return flagcxGetMulticastPointer(*mem, offset, *comm);
}

/* ================================================================
 * Internal helper: construct inter team from Comm inside DevNet
 *
 * flagcxTeamInter(const flagcxDevComm &) can't be called with a raw
 * DeviceAPI::Comm. We replicate the arithmetic using accessors that
 * work on both default and nvidia Comm types.
 * ================================================================ */

static FLAGCX_DEVICE_INLINE_DECORATOR flagcxTeam
flagcxMakeInterTeamFromNet(const flagcxDevNet &net) {
  flagcxTeam team;
  team._teamBase.nRanks = net._dc.getSize() / net._dc.getIntraSize();
  team._teamBase.rank = net._dc.getRank() / net._dc.getIntraSize();
  team._teamBase.stride = net._dc.getIntraSize();
  return team;
}

/* ================================================================
 * Category 6: Scalar Barrier — Intra (3)
 *
 * Construct barrier on the stack, call arrive/wait/sync once.
 * Team is always flagcxTeamIntra since it's an intra-node barrier.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierArriveS(const void *commOpaque, flagcxCoopKind_t coopKind,
                          uint32_t index, bool multimem,
                          flagcxDeviceMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxTeam team = flagcxTeamIntra(*comm);
  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopAny> bar(coop, *comm, team,
                                                          index, multimem);
  bar.arrive(order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierWaitS(const void *commOpaque, flagcxCoopKind_t coopKind,
                        uint32_t index, bool multimem,
                        flagcxDeviceMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxTeam team = flagcxTeamIntra(*comm);
  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopAny> bar(coop, *comm, team,
                                                          index, multimem);
  bar.wait(order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSyncS(const void *commOpaque, flagcxCoopKind_t coopKind,
                        uint32_t index, bool multimem,
                        flagcxDeviceMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxTeam team = flagcxTeamIntra(*comm);
  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopAny> bar(coop, *comm, team,
                                                          index, multimem);
  bar.sync(order);
}

/* ================================================================
 * Category 7: Scalar Barrier — Inter (3)
 *
 * Inter barriers take a flagcxDevNet (which contains the full comm).
 * Team is derived from net->_dc for the inter team.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxInterBarrierArriveS(const void *netOpaque, flagcxCoopKind_t coopKind,
                          uint32_t index, flagcxDeviceMemoryOrder_t order,
                          flagcxDevNetFenceLevel fence) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxTeam team = flagcxMakeInterTeamFromNet(*net);
  flagcxDevBarrier<flagcxTeamTagInter, flagcxCoopAny> bar(coop, *net, team,
                                                          index);
  bar.arrive(order, fence);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxInterBarrierWaitS(const void *netOpaque, flagcxCoopKind_t coopKind,
                        uint32_t index, flagcxDeviceMemoryOrder_t order,
                        flagcxDevNetFenceLevel fence) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxTeam team = flagcxMakeInterTeamFromNet(*net);
  flagcxDevBarrier<flagcxTeamTagInter, flagcxCoopAny> bar(coop, *net, team,
                                                          index);
  bar.wait(order, fence);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxInterBarrierSyncS(const void *netOpaque, flagcxCoopKind_t coopKind,
                        uint32_t index, flagcxDeviceMemoryOrder_t order,
                        flagcxDevNetFenceLevel fence) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxTeam team = flagcxMakeInterTeamFromNet(*net);
  flagcxDevBarrier<flagcxTeamTagInter, flagcxCoopAny> bar(coop, *net, team,
                                                          index);
  bar.sync(order, fence);
}

/* ================================================================
 * Category 8: Scalar Barrier — World (3)
 *
 * World barriers use flagcxTeamTagWorld tag dispatch.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxWorldBarrierArriveS(const void *netOpaque, flagcxCoopKind_t coopKind,
                          uint32_t index, bool multimem,
                          flagcxDeviceMemoryOrder_t order,
                          flagcxDevNetFenceLevel fence) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopAny> bar(
      coop, flagcxTeamTagWorld{}, *net, index, multimem);
  bar.arrive(order, fence);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxWorldBarrierWaitS(const void *netOpaque, flagcxCoopKind_t coopKind,
                        uint32_t index, bool multimem,
                        flagcxDeviceMemoryOrder_t order,
                        flagcxDevNetFenceLevel fence) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopAny> bar(
      coop, flagcxTeamTagWorld{}, *net, index, multimem);
  bar.wait(order, fence);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxWorldBarrierSyncS(const void *netOpaque, flagcxCoopKind_t coopKind,
                        uint32_t index, bool multimem,
                        flagcxDeviceMemoryOrder_t order,
                        flagcxDevNetFenceLevel fence) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopAny> bar(
      coop, flagcxTeamTagWorld{}, *net, index, multimem);
  bar.sync(order, fence);
}

/* ================================================================
 * Category 9: Net — Obtain Pre-Allocated Transport (1)
 *
 * Constructs a flagcxDevNet on the stack and returns a pointer.
 * NOTE: This returns a stack pointer — the caller must use it
 * immediately within the same kernel invocation scope. The returned
 * pointer is valid for the duration of the calling function.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR const void *
flagcxDevNetGetFromCommS(const void *commOpaque, int idx) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  // When always-inlined, the static local is promoted into the caller's
  // scope by LLVM. This provides stable storage for the returned pointer.
  static flagcxDevNet storage;
  ::new (&storage) flagcxDevNet(*comm, idx);
  return &storage;
}

/* ================================================================
 * Category 10: Net — Signal / Counter / Flush (scalar coop) (7+3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevNetReadSignalS(const void *netOpaque, flagcxDevNetSignal_t signalId,
                        int bits, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  return net->readSignal(signalId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitSignalS(const void *netOpaque, flagcxCoopKind_t coopKind,
                        flagcxDevNetSignal_t signalId, uint64_t least, int bits,
                        flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->waitSignal(coop, signalId, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitSignalMeetShadowS(const void *netOpaque,
                                  flagcxCoopKind_t coopKind,
                                  flagcxDevNetSignal_t signalId, int bits,
                                  flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->waitSignalMeetShadow(coop, signalId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevNetReadCounterS(const void *netOpaque, flagcxDevNetCounter_t counterId,
                         int bits, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  return net->readCounter(counterId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitCounterS(const void *netOpaque, flagcxCoopKind_t coopKind,
                         flagcxDevNetCounter_t counterId, uint64_t least,
                         int bits, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->waitCounter(coop, counterId, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetFlushS(const void *netOpaque, flagcxCoopKind_t coopKind,
                   flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->flush(coop, order);
}

/* Reset / Shadow (no coop needed) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetResetSignal(const void *netOpaque, flagcxDevNetSignal_t slot) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  net->resetSignal(slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetResetCounter(const void *netOpaque, flagcxDevNetCounter_t slot) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  net->resetCounter(slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetIncreaseSignalShadow(const void *netOpaque,
                                 flagcxDevNetSignal_t slot, uint64_t delta) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  net->increaseSignalShadow(slot, delta);
}

/* ================================================================
 * Category 11: Net — Two-Sided (scalar coop) (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetSendS(const void *netOpaque, flagcxCoopKind_t coopKind,
                  const void *memOpaque, size_t offset, size_t count,
                  flagcxDataType_t datatype, int peer) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  return (int)net->send(coop, *mem, offset, count, datatype, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetRecvS(const void *netOpaque, flagcxCoopKind_t coopKind,
                  const void *memOpaque, size_t offset, size_t count,
                  flagcxDataType_t datatype, int peer) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  return (int)net->recv(coop, *mem, offset, count, datatype, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetWaitS(const void *netOpaque, flagcxCoopKind_t coopKind) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  return (int)net->wait(coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetTermS(const void *netOpaque, flagcxCoopKind_t coopKind) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  return (int)net->term(coop);
}

/* ================================================================
 * Category 12: Net — One-Sided put (scalar coop + team kind) (16)
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS(const void *netOpaque, const void *commOpaque,
                 flagcxTeamKind_t teamKind, int peer, const void *dstOpaque,
                 size_t dstOffset, const void *srcOpaque, size_t srcOffset,
                 size_t bytes, flagcxCoopKind_t coopKind) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_None{}, flagcxDevNet_None{}, coop);
}

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevNetPutS_RSigInc(
    const void *netOpaque, const void *commOpaque, flagcxTeamKind_t teamKind,
    int peer, const void *dstOpaque, size_t dstOffset, const void *srcOpaque,
    size_t srcOffset, size_t bytes, flagcxCoopKind_t coopKind,
    flagcxDevNetSignal_t remoteSignal) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_SignalInc{remoteSignal}, flagcxDevNet_None{}, coop);
}

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevNetPutS_RSigAdd(
    const void *netOpaque, const void *commOpaque, flagcxTeamKind_t teamKind,
    int peer, const void *dstOpaque, size_t dstOffset, const void *srcOpaque,
    size_t srcOffset, size_t bytes, flagcxCoopKind_t coopKind,
    flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
           flagcxDevNet_None{}, coop);
}

/* (CtrInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevNetPutS_RCtrInc(
    const void *netOpaque, const void *commOpaque, flagcxTeamKind_t teamKind,
    int peer, const void *dstOpaque, size_t dstOffset, const void *srcOpaque,
    size_t srcOffset, size_t bytes, flagcxCoopKind_t coopKind,
    flagcxDevNetCounter_t remoteCounter) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_CounterInc{remoteCounter}, flagcxDevNet_None{}, coop);
}

/* (None, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevNetPutS_LSigInc(
    const void *netOpaque, const void *commOpaque, flagcxTeamKind_t teamKind,
    int peer, const void *dstOpaque, size_t dstOffset, const void *srcOpaque,
    size_t srcOffset, size_t bytes, flagcxCoopKind_t coopKind,
    flagcxDevNetSignal_t localSignal) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_None{}, flagcxDevNet_SignalInc{localSignal}, coop);
}

/* (SigInc, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigInc_LSigInc(const void *netOpaque, const void *commOpaque,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dstOpaque, size_t dstOffset,
                                 const void *srcOpaque, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevNetSignal_t remoteSignal,
                                 flagcxDevNetSignal_t localSignal) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_SignalInc{remoteSignal},
           flagcxDevNet_SignalInc{localSignal}, coop);
}

/* (SigAdd, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigAdd_LSigInc(const void *netOpaque, const void *commOpaque,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dstOpaque, size_t dstOffset,
                                 const void *srcOpaque, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevNetSignal_t remoteSignal,
                                 uint64_t remoteValue,
                                 flagcxDevNetSignal_t localSignal) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
           flagcxDevNet_SignalInc{localSignal}, coop);
}

/* (CtrInc, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RCtrInc_LSigInc(const void *netOpaque, const void *commOpaque,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dstOpaque, size_t dstOffset,
                                 const void *srcOpaque, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevNetCounter_t remoteCounter,
                                 flagcxDevNetSignal_t localSignal) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_CounterInc{remoteCounter},
           flagcxDevNet_SignalInc{localSignal}, coop);
}

/* (None, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevNetPutS_LSigAdd(
    const void *netOpaque, const void *commOpaque, flagcxTeamKind_t teamKind,
    int peer, const void *dstOpaque, size_t dstOffset, const void *srcOpaque,
    size_t srcOffset, size_t bytes, flagcxCoopKind_t coopKind,
    flagcxDevNetSignal_t localSignal, uint64_t localValue) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_None{}, flagcxDevNet_SignalAdd{localSignal, localValue},
           coop);
}

/* (SigInc, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigInc_LSigAdd(const void *netOpaque, const void *commOpaque,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dstOpaque, size_t dstOffset,
                                 const void *srcOpaque, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevNetSignal_t remoteSignal,
                                 flagcxDevNetSignal_t localSignal,
                                 uint64_t localValue) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_SignalInc{remoteSignal},
           flagcxDevNet_SignalAdd{localSignal, localValue}, coop);
}

/* (SigAdd, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigAdd_LSigAdd(
    const void *netOpaque, const void *commOpaque, flagcxTeamKind_t teamKind,
    int peer, const void *dstOpaque, size_t dstOffset, const void *srcOpaque,
    size_t srcOffset, size_t bytes, flagcxCoopKind_t coopKind,
    flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevNetSignal_t localSignal, uint64_t localValue) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
           flagcxDevNet_SignalAdd{localSignal, localValue}, coop);
}

/* (CtrInc, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RCtrInc_LSigAdd(const void *netOpaque, const void *commOpaque,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dstOpaque, size_t dstOffset,
                                 const void *srcOpaque, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevNetCounter_t remoteCounter,
                                 flagcxDevNetSignal_t localSignal,
                                 uint64_t localValue) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_CounterInc{remoteCounter},
           flagcxDevNet_SignalAdd{localSignal, localValue}, coop);
}

/* (None, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevNetPutS_LCtrInc(
    const void *netOpaque, const void *commOpaque, flagcxTeamKind_t teamKind,
    int peer, const void *dstOpaque, size_t dstOffset, const void *srcOpaque,
    size_t srcOffset, size_t bytes, flagcxCoopKind_t coopKind,
    flagcxDevNetCounter_t localCounter) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_None{}, flagcxDevNet_CounterInc{localCounter}, coop);
}

/* (SigInc, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigInc_LCtrInc(const void *netOpaque, const void *commOpaque,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dstOpaque, size_t dstOffset,
                                 const void *srcOpaque, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevNetSignal_t remoteSignal,
                                 flagcxDevNetCounter_t localCounter) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_SignalInc{remoteSignal},
           flagcxDevNet_CounterInc{localCounter}, coop);
}

/* (SigAdd, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigAdd_LCtrInc(const void *netOpaque, const void *commOpaque,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dstOpaque, size_t dstOffset,
                                 const void *srcOpaque, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevNetSignal_t remoteSignal,
                                 uint64_t remoteValue,
                                 flagcxDevNetCounter_t localCounter) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
           flagcxDevNet_CounterInc{localCounter}, coop);
}

/* (CtrInc, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RCtrInc_LCtrInc(const void *netOpaque, const void *commOpaque,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dstOpaque, size_t dstOffset,
                                 const void *srcOpaque, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevNetCounter_t remoteCounter,
                                 flagcxDevNetCounter_t localCounter) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
           flagcxDevNet_CounterInc{remoteCounter},
           flagcxDevNet_CounterInc{localCounter}, coop);
}

/* ================================================================
 * Category 13: Net — One-Sided signal (scalar) (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalSigIncS(const void *netOpaque, const void *commOpaque,
                          flagcxTeamKind_t teamKind, int peer,
                          flagcxCoopKind_t coopKind,
                          flagcxDevNetSignal_t signal) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->signal(team, peer, flagcxDevNet_SignalInc{signal}, coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalSigAddS(const void *netOpaque, const void *commOpaque,
                          flagcxTeamKind_t teamKind, int peer,
                          flagcxCoopKind_t coopKind,
                          flagcxDevNetSignal_t signal, uint64_t value) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->signal(team, peer, flagcxDevNet_SignalAdd{signal, value}, coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalCtrIncS(const void *netOpaque, const void *commOpaque,
                          flagcxTeamKind_t teamKind, int peer,
                          flagcxCoopKind_t coopKind,
                          flagcxDevNetCounter_t counter) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->signal(team, peer, flagcxDevNet_CounterInc{counter}, coop);
}

/* ================================================================
 * Category 14: Net — One-Sided putValue<uint64_t> (scalar) (4)
 * ================================================================ */

/* (None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValueS(const void *netOpaque, const void *commOpaque,
                      flagcxTeamKind_t teamKind, int peer,
                      const void *dstOpaque, size_t dstOffset, uint64_t value,
                      flagcxCoopKind_t coopKind) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->putValue(team, peer, *dst, dstOffset, value, flagcxDevNet_None{}, coop);
}

/* (SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValueS_RSigInc(const void *netOpaque, const void *commOpaque,
                              flagcxTeamKind_t teamKind, int peer,
                              const void *dstOpaque, size_t dstOffset,
                              uint64_t value, flagcxCoopKind_t coopKind,
                              flagcxDevNetSignal_t remoteSignal) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->putValue(team, peer, *dst, dstOffset, value,
                flagcxDevNet_SignalInc{remoteSignal}, coop);
}

/* (SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValueS_RSigAdd(const void *netOpaque, const void *commOpaque,
                              flagcxTeamKind_t teamKind, int peer,
                              const void *dstOpaque, size_t dstOffset,
                              uint64_t value, flagcxCoopKind_t coopKind,
                              flagcxDevNetSignal_t remoteSignal,
                              uint64_t remoteAddValue) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->putValue(team, peer, *dst, dstOffset, value,
                flagcxDevNet_SignalAdd{remoteSignal, remoteAddValue}, coop);
}

/* (CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValueS_RCtrInc(const void *netOpaque, const void *commOpaque,
                              flagcxTeamKind_t teamKind, int peer,
                              const void *dstOpaque, size_t dstOffset,
                              uint64_t value, flagcxCoopKind_t coopKind,
                              flagcxDevNetCounter_t remoteCounter) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->putValue(team, peer, *dst, dstOffset, value,
                flagcxDevNet_CounterInc{remoteCounter}, coop);
}

/* ================================================================
 * Category 15: Net — One-Sided get (scalar) (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetGetS(const void *netOpaque, const void *commOpaque,
                 flagcxTeamKind_t teamKind, int peer, const void *srcOpaque,
                 size_t srcOffset, const void *dstOpaque, size_t dstOffset,
                 size_t bytes, flagcxCoopKind_t coopKind) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->get(team, peer, *src, srcOffset, *dst, dstOffset, bytes, coop);
}

#endif /* FLAGCX_DEVICE_SCALAR_IR_IMPL_H_ */
