/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Unified One-Sided IR — Implementation.
 *
 * Transport-transparent dispatch: checks flagcxGetPeerPointer() for P2P
 * reachability, falls back to Net path otherwise.
 *
 * Included by the bitcode compilation unit via flagcx_device_scalar_ir_impl.h.
 *
 * NOTE: Implementation order matters. Signal/Wait/Flush/Reset (U4-U7) are
 * defined first because Put variants (U1, U3) call them for P2P signal
 * delivery on the data-complete path.
 ************************************************************************/
#ifndef FLAGCX_DEVICE_UNIFIED_IR_IMPL_H_
#define FLAGCX_DEVICE_UNIFIED_IR_IMPL_H_

#include "flagcx_device_unified_ir.h"

/* ================================================================
 * Internal helper: scoped memory fence
 * ================================================================ */

static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxScopedFence(flagcxDeviceScope_t scope) {
  switch (scope) {
    case flagcxDeviceScopeSystem:
      __threadfence_system();
      break;
    case flagcxDeviceScopeDevice:
      __threadfence();
      break;
    default:
      break; // Block/Thread: no fence needed
  }
}

/* ================================================================
 * Internal helper: cooperative memcpy (P2P path)
 *
 * Distributes byte copy across threads in the cooperative group.
 * Uses 4-byte aligned loads/stores for the bulk, byte copy for tail.
 *
 * Requires: dst and src must be 4-byte aligned.
 * ================================================================ */

static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopMemcpy(flagcxCoopKind_t coopKind, void *dst, const void *src,
                 size_t bytes) {
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  int rank = coop.threadRank();
  int size = coop.size();
  // 4-byte aligned bulk copy (safe for GPU — no strict alignment trap)
  uint32_t *d = (uint32_t *)dst;
  const uint32_t *s = (const uint32_t *)src;
  size_t n4 = bytes / 4;
  for (size_t i = (size_t)rank; i < n4; i += (size_t)size) {
    d[i] = s[i];
  }
  // Tail bytes (thread 0 only)
  size_t tail = bytes - n4 * 4;
  if (tail > 0 && rank == 0) {
    char *dc = (char *)dst + n4 * 4;
    const char *sc = (const char *)src + n4 * 4;
    for (size_t i = 0; i < tail; i++)
      dc[i] = sc[i];
  }
}

/* ================================================================
 * Category U4: Unified Signal (2)
 *
 * DEFINED FIRST — Put variants call these for P2P signal delivery.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalInc(const void *commOpaque, flagcxTeamKind_t teamKind, int peer,
                   flagcxDevNetSignal_t signal, flagcxCoopKind_t coopKind,
                   flagcxDeviceScope_t scope, int contextId) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  if (comm->_commBase.p2pSignalSupport(peer)) {
    // P2P fast path: direct atomic on peer's IPC-mapped signal buffer
    const void *netOpaque = flagcxDevNetGetFromCommS(commOpaque, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    uint64_t *peerBuf = comm->_commBase.getSignalPeerPtr(peer);
    int slot = net->contextId * comm->_commBase.signalCount + (int)signal;
    DeviceAPI::Atomic::fetchAdd(&peerBuf[slot], (uint64_t)1,
                                flagcxDeviceMemoryOrderRelease);
  } else {
    // Net FIFO fallback (inter-node)
    const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxDevNetSignalSigIncS(net, commOpaque, teamKind, peer, coopKind,
                              signal);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalAdd(const void *commOpaque, flagcxTeamKind_t teamKind, int peer,
                   flagcxDevNetSignal_t signal, uint64_t value,
                   flagcxCoopKind_t coopKind, flagcxDeviceScope_t scope,
                   int contextId) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  if (comm->_commBase.p2pSignalSupport(peer)) {
    // P2P fast path: direct atomic on peer's IPC-mapped signal buffer
    const void *netOpaque = flagcxDevNetGetFromCommS(commOpaque, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    uint64_t *peerBuf = comm->_commBase.getSignalPeerPtr(peer);
    int slot = net->contextId * comm->_commBase.signalCount + (int)signal;
    DeviceAPI::Atomic::fetchAdd(&peerBuf[slot], value,
                                flagcxDeviceMemoryOrderRelease);
  } else {
    // Net FIFO fallback (inter-node)
    const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxDevNetSignalSigAddS(net, commOpaque, teamKind, peer, coopKind, signal,
                              value);
  }
}

/* ================================================================
 * Category U5: Unified Wait (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignal(const void *commOpaque, flagcxDevNetSignal_t signal,
                    uint64_t least, int bits, flagcxCoopKind_t coopKind,
                    flagcxDeviceMemoryOrder_t order) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
  flagcxDevNetWaitSignalS(net, coopKind, signal, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitCounter(const void *commOpaque, flagcxDevNetCounter_t counter,
                     uint64_t least, int bits, flagcxCoopKind_t coopKind,
                     flagcxDeviceMemoryOrder_t order) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
  flagcxDevNetWaitCounterS(net, coopKind, counter, least, bits, order);
}

/* ================================================================
 * Category U6: Unified Read (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevReadSignal(const void *commOpaque, flagcxDevNetSignal_t signal,
                    int bits, flagcxDeviceMemoryOrder_t order) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
  return flagcxDevNetReadSignalS(net, signal, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevReadCounter(const void *commOpaque, flagcxDevNetCounter_t counter,
                     int bits, flagcxDeviceMemoryOrder_t order) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
  return flagcxDevNetReadCounterS(net, counter, bits, order);
}

/* ================================================================
 * Category U7: Unified Flush / Reset / Shadow (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevFlush(const void *commOpaque, flagcxCoopKind_t coopKind,
               flagcxDeviceMemoryOrder_t order) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
  flagcxDevNetFlushS(net, coopKind, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevResetSignal(const void *commOpaque, flagcxDevNetSignal_t slot) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
  flagcxDevNetResetSignal(net, slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevResetCounter(const void *commOpaque, flagcxDevNetCounter_t slot) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
  flagcxDevNetResetCounter(net, slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevIncreaseSignalShadow(const void *commOpaque, flagcxDevNetSignal_t slot,
                              uint64_t delta) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
  flagcxDevNetIncreaseSignalShadow(net, slot, delta);
}

/* ================================================================
 * Category U8: Unified Barrier (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevBarrierArrive(const void *commOpaque, flagcxTeamKind_t teamKind,
                       uint32_t index, flagcxCoopKind_t coopKind,
                       flagcxDeviceMemoryOrder_t order,
                       flagcxDeviceScope_t scope) {
  switch (teamKind) {
    case FLAGCX_TEAM_INTRA:
      flagcxIntraBarrierArriveS(commOpaque, coopKind, index,
                                /*multimem=*/false, order);
      break;
    case FLAGCX_TEAM_INTER: {
      const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
      flagcxInterBarrierArriveS(net, coopKind, index, order,
                                flagcxDevNetFenceLevel::Relaxed);
      break;
    }
    case FLAGCX_TEAM_WORLD: {
      const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
      flagcxWorldBarrierArriveS(net, coopKind, index, /*multimem=*/false, order,
                                flagcxDevNetFenceLevel::Relaxed);
      break;
    }
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevBarrierWait(const void *commOpaque, flagcxTeamKind_t teamKind,
                     uint32_t index, flagcxCoopKind_t coopKind,
                     flagcxDeviceMemoryOrder_t order,
                     flagcxDeviceScope_t scope) {
  switch (teamKind) {
    case FLAGCX_TEAM_INTRA:
      flagcxIntraBarrierWaitS(commOpaque, coopKind, index,
                              /*multimem=*/false, order);
      break;
    case FLAGCX_TEAM_INTER: {
      const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
      flagcxInterBarrierWaitS(net, coopKind, index, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
    case FLAGCX_TEAM_WORLD: {
      const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
      flagcxWorldBarrierWaitS(net, coopKind, index, /*multimem=*/false, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevBarrierSync(const void *commOpaque, flagcxTeamKind_t teamKind,
                     uint32_t index, flagcxCoopKind_t coopKind,
                     flagcxDeviceMemoryOrder_t order,
                     flagcxDeviceScope_t scope) {
  switch (teamKind) {
    case FLAGCX_TEAM_INTRA:
      flagcxIntraBarrierSyncS(commOpaque, coopKind, index,
                              /*multimem=*/false, order);
      break;
    case FLAGCX_TEAM_INTER: {
      const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
      flagcxInterBarrierSyncS(net, coopKind, index, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
    case FLAGCX_TEAM_WORLD: {
      const void *net = flagcxDevNetGetFromCommS(commOpaque, 0);
      flagcxWorldBarrierSyncS(net, coopKind, index, /*multimem=*/false, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
  }
}

/* ================================================================
 * Category U1: Unified Put (4)
 *
 * These come AFTER signal/wait so that P2P signal delivery calls
 * (flagcxDevSignalInc, flagcxDevSignalAdd) are already defined.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut(const void *commOpaque, const void *dstOpaque, size_t dstOffset,
             const void *srcOpaque, size_t srcOffset, size_t bytes,
             flagcxTeamKind_t teamKind, int peer, flagcxCoopKind_t coopKind,
             flagcxDeviceScope_t scope, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  void *peerPtr = flagcxGetPeerPointer(*dst, dstOffset, team, peer);
  if (peerPtr != nullptr) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, 0);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_None{}, flagcxDevNet_None{}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut_RSigInc(const void *commOpaque, const void *dstOpaque,
                     size_t dstOffset, const void *srcOpaque, size_t srcOffset,
                     size_t bytes, flagcxTeamKind_t teamKind, int peer,
                     flagcxCoopKind_t coopKind, flagcxDeviceScope_t scope,
                     flagcxDeviceMemoryOrder_t order,
                     flagcxDevNetSignal_t remoteSignal, int contextId) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  void *peerPtr = flagcxGetPeerPointer(*dst, dstOffset, team, peer);
  if (peerPtr != nullptr) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
    // Signal after data lands
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      flagcxScopedFence(flagcxDeviceScopeSystem);
      flagcxDevSignalInc(commOpaque, teamKind, peer, remoteSignal,
                         FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem,
                         contextId);
    }
    coop.sync();
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalInc{remoteSignal}, flagcxDevNet_None{}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevPut_RSigAdd(
    const void *commOpaque, const void *dstOpaque, size_t dstOffset,
    const void *srcOpaque, size_t srcOffset, size_t bytes,
    flagcxTeamKind_t teamKind, int peer, flagcxCoopKind_t coopKind,
    flagcxDeviceScope_t scope, flagcxDeviceMemoryOrder_t order,
    flagcxDevNetSignal_t remoteSignal, uint64_t signalValue, int contextId) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  void *peerPtr = flagcxGetPeerPointer(*dst, dstOffset, team, peer);
  if (peerPtr != nullptr) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      flagcxScopedFence(flagcxDeviceScopeSystem);
      flagcxDevSignalAdd(commOpaque, teamKind, peer, remoteSignal, signalValue,
                         FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem,
                         contextId);
    }
    coop.sync();
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalAdd{remoteSignal, signalValue},
             flagcxDevNet_None{}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut_RCtrInc(const void *commOpaque, const void *dstOpaque,
                     size_t dstOffset, const void *srcOpaque, size_t srcOffset,
                     size_t bytes, flagcxTeamKind_t teamKind, int peer,
                     flagcxCoopKind_t coopKind, flagcxDeviceScope_t scope,
                     flagcxDeviceMemoryOrder_t order,
                     flagcxDevNetCounter_t remoteCounter, int contextId) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  void *peerPtr = flagcxGetPeerPointer(*dst, dstOffset, team, peer);
  if (peerPtr != nullptr) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
    // Counter increment after data lands
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      flagcxScopedFence(flagcxDeviceScopeSystem);
      const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
      flagcxDevNetSignalCtrIncS(net, commOpaque, teamKind, peer,
                                FLAGCX_COOP_THREAD, remoteCounter);
    }
    coop.sync();
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_CounterInc{remoteCounter}, flagcxDevNet_None{}, coop);
  }
}

/* ================================================================
 * Category U2: Unified Get (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevGet(const void *commOpaque, const void *srcOpaque, size_t srcOffset,
             const void *dstOpaque, size_t dstOffset, size_t bytes,
             flagcxTeamKind_t teamKind, int peer, flagcxCoopKind_t coopKind,
             flagcxDeviceScope_t scope, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  void *peerPtr = flagcxGetPeerPointer(*src, srcOffset, team, peer);
  if (peerPtr != nullptr) {
    void *localDst = flagcxGetLocalPointer(*dst, dstOffset);
    flagcxCoopMemcpy(coopKind, localDst, peerPtr, bytes);
    if (order == flagcxDeviceMemoryOrderAcquire ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, 0);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->get(team, peer, *src, srcOffset, *dst, dstOffset, bytes, coop);
  }
}

/* ================================================================
 * Category U3: Unified PutValue (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPutValue(const void *commOpaque, const void *dstOpaque,
                  size_t dstOffset, uint64_t value, flagcxTeamKind_t teamKind,
                  int peer, flagcxCoopKind_t coopKind,
                  flagcxDeviceScope_t scope, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  void *peerPtr = flagcxGetPeerPointer(*dst, dstOffset, team, peer);
  if (peerPtr != nullptr) {
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    *(volatile uint64_t *)peerPtr = value;
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, 0);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->putValue(team, peer, *dst, dstOffset, value, flagcxDevNet_None{},
                  coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPutValue_RSigInc(const void *commOpaque, const void *dstOpaque,
                          size_t dstOffset, uint64_t value,
                          flagcxTeamKind_t teamKind, int peer,
                          flagcxCoopKind_t coopKind, flagcxDeviceScope_t scope,
                          flagcxDeviceMemoryOrder_t order,
                          flagcxDevNetSignal_t remoteSignal, int contextId) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  void *peerPtr = flagcxGetPeerPointer(*dst, dstOffset, team, peer);
  if (peerPtr != nullptr) {
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    *(volatile uint64_t *)peerPtr = value;
    flagcxScopedFence(flagcxDeviceScopeSystem);
    flagcxDevSignalInc(commOpaque, teamKind, peer, remoteSignal,
                       FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem, contextId);
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->putValue(team, peer, *dst, dstOffset, value,
                  flagcxDevNet_SignalInc{remoteSignal}, coop);
  }
}

#endif // FLAGCX_DEVICE_UNIFIED_IR_IMPL_H_
