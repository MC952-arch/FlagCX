/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API C-style wrapper functions for LLVM IR generation.
 *
 * DEPRECATED: This struct-based API (suffix "C") will be removed in a
 * future release. Prefer the scalar API (flagcx_device_scalar_ir.h,
 * suffix "S") which requires no struct instantiation and is compatible
 * with Triton / LLVM IR consumers.
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
 *
 * @param comm  Opaque pointer to flagcxDevComm (device communicator).
 * ================================================================ */

/** @brief Get this rank's global index. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevCommGetRank(const void *comm);
/** @brief Get total number of ranks. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevCommGetSize(const void *comm);
/** @brief Get this rank's intra-node index. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevCommGetIntraRank(const void *comm);
/** @brief Get intra-node group size. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevCommGetIntraSize(const void *comm);

/* ================================================================
 * Category 2: Cooperative Group — Init / Query / Sync (8)
 *
 * @param coop  Pointer to caller-allocated flagcxCoopAny struct (output).
 * ================================================================ */

/** @brief Initialize coop as block-level cooperation. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxCoopAnyInitBlockC(void *coop);
/** @brief Initialize coop as warp-level cooperation. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxCoopAnyInitWarpC(void *coop);
/** @brief Initialize coop as single-thread cooperation. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxCoopAnyInitThreadC(void *coop);
/** @brief Initialize coop as tile-span cooperation.
 *  @param t0      First tile index.
 *  @param nTiles  Number of tiles.
 *  @param id      Tile ID within the span. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxCoopAnyInitTileSpanC(void *coop, int t0, int nTiles, int id);
/** @brief Initialize coop as lane-masked cooperation.
 *  @param laneMask  Bitmask of active lanes. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxCoopAnyInitLanesC(void *coop, uint32_t laneMask);

/** @brief Get thread rank within coop. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxCoopThreadRankC(const void *coop);
/** @brief Get group size of coop. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxCoopSizeC(const void *coop);
/** @brief Synchronize all threads in coop. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxCoopSyncC(void *coop);

/* ================================================================
 * Category 3: Team Functions (5)
 *
 * @param comm  Opaque pointer to flagcxDevComm.
 * @param team  Pointer to caller-allocated flagcxTeam struct (output).
 * @param rank  Team-local rank to convert.
 * ================================================================ */

/** @brief Populate team struct for intra-node topology. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxGetTeamIntraC(const void *comm, void *team);
/** @brief Populate team struct for world topology. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxGetTeamWorldC(const void *comm, void *team);
/** @brief Populate team struct for inter-node topology. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxGetTeamInterC(const void *comm, void *team);
/** @brief Convert team-local rank to world rank. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxTeamRankToWorldC(const void *comm, const void *team, int rank);
/** @brief Convert team-local rank to intra-node rank. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxTeamRankToIntraC(const void *comm, const void *team, int rank);

/* ================================================================
 * Category 4: Pointer Access (4)
 *
 * @param mem     Opaque pointer to flagcxDevMem (memory descriptor).
 * @param offset  Byte offset into the memory region.
 * @param team    Opaque pointer to flagcxTeam (for peer pointer).
 * @param peer    Peer rank within the team.
 * @param comm    Opaque pointer to flagcxDevComm (for multicast).
 * ================================================================ */

/** @brief Get pointer to peer's memory region (team-relative). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void *
flagcxGetPeerPointerC(const void *mem, size_t offset, const void *team,
                      int peer);
/** @brief Get pointer to local memory at offset. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void *
flagcxGetLocalPointerC(const void *mem, size_t offset);
/** @brief Get pointer to intra-node peer's memory. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void *
flagcxGetIntraPointerC(const void *mem, size_t offset, int peer);
/** @brief Get multicast pointer spanning all comm ranks. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void *
flagcxGetMulticastPointerC(const void *mem, size_t offset, const void *comm);

/* ================================================================
 * Category 5: Utility (1)
 * ================================================================ */

/** @brief Get byte size of a flagcxDataType_t element. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR size_t
flagcxDataTypeSizeDevice(flagcxDataType_t dt);

/* ================================================================
 * Category 6: Intra-Node Barrier Session (4)
 *
 * @param session   Pointer to caller-allocated flagcxIntraBarrierSession_C.
 * @param coop      Pointer to initialized flagcxCoopAny struct.
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param team      Pointer to flagcxTeam struct.
 * @param index     Barrier channel index (typically blockIdx.x).
 * @param multimem  Whether to use multicast memory barrier variant.
 * @param order     Memory ordering semantics.
 * ================================================================ */

/** @brief Initialize an intra-node barrier session. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxIntraBarrierSessionInitC(void *session, const void *coop,
                               const void *comm, const void *team,
                               uint32_t index, bool multimem);
/** @brief Signal arrival at intra-node barrier. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxIntraBarrierSessionArriveC(void *session,
                                 flagcxDeviceMemoryOrder_t order);
/** @brief Wait for all peers at intra-node barrier. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxIntraBarrierSessionWaitC(void *session, flagcxDeviceMemoryOrder_t order);
/** @brief Arrive + wait (full sync) at intra-node barrier. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxIntraBarrierSessionSyncC(void *session, flagcxDeviceMemoryOrder_t order);

/* ================================================================
 * Category 7: Inter-Node Barrier Session (2)
 *
 * @param session  Pointer to caller-allocated flagcxInterBarrierSession_C.
 * @param coop     Pointer to initialized flagcxCoopAny struct.
 * @param trans    Pointer to flagcxDevNet (transport handle).
 * @param team     Pointer to flagcxTeam struct.
 * @param index    Barrier channel index.
 * @param order    Memory ordering semantics.
 * @param fence    Network fence level (controls DMA visibility).
 * ================================================================ */

/** @brief Initialize an inter-node barrier session. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxInterBarrierSessionInitC(void *session, const void *coop,
                               const void *trans, const void *team,
                               uint32_t index);
/** @brief Arrive + wait (full sync) at inter-node barrier. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxInterBarrierSessionSyncC(void *session, flagcxDeviceMemoryOrder_t order,
                               flagcxDevNetFenceLevel fence);

/* ================================================================
 * Category 8: World Barrier Session (2)
 *
 * @param session   Pointer to caller-allocated flagcxBarrierSession_C.
 * @param coop      Pointer to initialized flagcxCoopAny struct.
 * @param tag       World team tag (flagcxTeamTagWorld).
 * @param trans     Pointer to flagcxDevNet (transport handle).
 * @param index     Barrier channel index.
 * @param multimem  Whether to use multicast memory for intra phase.
 * @param order     Memory ordering semantics.
 * @param fence     Network fence level.
 * ================================================================ */

/** @brief Initialize a world barrier session (intra + inter). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxWorldBarrierSessionInitC(void *session, const void *coop,
                               flagcxTeamTagWorld tag, const void *trans,
                               uint32_t index, bool multimem);
/** @brief Arrive + wait (full sync) at world barrier. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxWorldBarrierSessionSyncC(void *session, flagcxDeviceMemoryOrder_t order,
                               flagcxDevNetFenceLevel fence);

/* ================================================================
 * Category 9: Transport — Init / Signal Read / Wait / Counter / Flush (7)
 *
 * @param trans     Pointer to caller-allocated flagcxDevNet struct.
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param idx       Context index.
 * @param coop      Pointer to flagcxCoopAny struct (for wait/flush).
 * @param signalId  Signal slot identifier.
 * @param counterId Counter slot identifier.
 * @param least     Minimum value to wait for.
 * @param bits      Bit width for comparison (32 or 64).
 * @param order     Memory ordering semantics.
 * ================================================================ */

/** @brief Initialize a transport handle (placement-new into trans). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetInitC(void *trans, const void *comm, int idx);
/** @brief Read a signal value (non-blocking). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR uint64_t
flagcxDevNetReadSignalC(const void *trans, flagcxDevNetSignal_t signalId,
                        int bits, flagcxDeviceMemoryOrder_t order);
/** @brief Spin-wait until signal >= least. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetWaitSignalC(const void *trans, const void *coop,
                        flagcxDevNetSignal_t signalId, uint64_t least, int bits,
                        flagcxDeviceMemoryOrder_t order);
/** @brief Spin-wait until signal meets its shadow value. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetWaitSignalMeetShadowC(const void *trans, const void *coop,
                                  flagcxDevNetSignal_t signalId, int bits,
                                  flagcxDeviceMemoryOrder_t order);
/** @brief Read a counter value (non-blocking). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR uint64_t
flagcxDevNetReadCounterC(const void *trans, flagcxDevNetCounter_t counterId,
                         int bits, flagcxDeviceMemoryOrder_t order);
/** @brief Spin-wait until counter >= least. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetWaitCounterC(const void *trans, const void *coop,
                         flagcxDevNetCounter_t counterId, uint64_t least,
                         int bits, flagcxDeviceMemoryOrder_t order);
/** @brief Flush pending RDMA/network writes. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetFlushC(const void *trans, const void *coop,
                   flagcxDeviceMemoryOrder_t order);

/* ================================================================
 * Category 9b: Net — Reset / Shadow (3)
 *
 * Shared with scalar API — no coop needed.
 * @param net    Opaque pointer to flagcxDevNet.
 * @param slot   Signal or counter slot.
 * @param delta  Value to add to signal shadow.
 * ================================================================ */

/** @brief Reset a signal slot to zero. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetResetSignal(const void *net, flagcxDevNetSignal_t slot);
/** @brief Reset a counter slot to zero. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetResetCounter(const void *net, flagcxDevNetCounter_t slot);
/** @brief Increase the local shadow for a signal. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetIncreaseSignalShadow(const void *net, flagcxDevNetSignal_t slot,
                                 uint64_t delta);

/* ================================================================
 * Category 10: Transport — Two-Sided (4)
 *
 * @param trans     Pointer to flagcxDevNet struct.
 * @param coop      Pointer to flagcxCoopAny struct.
 * @param mem       Opaque pointer to flagcxDevMem (buffer).
 * @param offset    Byte offset into buffer.
 * @param count     Number of elements.
 * @param datatype  Element data type.
 * @param peer      Remote rank.
 * @return          flagcxResult_t cast to int (0 = success).
 * ================================================================ */

/** @brief Initiate a send. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevNetSendC(const void *trans, const void *coop, const void *mem,
                  size_t offset, size_t count, flagcxDataType_t datatype,
                  int peer);
/** @brief Initiate a receive. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevNetRecvC(const void *trans, const void *coop, const void *mem,
                  size_t offset, size_t count, flagcxDataType_t datatype,
                  int peer);
/** @brief Wait for pending two-sided operations to complete. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevNetWaitC(const void *trans, const void *coop);
/** @brief Terminate the transport session. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevNetTermC(const void *trans, const void *coop);

/* ================================================================
 * Category 11: Transport — One-Sided put (16)
 *
 * Naming: flagcxDevNetPut[_R<remote>][_L<local>]C
 * Actions: None, SigInc, SigAdd, CtrInc
 *
 * @param trans          Pointer to flagcxDevNet struct.
 * @param team           Pointer to flagcxTeam struct.
 * @param peer           Destination rank within the team.
 * @param dst            Opaque pointer to destination flagcxDevMem.
 * @param dstOffset      Byte offset into destination memory.
 * @param src            Opaque pointer to source flagcxDevMem.
 * @param srcOffset      Byte offset into source memory.
 * @param bytes          Number of bytes to transfer.
 * @param coop           Pointer to flagcxCoopAny struct.
 * @param remoteSignal   Remote signal slot (R variants).
 * @param remoteValue    Value to add to remote signal (RSigAdd).
 * @param remoteCounter  Remote counter slot (RCtrInc).
 * @param localSignal    Local signal slot (L variants).
 * @param localValue     Value to add to local signal (LSigAdd).
 * @param localCounter   Local counter slot (LCtrInc).
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutC(const void *trans, const void *team, int peer, const void *dst,
                 size_t dstOffset, const void *src, size_t srcOffset,
                 size_t bytes, const void *coop);

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RSigIncC(const void *trans, const void *team, int peer,
                         const void *dst, size_t dstOffset, const void *src,
                         size_t srcOffset, size_t bytes, const void *coop,
                         flagcxDevNetSignal_t remoteSignal);

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetPut_RSigAddC(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue);

/* (CtrInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RCtrIncC(const void *trans, const void *team, int peer,
                         const void *dst, size_t dstOffset, const void *src,
                         size_t srcOffset, size_t bytes, const void *coop,
                         flagcxDevNetCounter_t remoteCounter);

/* (None, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_LSigIncC(const void *trans, const void *team, int peer,
                         const void *dst, size_t dstOffset, const void *src,
                         size_t srcOffset, size_t bytes, const void *coop,
                         flagcxDevNetSignal_t localSignal);

/* (SigInc, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RSigInc_LSigIncC(const void *trans, const void *team, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, const void *coop,
                                 flagcxDevNetSignal_t remoteSignal,
                                 flagcxDevNetSignal_t localSignal);

/* (SigAdd, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RSigAdd_LSigIncC(const void *trans, const void *team, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, const void *coop,
                                 flagcxDevNetSignal_t remoteSignal,
                                 uint64_t remoteValue,
                                 flagcxDevNetSignal_t localSignal);

/* (CtrInc, SigInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RCtrInc_LSigIncC(const void *trans, const void *team, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, const void *coop,
                                 flagcxDevNetCounter_t remoteCounter,
                                 flagcxDevNetSignal_t localSignal);

/* (None, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_LSigAddC(const void *trans, const void *team, int peer,
                         const void *dst, size_t dstOffset, const void *src,
                         size_t srcOffset, size_t bytes, const void *coop,
                         flagcxDevNetSignal_t localSignal, uint64_t localValue);

/* (SigInc, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RSigInc_LSigAddC(const void *trans, const void *team, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, const void *coop,
                                 flagcxDevNetSignal_t remoteSignal,
                                 flagcxDevNetSignal_t localSignal,
                                 uint64_t localValue);

/* (SigAdd, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RSigAdd_LSigAddC(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevNetSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevNetSignal_t localSignal, uint64_t localValue);

/* (CtrInc, SigAdd) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RCtrInc_LSigAddC(const void *trans, const void *team, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, const void *coop,
                                 flagcxDevNetCounter_t remoteCounter,
                                 flagcxDevNetSignal_t localSignal,
                                 uint64_t localValue);

/* (None, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_LCtrIncC(const void *trans, const void *team, int peer,
                         const void *dst, size_t dstOffset, const void *src,
                         size_t srcOffset, size_t bytes, const void *coop,
                         flagcxDevNetCounter_t localCounter);

/* (SigInc, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RSigInc_LCtrIncC(const void *trans, const void *team, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, const void *coop,
                                 flagcxDevNetSignal_t remoteSignal,
                                 flagcxDevNetCounter_t localCounter);

/* (SigAdd, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RSigAdd_LCtrIncC(const void *trans, const void *team, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, const void *coop,
                                 flagcxDevNetSignal_t remoteSignal,
                                 uint64_t remoteValue,
                                 flagcxDevNetCounter_t localCounter);

/* (CtrInc, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPut_RCtrInc_LCtrIncC(const void *trans, const void *team, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, const void *coop,
                                 flagcxDevNetCounter_t remoteCounter,
                                 flagcxDevNetCounter_t localCounter);

/* ================================================================
 * Category 12: Transport — One-Sided signal (3)
 *
 * Remotely increment/add a signal or counter without data transfer.
 *
 * @param trans   Pointer to flagcxDevNet struct.
 * @param team    Pointer to team struct (intra/inter/world).
 * @param peer    Destination rank within the team.
 * @param coop    Pointer to flagcxCoopAny struct.
 * @param signal  Remote signal slot to manipulate.
 * @param counter Remote counter slot to increment.
 * @param value   Value to add (SigAdd variant).
 * ================================================================ */

/** @brief Increment remote signal by 1. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetSignalSigIncC(const void *trans, const void *team, int peer,
                          const void *coop, flagcxDevNetSignal_t signal);
/** @brief Add value to remote signal. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetSignalSigAddC(const void *trans, const void *team, int peer,
                          const void *coop, flagcxDevNetSignal_t signal,
                          uint64_t value);
/** @brief Increment remote counter by 1. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetSignalCtrIncC(const void *trans, const void *team, int peer,
                          const void *coop, flagcxDevNetCounter_t counter);

/* ================================================================
 * Category 13: Transport — One-Sided putValue<uint64_t> (4)
 *
 * Write a single 64-bit value to a remote buffer with optional
 * remote signal/counter action. No LocalAction — only remote side effects.
 *
 * @param trans          Pointer to flagcxDevNet struct.
 * @param team           Pointer to team struct (intra/inter/world).
 * @param peer           Destination rank within the team.
 * @param dst            Remote destination buffer.
 * @param dstOffset      Byte offset into dst.
 * @param value          The 64-bit value to write.
 * @param coop           Pointer to flagcxCoopAny struct.
 * @param remoteSignal   Remote signal slot (R variants).
 * @param remoteAddValue Value to add to remote signal (RSigAdd).
 * @param remoteCounter  Remote counter slot (RCtrInc).
 * ================================================================ */

/** @brief Put a 64-bit value (no side effect). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutValueC(const void *trans, const void *team, int peer,
                      const void *dst, size_t dstOffset, uint64_t value,
                      const void *coop);

/** @brief Put a 64-bit value + increment remote signal. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutValue_RSigIncC(const void *trans, const void *team, int peer,
                              const void *dst, size_t dstOffset, uint64_t value,
                              const void *coop,
                              flagcxDevNetSignal_t remoteSignal);

/** @brief Put a 64-bit value + add to remote signal. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetPutValue_RSigAddC(
    const void *trans, const void *team, int peer, const void *dst,
    size_t dstOffset, uint64_t value, const void *coop,
    flagcxDevNetSignal_t remoteSignal, uint64_t remoteAddValue);

/** @brief Put a 64-bit value + increment remote counter. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutValue_RCtrIncC(const void *trans, const void *team, int peer,
                              const void *dst, size_t dstOffset, uint64_t value,
                              const void *coop,
                              flagcxDevNetCounter_t remoteCounter);

/* ================================================================
 * Category 14: Transport — One-Sided get (1)
 *
 * RDMA read from a remote peer's buffer into local memory.
 *
 * @param trans      Pointer to flagcxDevNet struct.
 * @param team       Pointer to team struct (intra/inter/world).
 * @param peer       Source rank within the team.
 * @param src        Remote source buffer.
 * @param srcOffset  Byte offset into src.
 * @param dst        Local destination buffer.
 * @param dstOffset  Byte offset into dst.
 * @param bytes      Number of bytes to transfer.
 * @param coop       Pointer to flagcxCoopAny struct.
 * ================================================================ */

/** @brief RDMA get (remote read). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetGetC(const void *trans, const void *team, int peer, const void *src,
                 size_t srcOffset, const void *dst, size_t dstOffset,
                 size_t bytes, const void *coop);

#endif /* FLAGCX_DEVICE_WRAPPER_H_ */
