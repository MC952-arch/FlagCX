/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * IB Retransmission Support - Implementation
 ************************************************************************/

#include "ib_retrans.h"
#include "flagcx_common.h"
#include "ibvwrap.h"
#include "param.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

FLAGCX_PARAM(IbRetransEnable, "IB_RETRANS_ENABLE", 0);
FLAGCX_PARAM(IbRetransTimeout, "IB_RETRANS_TIMEOUT", 5000);
FLAGCX_PARAM(IbRetransMaxRetry, "IB_RETRANS_MAX_RETRY", 10);
FLAGCX_PARAM(IbRetransAckInterval, "IB_RETRANS_ACK_INTERVAL", 16);
FLAGCX_PARAM(IbMaxOutstanding, "IB_MAX_OUTSTANDING", 16);
flagcxResult_t flagcxIbRetransInit(struct flagcxIbRetransState *state) {
  if (!state)
    return flagcxInternalError;

  memset(state, 0, sizeof(struct flagcxIbRetransState));

  const int requested = flagcxParamIbRetransEnable();
  state->enabled = requested && flagcxIbRetransUdSupported();
  if (requested && !state->enabled) {
    INFO(FLAGCX_INIT | FLAGCX_NET,
         "IB retransmission requires UD AH/SRQ support and is unavailable "
         "with the selected verbs ABI; disabling it");
  }
  state->maxRetry = flagcxParamIbRetransMaxRetry();
  state->ackInterval = flagcxParamIbRetransAckInterval();
  state->minRtoUs = flagcxParamIbRetransTimeout();
  state->maxRtoUs = state->minRtoUs * 64;

  TRACE(FLAGCX_NET,
        "flagcxIbRetransInit: enabled=%d, max_retry=%d, ack_interval=%d, "
        "min_rto=%luus",
        state->enabled, state->maxRetry, state->ackInterval,
        (unsigned long)state->minRtoUs);

  state->rtoUs = state->minRtoUs;
  state->srttUs = 0;
  state->rttvarUs = 0;

  state->sendSeq = 0;
  state->sendUna = 0;
  state->recvSeq = 0;
  state->lastAckSeq = 0;
  state->lastAckSendTimeUs = 0;

  state->bufferHead = 0;
  state->bufferTail = 0;
  state->bufferCount = 0;

  state->lastAckTimeUs = flagcxIbGetTimeUs();

  if (state->enabled) {
    INFO(FLAGCX_INIT | FLAGCX_NET,
         "Retransmission enabled: RTO=%uus, MaxRetry=%d, AckInterval=%d",
         state->minRtoUs, state->maxRetry, state->ackInterval);
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbRetransDestroy(struct flagcxIbRetransState *state) {
  if (!state)
    return flagcxInternalError;

  if (state->enabled) {
    flagcxIbRetransPrintStats(state, "Final");
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbRetransAddPacket(struct flagcxIbRetransState *state,
                                        uint32_t seq, uint32_t size, void *data,
                                        uint64_t remote_addr, uint32_t *lkeys,
                                        uint32_t *rkeys) {

  if (!state || !state->enabled)
    return flagcxSuccess;

  if (state->bufferCount >= FLAGCX_IB_RETRANS_MAX_INFLIGHT) {
    static __thread int warnCount = 0;
    if (warnCount++ < 5) {
      WARN("Retransmission buffer full: count=%d, max=%d. "
           "send_seq=%u, send_una=%u, recv_seq=%u, total_sent=%lu, "
           "total_acked=%lu",
           state->bufferCount, FLAGCX_IB_RETRANS_MAX_INFLIGHT, state->sendSeq,
           state->sendUna, state->recvSeq, (unsigned long)state->totalSent,
           (unsigned long)state->totalAcked);
      WARN("This usually means ACK is not being received properly.");
      WARN("Dropping oldest packet to avoid deadlock (seq=%u)",
           state->buffer[state->bufferHead].seq);
    }

    state->buffer[state->bufferHead].valid = 0;
    state->bufferHead =
        (state->bufferHead + 1) % FLAGCX_IB_RETRANS_MAX_INFLIGHT;
    state->bufferCount--;
  }

  if (state->bufferCount > FLAGCX_IB_RETRANS_MAX_INFLIGHT * 3 / 4) {
    static __thread uint64_t lastWarnTime = 0;
    uint64_t now = flagcxIbGetTimeUs();
    if (now - lastWarnTime > 1000000) {
      WARN("Retransmission buffer %d%% full (%d/%d). ACK may not be working "
           "properly.",
           (state->bufferCount * 100) / FLAGCX_IB_RETRANS_MAX_INFLIGHT,
           state->bufferCount, FLAGCX_IB_RETRANS_MAX_INFLIGHT);
      lastWarnTime = now;
    }
  }

  int idx = state->bufferTail;
  struct flagcxIbRetransEntry *entry = &state->buffer[idx];

  entry->seq = seq;
  entry->size = size;
  entry->sendTimeUs = flagcxIbGetTimeUs();
  entry->remoteAddr = remote_addr;
  entry->data = data;

  if (lkeys) {
    memcpy(entry->lkeys, lkeys, sizeof(entry->lkeys));
  }
  if (rkeys) {
    memcpy(entry->rkeys, rkeys, sizeof(entry->rkeys));
  }

  entry->retryCount = 0;
  entry->valid = 1;

  state->bufferTail = (state->bufferTail + 1) % FLAGCX_IB_RETRANS_MAX_INFLIGHT;
  state->bufferCount++;
  state->totalSent++;

  return flagcxSuccess;
}

static void flagcxIbUpdateRTO(struct flagcxIbRetransState *state,
                              uint64_t rtt_us) {
  if (state->srttUs == 0) {
    state->srttUs = rtt_us;
    state->rttvarUs = rtt_us / 2;
  } else {
    // RFC 6298: SRTT = (1 - alpha) * SRTT + alpha * RTT
    // RTTVAR = (1 - beta) * RTTVAR + beta * |SRTT - RTT|
    // alpha = 1/8, beta = 1/4
    uint64_t delta = (rtt_us > state->srttUs) ? (rtt_us - state->srttUs)
                                              : (state->srttUs - rtt_us);
    state->rttvarUs = (3 * state->rttvarUs + delta) / 4;
    state->srttUs = (7 * state->srttUs + rtt_us) / 8;
  }

  state->rtoUs = state->srttUs + 4 * state->rttvarUs;

  if (state->rtoUs < state->minRtoUs) {
    state->rtoUs = state->minRtoUs;
  }
  if (state->rtoUs > state->maxRtoUs) {
    state->rtoUs = state->maxRtoUs;
  }

  TRACE(FLAGCX_NET,
        "Updated RTO: RTT=%luus, SRTT=%luus, RTTVAR=%luus, RTO=%luus",
        (unsigned long)rtt_us, (unsigned long)state->srttUs,
        (unsigned long)state->rttvarUs, (unsigned long)state->rtoUs);
}

flagcxResult_t flagcxIbRetransProcessAck(struct flagcxIbRetransState *state,
                                         struct flagcxIbAckMsg *ack_msg) {

  if (!state || !state->enabled || !ack_msg)
    return flagcxSuccess;

  uint32_t ackSeq = ack_msg->ackSeq;
  uint64_t nowUs = flagcxIbGetTimeUs();

  TRACE(FLAGCX_NET, "Processing ACK: ack_seq=%u, send_una=%u, buffer_count=%d",
        ackSeq, state->sendUna, state->bufferCount);

  if (ack_msg->timestampUs > 0) {
    uint64_t rttUs = nowUs - ack_msg->timestampUs;
    if (rttUs < 10000000) {
      flagcxIbUpdateRTO(state, rttUs);
    }
  }

  state->lastAckTimeUs = nowUs;

  int freed = 0;
  while (state->bufferCount > 0) {
    struct flagcxIbRetransEntry *entry = &state->buffer[state->bufferHead];

    if (!entry->valid) {
      state->bufferHead =
          (state->bufferHead + 1) % FLAGCX_IB_RETRANS_MAX_INFLIGHT;
      state->bufferCount--;
    } else if (flagcxIbSeqLeq(entry->seq, ackSeq)) {
      entry->valid = 0;
      state->bufferHead =
          (state->bufferHead + 1) % FLAGCX_IB_RETRANS_MAX_INFLIGHT;
      state->bufferCount--;
      state->totalAcked++;
      freed++;

      state->sendUna = entry->seq + 1;
    } else {
      break;
    }
  }

  uint64_t sackBitmap = ack_msg->sackBitmap;
  uint16_t sackCount = ack_msg->sackBitmapCount;

  if (sackBitmap != 0 && sackCount > 0) {
    TRACE(FLAGCX_NET, "Processing SACK: bitmap=0x%lx, count=%u",
          (unsigned long)sackBitmap, sackCount);

    int idx = state->bufferHead;
    for (int i = 0; i < state->bufferCount && i < 64; i++) {
      struct flagcxIbRetransEntry *entry = &state->buffer[idx];

      if (entry->valid) {
        uint32_t entryOffset = entry->seq - state->sendUna;

        if (entryOffset < 64) {
          if (sackBitmap & (1ULL << entryOffset)) {
            TRACE(FLAGCX_NET, "SACK confirmed packet: seq=%u", entry->seq);
            entry->valid = 0;
            state->totalAcked++;
            freed++;
          }
        }
      }

      idx = (idx + 1) % FLAGCX_IB_RETRANS_MAX_INFLIGHT;
    }

    while (state->bufferCount > 0) {
      struct flagcxIbRetransEntry *entry = &state->buffer[state->bufferHead];
      if (!entry->valid) {
        state->bufferHead =
            (state->bufferHead + 1) % FLAGCX_IB_RETRANS_MAX_INFLIGHT;
        state->bufferCount--;
      } else {
        break;
      }
    }
  }

  if (freed > 0) {
    TRACE(FLAGCX_NET,
          "ACK processed: freed %d packets, remaining=%d, ack_seq=%u", freed,
          state->bufferCount, ackSeq);
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbRetransResendViaSend(struct flagcxIbSendComm *comm,
                                            uint32_t seq) {

  if (!comm || !comm->retrans.enabled)
    return flagcxSuccess;

  // Check if retrans_hdr_mr is initialized (required for retransmission)
  if (!comm->retransHdrMr) {
    // retrans_hdr_mr not initialized, likely due to initialization failure
    // Disable retransmission to prevent further attempts
    comm->retrans.enabled = 0;
    TRACE(FLAGCX_NET,
          "retrans_hdr_mr is NULL, disabling retransmission for seq=%u", seq);
    return flagcxSuccess;
  }

  struct flagcxIbRetransEntry *entry = NULL;
  int idx = comm->retrans.bufferHead;
  for (int i = 0; i < comm->retrans.bufferCount; i++) {
    if (comm->retrans.buffer[idx].valid &&
        comm->retrans.buffer[idx].seq == seq) {
      entry = &comm->retrans.buffer[idx];
      break;
    }
    idx = (idx + 1) % FLAGCX_IB_RETRANS_MAX_INFLIGHT;
  }

  if (!entry) {
    TRACE(FLAGCX_NET, "Retrans: seq=%u not found in buffer (already acked?)",
          seq);
    return flagcxSuccess;
  }

  // Validate entry data
  if (!entry->data || entry->size == 0) {
    WARN("Invalid entry data for retrans seq=%u: data=%p, size=%u", seq,
         entry->data, entry->size);
    return flagcxInternalError;
  }

  // Flow control: Check if we have room for retransmission
  int totalOutstanding = comm->outstandingSends + comm->outstandingRetrans;
  if (totalOutstanding >= comm->maxOutstanding) {
    TRACE(FLAGCX_NET,
          "Retrans deferred: outstanding=%d (sends=%d + retrans=%d) >= max=%d",
          totalOutstanding, comm->outstandingSends, comm->outstandingRetrans,
          comm->maxOutstanding);
    return flagcxSuccess; // Defer retransmission, will retry later
  }

  int qpIndex = comm->retrans.retransQPIndex % comm->base.nqps;
  comm->retrans.retransQPIndex++;

  struct flagcxIbQp *qp = &comm->base.qps[qpIndex];

  // Validate QP
  if (!qp || !qp->qp) {
    WARN("Invalid QP for retrans seq=%u, qpIndex=%d", seq, qpIndex);
    return flagcxInternalError;
  }

  int devIndex = qp->devIndex;

  // Validate devIndex to prevent out-of-bounds access
  if (devIndex < 0 || devIndex >= comm->base.ndevs) {
    WARN("Invalid devIndex=%d (ndevs=%d) for retrans seq=%u, qpIndex=%d",
         devIndex, comm->base.ndevs, seq, qpIndex);
    return flagcxInternalError;
  }

  // retrans_hdr_mr is already validated at function entry

  struct ibv_send_wr wr;
  struct ibv_sge sge[2];
  memset(&wr, 0, sizeof(wr));

  struct flagcxIbRetransHdr *hdr = &comm->retransHdrPool[seq % 32];
  hdr->magic = FLAGCX_RETRANS_MAGIC;
  hdr->seq = seq;
  hdr->size = entry->size;
  hdr->remoteAddr = entry->remoteAddr;

  sge[0].addr = (uint64_t)hdr;
  sge[0].length = sizeof(struct flagcxIbRetransHdr);
  sge[0].lkey = comm->retransHdrMr->lkey;

  sge[1].addr = (uint64_t)entry->data;
  sge[1].length = entry->size;
  sge[1].lkey = entry->lkeys[devIndex];

  wr.wr_id = FLAGCX_RETRANS_WR_ID;
  wr.sg_list = sge;
  wr.num_sge = 2;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.next = NULL;

  struct ibv_send_wr *badWr;
  flagcxResult_t result = flagcxWrapIbvPostSend(qp->qp, &wr, &badWr);

  if (result != flagcxSuccess) {
    // Send queue might be full, try polling completion queue to free up WRs
    // devIndex is already validated above
    struct flagcxIbNetCommDevBase *devBase = &comm->devs[devIndex].base;
    if (devBase && devBase->cq) {
      struct ibv_wc wcs[64];
      int nCqe = 0;
      int totalPolled = 0;

      // Aggressively poll completions to free up send queue space
      for (int poll_iter = 0; poll_iter < 8; poll_iter++) {
        flagcxWrapIbvPollCq(devBase->cq, 64, wcs, &nCqe);
        totalPolled += nCqe;
        if (nCqe == 0)
          break;
      }

      if (totalPolled > 0) {
        TRACE(FLAGCX_NET,
              "Retrans SEND failed, polled %d completions to free WRs (seq=%u)",
              totalPolled, seq);

        // Update outstanding_retrans for retrans completions only
        // Regular sends will be handled by normal completion processing
        for (int i = 0; i < totalPolled; i++) {
          if (wcs[i].wr_id == FLAGCX_RETRANS_WR_ID &&
              wcs[i].status == IBV_WC_SUCCESS) {
            comm->outstandingRetrans--;
          }
        }

        // Retry sending after polling
        result = flagcxWrapIbvPostSend(qp->qp, &wr, &badWr);
        if (result == flagcxSuccess) {
          TRACE(FLAGCX_NET,
                "Retrans SEND succeeded after polling completions (seq=%u)",
                seq);
        }
      }
    }

    if (result != flagcxSuccess) {
      TRACE(FLAGCX_NET,
            "Retrans SEND failed for seq=%u, qp=%d after polling (deferring)",
            seq, qpIndex);
      // Return success to defer retransmission, will retry later when queue has
      // space
      return flagcxSuccess;
    }
  }

  entry->retryCount++;
  entry->sendTimeUs = flagcxIbGetTimeUs();
  comm->retrans.totalRetrans++;
  comm->outstandingRetrans++;

  return flagcxSuccess;
}

flagcxResult_t flagcxIbRetransCheckTimeout(struct flagcxIbRetransState *state,
                                           struct flagcxIbSendComm *comm) {

  if (!state || !state->enabled || !comm)
    return flagcxSuccess;
  if (state->bufferCount == 0)
    return flagcxSuccess;

  uint64_t nowUs = flagcxIbGetTimeUs();
  int retransCount = 0;
  const int MAX_RETRANS_PER_CALL = 1;

  int idx = state->bufferHead;
  for (int i = 0; i < state->bufferCount && retransCount < MAX_RETRANS_PER_CALL;
       i++) {
    struct flagcxIbRetransEntry *entry = &state->buffer[idx];

    if (entry->valid) {
      uint64_t elapsedUs = nowUs - entry->sendTimeUs;

      if (elapsedUs >= state->rtoUs) {
        if (entry->retryCount >= state->maxRetry) {
          WARN("Packet exceeded max retries: seq=%u, retry=%d, max=%d. "
               "Transmission failed, aborting operation.",
               entry->seq, entry->retryCount, state->maxRetry);
          return flagcxRemoteError;
        }

        flagcxResult_t retransResult =
            flagcxIbRetransResendViaSend(comm, entry->seq);
        if (retransResult == flagcxSuccess) {
          state->totalTimeout++;
          retransCount++;

          state->rtoUs = (state->rtoUs * 2 > state->maxRtoUs)
                             ? state->maxRtoUs
                             : state->rtoUs * 2;
        } else {
          break;
        }
      }
    }

    idx = (idx + 1) % FLAGCX_IB_RETRANS_MAX_INFLIGHT;
  }

  if (retransCount > 0) {
    TRACE(FLAGCX_NET, "Retransmitted %d packets, RTO=%luus, pending=%d",
          retransCount, (unsigned long)state->rtoUs, state->bufferCount);
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbRetransRecvPacket(struct flagcxIbRetransState *state,
                                         uint32_t seq,
                                         struct flagcxIbAckMsg *ack_msg,
                                         int *should_ack) {

  if (!state || !state->enabled || !ack_msg || !should_ack) {
    *should_ack = 0;
    return flagcxSuccess;
  }

  *should_ack = 0;
  uint64_t nowUs = flagcxIbGetTimeUs();

  const uint32_t ack_interval =
      (state->ackInterval > 0) ? (uint32_t)state->ackInterval : 1;

  if (seq == state->recvSeq) {
    state->recvSeq = (state->recvSeq + 1) & 0xFFFF;

    uint16_t delta = (uint16_t)((state->recvSeq - state->lastAckSeq) & 0xFFFF);
    if (delta >= ack_interval || nowUs - state->lastAckSendTimeUs >= 1000 ||
        state->recvSeq == 1) {
      *should_ack = 1;
    }
  } else if (flagcxIbSeqLess(seq, state->recvSeq)) {
    *should_ack = 1;
  } else {
    *should_ack = 1;

    int gap = seq - state->recvSeq;
    if (gap > 0 && gap < 64) {
      ack_msg->sackBitmap |= (1ULL << (gap - 1));

      ack_msg->sackBitmapCount = 0;
      uint64_t bitmap = ack_msg->sackBitmap;
      while (bitmap) {
        ack_msg->sackBitmapCount += (bitmap & 1);
        bitmap >>= 1;
      }

      TRACE(FLAGCX_NET, "SACK: gap=%d, bitmap=0x%lx, count=%u", gap,
            (unsigned long)ack_msg->sackBitmap, ack_msg->sackBitmapCount);
    }
  }

  if (*should_ack) {
    ack_msg->ackSeq = (state->recvSeq - 1) & 0xFFFF;
    ack_msg->timestampUs = nowUs;
    ack_msg->peerId = 0;
    ack_msg->flowId = 0;
    ack_msg->path = 0;
    state->lastAckSeq = state->recvSeq;
    state->lastAckSendTimeUs = nowUs;
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbRetransPiggybackAck(struct flagcxIbSendFifo *fifo_elem,
                                           struct flagcxIbAckMsg *ack_msg) {
  return flagcxSuccess;
}

flagcxResult_t flagcxIbRetransExtractAck(struct flagcxIbSendFifo *fifo_elem,
                                         struct flagcxIbAckMsg *ack_msg) {
  return flagcxSuccess;
}

void flagcxIbRetransPrintStats(struct flagcxIbRetransState *state,
                               const char *prefix) {
  if (!state || !state->enabled)
    return;

  // Statistics printing disabled
}

flagcxResult_t flagcxIbRetransSendAckViaUd(struct flagcxIbRecvComm *comm,
                                           struct flagcxIbAckMsg *ack_msg,
                                           int devIndex) {

  if (!comm || !ack_msg || devIndex >= comm->base.ndevs) {
    WARN("Invalid parameters for sending ACK: comm=%p, ack_msg=%p, "
         "devIndex=%d, ndevs=%d",
         comm, ack_msg, devIndex, comm ? comm->base.ndevs : -1);
    return flagcxInternalError;
  }

  struct flagcxIbRecvCommDev *commDev = &comm->devs[devIndex];
  struct flagcxIbCtrlQp *ctrlQp = &commDev->ctrlQp;

  if (!ctrlQp->qp || !ctrlQp->ah) {
    WARN("Control QP not initialized: qp=%p, ah=%p", ctrlQp->qp, ctrlQp->ah);
    return flagcxInternalError;
  }

  static __thread int ackSendCountDebug = 0;
  if ((ackSendCountDebug++ % 100) == 0) {
    TRACE(FLAGCX_NET,
          "Sending ACK #%d: ack_seq=%u, sack_bitmap=0x%lx, remote_qpn=%u",
          ackSendCountDebug, ack_msg->ackSeq,
          (unsigned long)ack_msg->sackBitmap, ctrlQp->remoteQpn);
  }

  // Aggressively poll completions to free up send queue space
  int totalPolled = 0;
  for (int poll_iter = 0; poll_iter < 16; poll_iter++) {
    struct ibv_wc wcs[64];
    int nCqe = 0;
    flagcxWrapIbvPollCq(ctrlQp->cq, 64, wcs, &nCqe);
    totalPolled += nCqe;
    if (nCqe == 0)
      break;
  }

  if (totalPolled > 0) {
    TRACE(FLAGCX_NET, "Polled %d ACK completions before sending", totalPolled);
  }

  struct flagcxIbAckMsg *ackBuf = (struct flagcxIbAckMsg *)commDev->ackBuffer;
  if (!ackBuf) {
    WARN("IBUC Retrans: ackBuffer is NULL for devIndex=%d", devIndex);
    return flagcxInternalError;
  }

  memcpy(ackBuf, ack_msg, sizeof(struct flagcxIbAckMsg));

  struct ibv_sge sge;
  sge.addr = (uint64_t)ackBuf;
  sge.length = sizeof(struct flagcxIbAckMsg);
  sge.lkey = commDev->ackMr->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.next = NULL;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;

  wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;

  wr.wr.ud.ah = ctrlQp->ah;
  wr.wr.ud.remote_qpn = ctrlQp->remoteQpn;
  wr.wr.ud.remote_qkey = ctrlQp->remoteQkey;

  struct ibv_send_wr *badWr;
  flagcxResult_t result = flagcxWrapIbvPostSend(ctrlQp->qp, &wr, &badWr);

  // If send queue is full, try more aggressive polling and retry
  if (result != flagcxSuccess) {
    // More aggressive polling with multiple rounds
    int retryPolled = 0;
    for (int retry_round = 0; retry_round < 3; retry_round++) {
      for (int i = 0; i < 8; i++) {
        struct ibv_wc wcs[64];
        int n = 0;
        flagcxWrapIbvPollCq(ctrlQp->cq, 64, wcs, &n);
        retryPolled += n;
        if (n == 0)
          break;
      }

      // Try sending again after polling
      result = flagcxWrapIbvPostSend(ctrlQp->qp, &wr, &badWr);
      if (result == flagcxSuccess) {
        break;
      }

      // Small delay to allow hardware to process completions
      if (retry_round < 2) {
        usleep(10); // 10 microseconds
      }
    }

    if (result != flagcxSuccess) {
      // ACK is periodic, so it's acceptable to drop this one if queue is full
      // The next ACK will be sent soon anyway
      return flagcxSuccess; // Return success to avoid breaking the flow
    }
  }

  TRACE(FLAGCX_NET, "Sent ACK via UD: ack_seq=%u, sack_count=%u",
        ack_msg->ackSeq, ack_msg->sackBitmapCount);

  return flagcxSuccess;
}

flagcxResult_t flagcxIbRetransRecvAckViaUd(struct flagcxIbSendComm *comm,
                                           int devIndex) {

  if (!comm || devIndex >= comm->base.ndevs) {
    return flagcxInternalError;
  }

  struct flagcxIbSendCommDev *commDev = &comm->devs[devIndex];
  struct flagcxIbCtrlQp *ctrlQp = &commDev->ctrlQp;

  if (!ctrlQp->qp || !ctrlQp->cq) {
    TRACE(FLAGCX_NET, "IBUC Retrans: RecvAckViaUd QP not ready qp=%p, cq=%p",
          ctrlQp->qp, ctrlQp->cq);
    return flagcxInternalError;
  }

  struct ibv_wc wc;
  int nCqe = 0;
  flagcxResult_t result = flagcxWrapIbvPollCq(ctrlQp->cq, 1, &wc, &nCqe);

  if (result != flagcxSuccess || nCqe == 0) {
    return flagcxSuccess;
  }

  if (wc.status != IBV_WC_SUCCESS) {
    WARN("ACK completion with error: status=%d", wc.status);
    return flagcxRemoteError;
  }

  int bufIdx = (int)wc.wr_id;
  size_t bufEntrySize =
      sizeof(struct flagcxIbAckMsg) + FLAGCX_IB_ACK_BUF_PADDING;
  char *bufBase = (char *)commDev->ackBuffer + bufIdx * bufEntrySize;
  struct flagcxIbAckMsg *ackMsg =
      (struct flagcxIbAckMsg *)(bufBase + FLAGCX_IB_ACK_BUF_PADDING);

  FLAGCXCHECK(flagcxIbRetransProcessAck(&comm->retrans, ackMsg));

  struct ibv_sge sge;
  sge.addr = (uint64_t)bufBase;
  sge.length = bufEntrySize;
  sge.lkey = commDev->ackMr->lkey;

  struct ibv_recv_wr recvWr;
  memset(&recvWr, 0, sizeof(recvWr));
  recvWr.wr_id = bufIdx;
  recvWr.next = NULL;
  recvWr.sg_list = &sge;
  recvWr.num_sge = 1;

  struct ibv_recv_wr *badWr;
  FLAGCXCHECK(flagcxWrapIbvPostRecv(ctrlQp->qp, &recvWr, &badWr));

  TRACE(FLAGCX_NET, "Re-posted recv WR for ACK buffer %d", bufIdx);

  return flagcxSuccess;
}
