/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * All rights reserved.
 *
 * IBUC Retransmission Support - Header
 ************************************************************************/

#ifndef FLAGCX_IBUC_RETRANS_H_
#define FLAGCX_IBUC_RETRANS_H_

#include "flagcx_common.h"
#include "ib_common.h"
#include <stdint.h>
#include <time.h>

// Retransmission constants
#define FLAGCX_RETRANS_MAGIC                                                   \
  0xDEADBEEF // Magic number for retransmission header
#define FLAGCX_RETRANS_WR_ID                                                   \
  0xFFFFFFFEULL // WR ID for retransmission completions
#ifdef USE_IBUC
#define FLAGCX_IBUC_DATA_RECV_WR_ID_PREFIX 0x4000000000000000ULL
#define FLAGCX_IBUC_RETRANS_RECV_WR_ID_PREFIX 0x8000000000000000ULL
#define FLAGCX_IB_RETRANS_SEQ_MASK 0x0FFFu
#define FLAGCX_IBUC_GENERATION_MASK 0x0FFFu
#else
#define FLAGCX_IB_RETRANS_SEQ_MASK 0xFFFFu
#endif

extern int64_t flagcxParamIbRetransEnable(void);
extern int64_t flagcxParamIbRetransTimeout(void);
extern int64_t flagcxParamIbRetransMaxRetry(void);
extern int64_t flagcxParamIbRetransAckInterval(void);

// The retransmission ACK channel requires UD address-handle support. The
// payload retransmission itself uses a dedicated RC QP in IBUC; legacy IBRC
// retransmission may additionally use SRQ operations.
bool flagcxIbRetransUdSupported(void);
extern int64_t flagcxParamIbMaxOutstanding(void);

static inline uint64_t flagcxIbGetTimeUs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static inline int flagcxIbSeqLess(uint32_t a, uint32_t b) {
  uint32_t diff = (a - b) & FLAGCX_IB_RETRANS_SEQ_MASK;
  return diff != 0 && diff > (FLAGCX_IB_RETRANS_SEQ_MASK >> 1);
}

static inline int flagcxIbSeqLeq(uint32_t a, uint32_t b) {
  return a == b || flagcxIbSeqLess(a, b);
}

flagcxResult_t flagcxIbRetransInit(struct flagcxIbRetransState *state);

flagcxResult_t flagcxIbRetransDestroy(struct flagcxIbRetransState *state);

flagcxResult_t flagcxIbRetransAddPacket(struct flagcxIbRetransState *state,
                                        uint32_t seq, uint32_t size, void *data,
                                        uint64_t remote_addr, uint32_t *lkeys,
                                        uint32_t *rkeys);

#ifdef USE_IBUC
flagcxResult_t flagcxIbRetransAddBatch(struct flagcxIbRetransState *state,
                                       uint32_t seq, uint8_t remoteRequestSlot,
                                       uint16_t remoteGeneration,
                                       uint8_t ackDevIndex, int nreqs,
                                       struct flagcxIbRequest **reqs);
#endif

flagcxResult_t flagcxIbRetransProcessAck(struct flagcxIbRetransState *state,
                                         struct flagcxIbAckMsg *ack_msg);

flagcxResult_t flagcxIbRetransCheckTimeout(struct flagcxIbRetransState *state,
                                           struct flagcxIbSendComm *comm);

flagcxResult_t flagcxIbRetransRecvPacket(struct flagcxIbRetransState *state,
                                         uint32_t seq,
                                         struct flagcxIbAckMsg *ack_msg,
                                         int *should_ack);

flagcxResult_t flagcxIbRetransPiggybackAck(struct flagcxIbSendFifo *fifo_elem,
                                           struct flagcxIbAckMsg *ack_msg);

flagcxResult_t flagcxIbRetransExtractAck(struct flagcxIbSendFifo *fifo_elem,
                                         struct flagcxIbAckMsg *ack_msg);

static inline uint32_t flagcxIbEncodeImmData(uint32_t seq, uint32_t size) {
  return ((seq & 0xFFFF) << 16) | (size & 0xFFFF);
}

static inline void flagcxIbDecodeImmData(uint32_t imm_data, uint32_t *seq,
                                         uint32_t *size) {
  *seq = (imm_data >> 16) & 0xFFFF;
  *size = imm_data & 0xFFFF;
}

#ifdef USE_IBUC
static inline uint32_t flagcxIbucEncodeImmData(uint32_t seq,
                                               uint8_t requestSlot,
                                               uint16_t generation) {
  return ((seq & FLAGCX_IB_RETRANS_SEQ_MASK) << 20) |
         ((generation & FLAGCX_IBUC_GENERATION_MASK) << 8) | requestSlot;
}

static inline void flagcxIbucDecodeImmData(uint32_t immData, uint32_t *seq,
                                           uint8_t *requestSlot,
                                           uint16_t *generation) {
  *seq = (immData >> 20) & FLAGCX_IB_RETRANS_SEQ_MASK;
  *generation = (immData >> 8) & FLAGCX_IBUC_GENERATION_MASK;
  *requestSlot = immData & 0xFF;
}

static inline bool
flagcxIbucRequestMatchesGeneration(const struct flagcxIbRequest *request,
                                   uint16_t generation) {
  return request != NULL && request->type == FLAGCX_NET_IB_REQ_RECV &&
         request->retransGeneration ==
             (generation & FLAGCX_IBUC_GENERATION_MASK);
}
#endif

void flagcxIbRetransPrintStats(struct flagcxIbRetransState *state,
                               const char *prefix);

flagcxResult_t flagcxIbCreateCtrlQp(struct ibv_context *context,
                                    struct ibv_pd *pd, uint8_t port_num,
                                    struct flagcxIbCtrlQp *ctrlQp);

flagcxResult_t flagcxIbDestroyCtrlQp(struct flagcxIbCtrlQp *ctrlQp);

flagcxResult_t
flagcxIbSetupCtrlQpConnection(struct ibv_context *context, struct ibv_pd *pd,
                              struct flagcxIbCtrlQp *ctrlQp,
                              uint32_t remote_qpn, union ibv_gid *remote_gid,
                              uint32_t remote_lid, uint8_t port_num,
                              uint8_t link_layer, uint8_t local_gid_index);

flagcxResult_t flagcxIbRetransSendAckViaUd(struct flagcxIbRecvComm *comm,
                                           struct flagcxIbAckMsg *ack_msg,
                                           int devIndex);

flagcxResult_t flagcxIbRetransRecvAckViaUd(struct flagcxIbSendComm *comm,
                                           int devIndex);

flagcxResult_t flagcxIbRetransResendViaSend(struct flagcxIbSendComm *comm,
                                            uint32_t seq);

flagcxResult_t flagcxIbCreateSrq(struct ibv_context *context, struct ibv_pd *pd,
                                 struct flagcxIbSrqMgr *srqMgr);

flagcxResult_t flagcxIbDestroySrq(struct flagcxIbSrqMgr *srqMgr);

flagcxResult_t flagcxIbSrqPostRecv(struct flagcxIbSrqMgr *srqMgr, int count);

#endif // FLAGCX_IBUC_RETRANS_H_
