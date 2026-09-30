/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_COLL_PROXY_TRANSPORT_H_
#define FLAGCX_COLL_PROXY_TRANSPORT_H_

#include "net_transport.h"

#define FLAGCX_COLL_PROXY_MAX_STEPS 16
#define FLAGCX_COLL_PROXY_MAX_CHANNELS 32

// Completion state owned by one collective proxy operation. The proxy may
// poll requests in any order, but only a contiguous completed prefix is
// retired back to the collective step/FIFO.
struct flagcxCollProxyTransport {
  struct flagcxNetCompletionScoreboard scoreboard;
  struct flagcxNetCompletionEntry entries[FLAGCX_COLL_PROXY_MAX_STEPS];
  struct flagcxNetSubmitContext contexts[FLAGCX_COLL_PROXY_MAX_STEPS];
  uint64_t orderingKey;
  uint64_t generation;
  uint64_t nextSubmit;
  uint64_t *laneMask;
  uint32_t submitFlags;
  uint32_t capacity;
  uint32_t initialized;
};

// A transport ordering key is stable for the lifetime of a
// (channel, peer, direction) domain. Direction is canonical wire direction:
// send is src->dst on both endpoints.
uint64_t flagcxCollProxyOrderingKey(int channelId, int srcRank, int dstRank);

// Select one of a small, fixed set of internal channels. Channel zero is
// reserved for the public point-to-point compatibility path. Both endpoints
// pass canonical src/dst and therefore derive the same connection.
int flagcxCollProxyChannelForEdge(int srcRank, int dstRank, uint32_t phase);

flagcxResult_t
flagcxCollProxyTransportInit(struct flagcxCollProxyTransport *transport,
                             uint32_t capacity, uint64_t generation,
                             uint64_t orderingKey, uint32_t submitFlags);

// Reserve the next sequence before calling an adaptor. If the adaptor applies
// backpressure without accepting the request, cancel the reservation.
flagcxResult_t
flagcxCollProxyTrackNext(struct flagcxCollProxyTransport *transport,
                         struct flagcxNetSubmitContext **context);
flagcxResult_t
flagcxCollProxyCancel(struct flagcxCollProxyTransport *transport,
                      const struct flagcxNetSubmitContext *context);
flagcxResult_t
flagcxCollProxyComplete(struct flagcxCollProxyTransport *transport,
                        const struct flagcxNetSubmitContext *context,
                        flagcxResult_t result, uint32_t *advanced);

#endif // FLAGCX_COLL_PROXY_TRANSPORT_H_
