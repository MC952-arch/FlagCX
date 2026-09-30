/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "coll_proxy_transport.h"

#include <string.h>

uint64_t flagcxCollProxyOrderingKey(int channelId, int srcRank, int dstRank) {
  // Rank fields are deliberately wider than current communicator limits. The
  // seed bit keeps the packed identity non-zero before it is mixed.
  uint64_t value = (uint64_t)((uint32_t)channelId & 0x1fU) << 49 |
                   (uint64_t)((uint32_t)srcRank & 0xffffffU) << 25 |
                   (uint64_t)((uint32_t)dstRank & 0xffffffU) << 1 | 1ULL;
  // Mix the packed identity before modulo lane selection. Rank/channel bits
  // otherwise leave the low bit constant and collapse every two-QP domain
  // onto the same lane.
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value == 0 ? 1 : value;
}

int flagcxCollProxyChannelForEdge(int srcRank, int dstRank, uint32_t phase) {
  if (FLAGCX_COLL_PROXY_MAX_CHANNELS <= 1)
    return 0;
  const uint32_t internalChannels = FLAGCX_COLL_PROXY_MAX_CHANNELS - 1 < 4
                                        ? FLAGCX_COLL_PROXY_MAX_CHANNELS - 1
                                        : 4;
  uint64_t value = (uint64_t)(uint32_t)srcRank * 0x9e3779b185ebca87ULL;
  value ^= (uint64_t)(uint32_t)dstRank * 0xc2b2ae3d27d4eb4fULL;
  value ^= (uint64_t)phase * 0x165667b19e3779f9ULL;
  value ^= value >> 33;
  return 1 + (int)(value % internalChannels);
}

flagcxResult_t
flagcxCollProxyTransportInit(struct flagcxCollProxyTransport *transport,
                             uint32_t capacity, uint64_t generation,
                             uint64_t orderingKey, uint32_t submitFlags) {
  if (transport == NULL || capacity == 0 ||
      capacity > FLAGCX_COLL_PROXY_MAX_STEPS)
    return flagcxInvalidArgument;
  memset(transport, 0, sizeof(*transport));
  transport->orderingKey = orderingKey;
  transport->generation = generation;
  transport->capacity = capacity;
  transport->submitFlags = submitFlags | FLAGCX_NET_SUBMIT_DATA;
  flagcxResult_t result = flagcxNetCompletionScoreboardInit(
      &transport->scoreboard, transport->entries, capacity, generation, 0);
  if (result != flagcxSuccess)
    return result;
  transport->initialized = 1;
  return flagcxSuccess;
}

flagcxResult_t
flagcxCollProxyTrackNext(struct flagcxCollProxyTransport *transport,
                         struct flagcxNetSubmitContext **context) {
  if (transport == NULL || context == NULL || !transport->initialized)
    return flagcxInvalidArgument;
  const uint64_t sequence = transport->nextSubmit;
  struct flagcxNetSubmitContext candidate = {
      transport->orderingKey, 0,
      transport->generation,  sequence,
      transport->submitFlags, transport->laneMask};
  flagcxResult_t result =
      flagcxNetTrackSubmit(&transport->scoreboard, &candidate, NULL);
  if (result != flagcxSuccess)
    return result;
  struct flagcxNetSubmitContext *next =
      &transport->contexts[sequence % transport->capacity];
  *next = candidate;
  transport->nextSubmit++;
  *context = next;
  return flagcxSuccess;
}

flagcxResult_t
flagcxCollProxyCancel(struct flagcxCollProxyTransport *transport,
                      const struct flagcxNetSubmitContext *context) {
  if (transport == NULL || context == NULL || !transport->initialized ||
      transport->nextSubmit == 0 ||
      context->sequence + 1 != transport->nextSubmit)
    return flagcxInvalidArgument;
  flagcxResult_t result = flagcxNetTrackCancel(&transport->scoreboard, context);
  if (result != flagcxSuccess)
    return result;
  transport->nextSubmit--;
  return flagcxSuccess;
}

flagcxResult_t
flagcxCollProxyComplete(struct flagcxCollProxyTransport *transport,
                        const struct flagcxNetSubmitContext *context,
                        flagcxResult_t result, uint32_t *advanced) {
  if (transport == NULL || context == NULL || advanced == NULL ||
      !transport->initialized)
    return flagcxInvalidArgument;
  return flagcxNetTrackCompletion(&transport->scoreboard, context, result,
                                  advanced);
}
