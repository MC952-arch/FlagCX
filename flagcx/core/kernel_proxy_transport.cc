/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "kernel_proxy_transport.h"

#include "device_api/completion_word.h"
#include "device_api/fifo_producer_gate.h"
#include "flagcx_kernel_core.h"

#include <sched.h>
#include <stdlib.h>
#include <string.h>

static void
flagcxKernelProxyRequestReset(struct flagcxKernelProxyRequest *request) {
  memset(request, 0, sizeof(*request));
  request->completionResult = flagcxSuccess;
  request->peer = -1;
  request->stagingSlot = -1;
  request->state = FLAGCX_KERNEL_PROXY_REQUEST_FREE;
}

flagcxResult_t
flagcxKernelProxyTransportInit(struct flagcxKernelProxyTransport *transport,
                               uint32_t capacity, uint32_t stagingSlotCount,
                               uint64_t generation, uint64_t orderingKey) {
  if (transport == NULL || capacity == 0 || generation == 0)
    return flagcxInvalidArgument;

  memset(transport, 0, sizeof(*transport));
  transport->completionEntries = (struct flagcxNetCompletionEntry *)calloc(
      capacity, sizeof(struct flagcxNetCompletionEntry));
  transport->requests = (struct flagcxKernelProxyRequest *)calloc(
      capacity, sizeof(struct flagcxKernelProxyRequest));
  if (stagingSlotCount != 0)
    transport->stagingInUse =
        (uint8_t *)calloc(stagingSlotCount, sizeof(uint8_t));
  if (transport->completionEntries == NULL || transport->requests == NULL ||
      (stagingSlotCount != 0 && transport->stagingInUse == NULL)) {
    flagcxKernelProxyTransportDestroy(transport);
    return flagcxSystemError;
  }

  for (uint32_t i = 0; i < capacity; ++i)
    flagcxKernelProxyRequestReset(&transport->requests[i]);
  flagcxResult_t result = flagcxNetCompletionScoreboardInit(
      &transport->scoreboard, transport->completionEntries, capacity,
      generation, 1);
  if (result != flagcxSuccess) {
    flagcxKernelProxyTransportDestroy(transport);
    return result;
  }

  transport->capacity = capacity;
  transport->stagingSlotCount = stagingSlotCount;
  transport->generation = generation;
  transport->orderingKey = orderingKey;
  transport->nextSequence = 1;
  return flagcxSuccess;
}

void flagcxKernelProxyTransportDestroy(
    struct flagcxKernelProxyTransport *transport) {
  if (transport == NULL)
    return;
  free(transport->stagingInUse);
  free(transport->requests);
  free(transport->completionEntries);
  memset(transport, 0, sizeof(*transport));
}

flagcxResult_t flagcxKernelProxyCloseFifoProducerGate(uint64_t *fifoBuffer) {
  if (fifoBuffer == NULL)
    return flagcxInvalidArgument;
  flagcxCompletionWord_t *producerState =
      flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProducerState);
  const flagcxCompletionWord_t closedMask =
      flagcxFifoProducerClosedMask<flagcxCompletionWord_t>();
  const flagcxCompletionWord_t activeMask =
      flagcxFifoProducerActiveMask<flagcxCompletionWord_t>();
  __atomic_fetch_or(producerState, closedMask, __ATOMIC_ACQ_REL);
  while ((__atomic_load_n(producerState, __ATOMIC_ACQUIRE) & activeMask) != 0)
    sched_yield();
  return flagcxSuccess;
}

flagcxResult_t flagcxKernelProxyFinalizeTerminalFifo(uint64_t *fifoBuffer) {
  if (fifoBuffer == NULL)
    return flagcxInvalidArgument;
  flagcxCompletionWord_t *producerState =
      flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProducerState);
  const flagcxCompletionWord_t state =
      __atomic_load_n(producerState, __ATOMIC_ACQUIRE);
  const flagcxCompletionWord_t closedMask =
      flagcxFifoProducerClosedMask<flagcxCompletionWord_t>();
  const flagcxCompletionWord_t activeMask =
      flagcxFifoProducerActiveMask<flagcxCompletionWord_t>();
  if ((state & closedMask) == 0 || (state & activeMask) != 0)
    return flagcxInvalidUsage;

  flagcxCompletionWord_t produced =
      __atomic_load_n(flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProduced),
                      __ATOMIC_ACQUIRE);
  __atomic_store_n(flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxConsumed),
                   produced, __ATOMIC_RELEASE);
  __atomic_store_n(flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxCompleted),
                   produced, __ATOMIC_RELEASE);
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyTrackNext(struct flagcxKernelProxyTransport *transport,
                           uint32_t flags,
                           struct flagcxNetSubmitContext *submit) {
  if (transport == NULL || submit == NULL || transport->capacity == 0)
    return flagcxInvalidArgument;
  struct flagcxNetSubmitContext context = {};
  context.orderingKey = transport->orderingKey;
  context.generation = transport->generation;
  context.sequence = transport->nextSequence;
  context.flags = flags;
  flagcxResult_t result =
      flagcxNetTrackSubmit(&transport->scoreboard, &context, NULL);
  if (result != flagcxSuccess)
    return result;
  transport->nextSequence++;
  *submit = context;
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyReserveRequest(struct flagcxKernelProxyTransport *transport,
                                const struct flagcxNetSubmitContext *submit,
                                int peer, int stagingSlot, uint32_t *slot) {
  if (transport == NULL || submit == NULL || slot == NULL || peer < 0)
    return flagcxInvalidArgument;
  for (uint32_t i = 0; i < transport->capacity; ++i) {
    struct flagcxKernelProxyRequest *request = &transport->requests[i];
    if (request->state != FLAGCX_KERNEL_PROXY_REQUEST_FREE)
      continue;
    request->submit = *submit;
    request->peer = peer;
    request->stagingSlot = stagingSlot;
    request->state = FLAGCX_KERNEL_PROXY_REQUEST_RESERVED;
    *slot = i;
    return flagcxSuccess;
  }
  return flagcxInProgress;
}

flagcxResult_t
flagcxKernelProxyPublishRequest(struct flagcxKernelProxyTransport *transport,
                                uint32_t slot, void *request,
                                flagcxResult_t completionResult) {
  if (transport == NULL || request == NULL || slot >= transport->capacity)
    return flagcxInvalidArgument;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_RESERVED)
    return flagcxInvalidArgument;
  entry->request = request;
  entry->completionResult = completionResult;
  entry->state = FLAGCX_KERNEL_PROXY_REQUEST_POSTED;
  transport->nativeInflight++;
  return flagcxSuccess;
}

void flagcxKernelProxyCancelRequest(
    struct flagcxKernelProxyTransport *transport, uint32_t slot) {
  if (transport == NULL || slot >= transport->capacity)
    return;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state == FLAGCX_KERNEL_PROXY_REQUEST_RESERVED)
    flagcxKernelProxyRequestReset(entry);
}

flagcxResult_t
flagcxKernelProxyCompleteRequest(struct flagcxKernelProxyTransport *transport,
                                 uint32_t slot, flagcxResult_t result,
                                 uint32_t *advanced, int *releasedStagingSlot) {
  if (transport == NULL || advanced == NULL || releasedStagingSlot == NULL ||
      slot >= transport->capacity)
    return flagcxInvalidArgument;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_POSTED ||
      transport->nativeInflight == 0)
    return flagcxInvalidArgument;

  flagcxResult_t trackResult = flagcxNetTrackCompletion(
      &transport->scoreboard, &entry->submit, result, advanced);
  if (trackResult != flagcxSuccess)
    return trackResult;
  *releasedStagingSlot = entry->stagingSlot;
  transport->nativeInflight--;
  flagcxKernelProxyRequestReset(entry);
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyCompleteImmediate(struct flagcxKernelProxyTransport *transport,
                                   const struct flagcxNetSubmitContext *submit,
                                   flagcxResult_t result, uint32_t *advanced) {
  if (transport == NULL || submit == NULL || advanced == NULL)
    return flagcxInvalidArgument;
  return flagcxNetTrackCompletion(&transport->scoreboard, submit, result,
                                  advanced);
}

flagcxResult_t flagcxKernelProxyAcquireStagingSlot(
    struct flagcxKernelProxyTransport *transport, int *slot) {
  if (transport == NULL || slot == NULL || transport->stagingSlotCount == 0 ||
      transport->stagingInUse == NULL)
    return flagcxInvalidArgument;
  for (uint32_t i = 0; i < transport->stagingSlotCount; ++i) {
    uint32_t candidate =
        (transport->stagingCursor + i) % transport->stagingSlotCount;
    if (transport->stagingInUse[candidate] != 0)
      continue;
    transport->stagingInUse[candidate] = 1;
    transport->stagingCursor = (candidate + 1) % transport->stagingSlotCount;
    *slot = (int)candidate;
    return flagcxSuccess;
  }
  return flagcxInProgress;
}

flagcxResult_t flagcxKernelProxyReleaseStagingSlot(
    struct flagcxKernelProxyTransport *transport, int slot) {
  if (transport == NULL || slot < 0 ||
      (uint32_t)slot >= transport->stagingSlotCount ||
      transport->stagingInUse == NULL || transport->stagingInUse[slot] == 0)
    return flagcxInvalidArgument;
  transport->stagingInUse[slot] = 0;
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyQuery(struct flagcxKernelProxyTransport *transport,
                       uint64_t *nextSequence, uint32_t *inFlight,
                       flagcxResult_t *firstError) {
  if (transport == NULL)
    return flagcxInvalidArgument;
  return flagcxNetCompletionScoreboardQuery(&transport->scoreboard,
                                            nextSequence, inFlight, firstError);
}

flagcxResult_t
flagcxKernelProxyReleaseReady(struct flagcxKernelProxyTransport *transport,
                              const struct flagcxNetSubmitContext *submit,
                              int *ready, flagcxResult_t *firstError) {
  if (transport == NULL || submit == NULL || ready == NULL ||
      firstError == NULL)
    return flagcxInvalidArgument;
  uint64_t nextSequence = 0;
  uint32_t inFlight = 0;
  flagcxResult_t result =
      flagcxKernelProxyQuery(transport, &nextSequence, &inFlight, firstError);
  if (result != flagcxSuccess)
    return result;
  *ready = nextSequence == submit->sequence;
  return flagcxSuccess;
}
