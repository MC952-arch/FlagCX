/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_FIFO_PRODUCER_GATE_H_
#define FLAGCX_FIFO_PRODUCER_GATE_H_

#include "device_utils.h"
#include "flagcx_device_enums.h"

#include <stdint.h>

// The close bit and active-producer count share one atomic word. This makes
// closing the gate atomic with acquiring a producer reference: either the
// producer increments the count first and the host waits for it, or the host
// closes the gate first and the producer is rejected.
template <typename Word>
FLAGCX_HOST_DEVICE_INLINE constexpr Word flagcxFifoProducerClosedMask() {
  return Word{1} << (sizeof(Word) * 8 - 1);
}

template <typename Word>
FLAGCX_HOST_DEVICE_INLINE constexpr Word flagcxFifoProducerActiveMask() {
  return static_cast<Word>(~flagcxFifoProducerClosedMask<Word>());
}

template <typename Atomic, typename Word>
FLAGCX_DEVICE_INLINE_DECORATOR bool
flagcxFifoProducerTryEnter(Word *producerState) {
  const Word closedMask = flagcxFifoProducerClosedMask<Word>();
  // Linearize producer entry with a single RMW. If close wins first, the
  // producer temporarily contributes to the active count, observes the close
  // bit in the returned value, and immediately drops that reference without
  // reserving a FIFO sequence. If entry wins first, the closer observes the
  // active reference and waits for flagcxFifoProducerLeave().
  Word observed =
      Atomic::fetchAdd(producerState, Word{1}, flagcxDeviceMemoryOrderAcqRel);
  if ((observed & closedMask) != 0) {
    Atomic::fetchSub(producerState, Word{1}, flagcxDeviceMemoryOrderRelease);
    return false;
  }
  return true;
}

template <typename Atomic, typename Word>
FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxFifoProducerLeave(Word *producerState) {
  Atomic::fetchSub(producerState, Word{1}, flagcxDeviceMemoryOrderRelease);
}

#endif // FLAGCX_FIFO_PRODUCER_GATE_H_
