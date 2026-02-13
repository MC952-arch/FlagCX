#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "device_utils.h"

FLAGCX_DEVICE_DECORATOR size_t
getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype) {
  switch (dtype) {
    // case flagcxInt8:
    case flagcxChar:
      return sizeof(char); // 1 byte
    case flagcxUint8:
      return sizeof(unsigned char); // 1 byte
    // case flagcxInt32:
    case flagcxInt:
      return sizeof(int); // 4 bytes
    case flagcxUint32:
      return sizeof(unsigned int); // 4 bytes
    case flagcxInt64:
      return sizeof(long long); // 8 bytes
    case flagcxUint64:
      return sizeof(unsigned long long); // 8 bytes
    // case flagcxFloat16:
    case flagcxHalf:
      return 2; // Half precision float is 2 bytes
    // case flagcxFloat32:
    case flagcxFloat:
      return sizeof(float); // 4 bytes
    // case flagcxFloat64:
    case flagcxDouble:
      return sizeof(double); // 8 bytes
    case flagcxBfloat16:
      return 2; // BFloat16 is typically 2 bytes
    default:
      return 0;
  }
}

FLAGCX_DEVICE_DECORATOR void
flagcxDeviceTrigger::setValue(uint64_t addr, uint64_t count, uint64_t peerRank,
                              uint64_t datatype, uint64_t type) {
  fst = addr;
  snd = (count & flagcxTriggerMask(flagcxReduceTriggerBitsCount))
            << flagcxDeviceTriggerOffCount |
        (peerRank & flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank))
            << flagcxDeviceTriggerOffPeerRank |
        (datatype & flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype))
            << flagcxDeviceTriggerOffDatatype |
        (type & flagcxTriggerMask(flagcxDeviceTriggerBitsPrim))
            << flagcxDeviceTriggerOffPrim;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceSend(const void *sendbuff, size_t count, flagcxDataType_t datatype,
                 int peer, void *fifoBuffer) {
  enqueue(fifoBuffer, (uint64_t)((uintptr_t)sendbuff), count, peer, datatype,
          flagcxDevicePrimSend);
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceRecv(void *recvbuff, size_t count, flagcxDataType_t datatype,
                 int peer, void *fifoBuffer) {
  enqueue(fifoBuffer, (uint64_t)((uintptr_t)recvbuff), count, peer, datatype,
          flagcxDevicePrimRecv);
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t flagcxDeviceTerm(void *fifoBuffer) {
  enqueue(fifoBuffer, 0, 0, 0, 0, flagcxDevicePrimTerm);
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t flagcxDeviceWait(void *fifoBuffer) {
  // Enqueue WAIT primitive so host knows to synchronize
  enqueue(fifoBuffer, 0, 0, 0, 0, flagcxDevicePrimWait);

  volatile uint64_t *buffer = (volatile uint64_t *)fifoBuffer;

  // Memory fence to ensure all previous enqueues are visible
  FLAGCX_DEVICE_THREAD_FENCE();

  // Wait until all items are consumed (consumed catches up to produced)
  int iter = 0;
  while (buffer[2] > buffer[1]) {
    spinBackoff(iter++);
  }
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t enqueue(void *fifoBuffer, uint64_t addr,
                                               uint64_t count,
                                               uint64_t peerRank,
                                               uint64_t datatype,
                                               uint64_t type) {
  volatile uint64_t *buffer = (volatile uint64_t *)fifoBuffer;
  uint64_t capacity = buffer[0];

  // 1. Atomically reserve a slot
  uint64_t mySlot = atomicAdd((unsigned long long *)&buffer[2], 1ULL);

  // 2. Wait until there's space (mySlot - consumed < capacity)
  int iter = 0;
  while ((int64_t)(mySlot - buffer[1]) >= (int64_t)capacity) {
    spinBackoff(iter++);
  }

  // 3. Compute slot index and get pointer to slot's raw uint64_t fields
  uint64_t idx = mySlot % capacity;
  volatile uint64_t *slotFst =
      buffer + 3 + idx * (sizeof(flagcxDeviceTrigger) / sizeof(uint64_t));
  volatile uint64_t *slotSnd = slotFst + 1;

  // 4. Write address first (explicit volatile write)
  *slotFst = addr;

  // 5. Build snd value with valid bit set
  uint64_t sndValue =
      (count & flagcxTriggerMask(flagcxDeviceTriggerBitsCount))
          << flagcxDeviceTriggerOffCount |
      (peerRank & flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank))
          << flagcxDeviceTriggerOffPeerRank |
      (datatype & flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype))
          << flagcxDeviceTriggerOffDatatype |
      (type & flagcxTriggerMask(flagcxDeviceTriggerBitsPrim))
          << flagcxDeviceTriggerOffPrim |
      flagcxDeviceTriggerValidMask;  // Set valid bit

  // 6. Memory fence to ensure fst is visible before snd
  FLAGCX_DEVICE_THREAD_FENCE();

  // 7. Write snd with valid bit (explicit volatile write, signals data is ready)
  *slotSnd = sndValue;

  return flagcxSuccess;
}