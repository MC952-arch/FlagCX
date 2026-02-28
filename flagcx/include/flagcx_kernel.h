#ifndef FLAGCX_KERNEL_H_
#define FLAGCX_KERNEL_H_

#include "adaptor.h"
#include "flagcx.h"

#define FLAGCX_FIFO_CAPACITY 128
#define flagcxTriggerMask(w) ((w == 64) ? ~0ull : ((1ull << w) - 1))

typedef enum {
  flagcxDevicePrimSend = 0,
  flagcxDevicePrimRecv = 1,
  flagcxDevicePrimTerm = 2,
  flagcxDevicePrimWait = 3
} flagcxDevicePrim;

// Unified buffer index enumeration for fifo
// Layout: [capacity][consumed][produced][terminate][data...]
// Note: flagcxFifoIdxTerminate is only used by flagcxReduceTrigger fifo
typedef enum {
  flagcxFifoIdxCapacity = 0,
  flagcxFifoIdxConsumed = 1,
  flagcxFifoIdxProduced = 2,
  flagcxFifoIdxTerminate = 3,
  flagcxFifoIdxData = 4
} flagcxFifoIndex;

typedef enum {
  flagcxReduceTriggerAvailable = 0,
  flagcxReduceTriggerEnqueued = 1,
  flagcxReduceTriggerInprogress = 2,
  flagcxReduceTriggerComplete = 3
} flagcxReduceTriggerState;

constexpr unsigned int flagcxDeviceTriggerBitsAddr = 64;
constexpr unsigned int flagcxDeviceTriggerOffCount = 0;
constexpr unsigned int flagcxDeviceTriggerBitsCount = 32;
constexpr unsigned int flagcxDeviceTriggerOffPeerRank =
    flagcxDeviceTriggerOffCount + flagcxDeviceTriggerBitsCount;
constexpr unsigned int flagcxDeviceTriggerBitsPeerRank = 20;
constexpr unsigned int flagcxDeviceTriggerOffDatatype =
    flagcxDeviceTriggerOffPeerRank + flagcxDeviceTriggerBitsPeerRank;
constexpr unsigned int flagcxDeviceTriggerBitsDatatype = 4;
constexpr unsigned int flagcxDeviceTriggerOffPrim =
    flagcxDeviceTriggerOffDatatype + flagcxDeviceTriggerBitsDatatype;
constexpr unsigned int flagcxDeviceTriggerBitsPrim = 4;
constexpr unsigned int flagcxDeviceTriggerBitsFifoReserved = 1;
// Valid bit for lock-free MPSC FIFO (bit 63 of snd field)
constexpr unsigned int flagcxDeviceTriggerOffValid = 63;
constexpr uint64_t flagcxDeviceTriggerValidMask = (1ULL << 63);

constexpr unsigned int flagcxReduceTriggerBitsAddr = 64;
constexpr unsigned int flagcxReduceTriggerOffCount = 0;
constexpr unsigned int flagcxReduceTriggerBitsCount = 32;
constexpr unsigned int flagcxReduceTriggerOffNThreads =
    flagcxReduceTriggerOffCount + flagcxReduceTriggerBitsCount;
constexpr unsigned int flagcxReduceTriggerBitsNThreads = 16;
constexpr unsigned int flagcxReduceTriggerOffDatatype =
    flagcxReduceTriggerOffNThreads + flagcxReduceTriggerBitsNThreads;
constexpr unsigned int flagcxReduceTriggerBitsDatatype = 4;
constexpr unsigned int flagcxReduceTriggerOffRedop =
    flagcxReduceTriggerOffDatatype + flagcxReduceTriggerBitsDatatype;
constexpr unsigned int flagcxReduceTriggerBitsRedop = 4;
constexpr unsigned int flagcxReduceTriggerOffState =
    flagcxReduceTriggerOffRedop + flagcxReduceTriggerBitsRedop;
/* op state: 0 for available, 1 for enqueued, 2 for in-progress, 3 for done */
constexpr unsigned int flagcxReduceTriggerBitsState = 2;
constexpr unsigned int flagcxReduceTriggerBitsFifoReserved = 1;

struct flagcxDeviceTrigger {
  uint64_t fst;
  uint64_t snd;

  FLAGCX_HOST_DECORATOR uint64_t getAddr();
  FLAGCX_HOST_DECORATOR uint64_t getCount();
  FLAGCX_HOST_DECORATOR uint64_t getPeerRank();
  FLAGCX_HOST_DECORATOR uint64_t getDatatype();
  FLAGCX_HOST_DECORATOR uint64_t getType();
  FLAGCX_DEVICE_DECORATOR void setValue(uint64_t addr, uint64_t count,
                                        uint64_t peerRank, uint64_t datatype,
                                        uint64_t type);
};
typedef flagcxDeviceTrigger *flagcxDeviceTrigger_t;

struct alignas(16) flagcxReduceTrigger {
  uint64_t value[4];

#ifdef COMPILE_KERNEL
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getInput1();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getInput2();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getOutput();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getCount();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getNThreads();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getDatatype();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getRedop();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getState();
  FLAGCX_DEVICE_INLINE_DECORATOR void setComplete();
#endif
  FLAGCX_HOST_DECORATOR void setValue(uint64_t fst, uint64_t snd, uint64_t out,
                                      size_t count, size_t nthreads,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t redOp,
                                      flagcxReduceTriggerState state);
  FLAGCX_HOST_DECORATOR uint64_t pollState();
  FLAGCX_HOST_DECORATOR void setState(int state);
};
typedef flagcxReduceTrigger *flagcxReduceTrigger_t;

struct flagcxFifo {
  // Unified fifo layout: [capacity][consumed][produced][terminate][data...]
  // flagcxDeviceTrigger fifo: terminate slot is reserved but unused
  // flagcxReduceTrigger fifo: terminate slot is used
  // See flagcxFifoIndex enumeration for index values
  uint64_t *buffer;

public:
  flagcxFifo() {}
  ~flagcxFifo() {}
  flagcxResult_t flagcxFifoInit();
  flagcxResult_t flagcxRedFifoInit();
  flagcxResult_t flagcxFifoDestroy();
  flagcxResult_t flagcxRedFifoDestroy();
};
typedef struct flagcxFifo *flagcxFifo_t;

FLAGCX_HOST_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                             flagcxDeviceTrigger_t trigger);
FLAGCX_HOST_DECORATOR flagcxResult_t enqueue(void *fifoBuffer, uint64_t addr1,
                                             uint64_t addr2, uint64_t addr3,
                                             size_t count, size_t nthreads,
                                             flagcxDataType_t datatype,
                                             flagcxRedOp_t redop, int *idx);
#ifdef COMPILE_KERNEL
FLAGCX_DEVICE_DECORATOR flagcxResult_t enqueue(void *fifoBuffer, uint64_t addr,
                                               uint64_t count,
                                               uint64_t peerRank,
                                               uint64_t datatype,
                                               uint64_t type);
FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(volatile uint64_t *buffer,
                                                      int *idx);

FLAGCX_DEVICE_DECORATOR size_t
getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype);

FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceSend(const void *sendbuff, size_t count, flagcxDataType_t datatype,
                 int peer, void *fifoBuffer);
FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceRecv(void *sendbuff, size_t count, flagcxDataType_t datatype,
                 int peer, void *fifoBuffer);
FLAGCX_DEVICE_DECORATOR flagcxResult_t flagcxDeviceTerm(void *fifoBuffer);
FLAGCX_DEVICE_DECORATOR flagcxResult_t flagcxDeviceWait(void *fifoBuffer);
FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer);
#endif // COMPILE_KERNEL

flagcxResult_t flagcxP2pDemo(const void *sendbuff, void *recvbuff, size_t count,
                             flagcxDataType_t datatype, flagcxComm_t comm,
                             flagcxStream_t stream);
void flagcxLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, flagcxStream_t stream);

// ==========================================================================
// Device Communicator — Host-side lifecycle management
// ==========================================================================

// Requirements for creating a device communicator
typedef struct {
  int lsaBarrierCount; // Number of LSA barrier slots (typically = CTA count)
  int lsaMultimem;     // Whether to enable multicast/multimem (0/1)
  int ginBarrierCount; // Number of GIN barrier slots for inter-node
  int ginSignalCount;  // Number of GIN signal slots
} flagcxDevCommRequirements;

#define FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER                               \
  { 0, 0, 0, 0 }

// Opaque handle to a device communicator (host-side lifetime management).
// Internally wraps ncclDevComm on NVIDIA backend.
typedef struct flagcxDevCommInternal *flagcxDevComm_t;

// Create a device communicator for custom kernel usage.
// On NVIDIA backend, internally calls pncclDevCommCreate with the given
// requirements.  The returned handle must be destroyed with
// flagcxDevCommDestroy.
flagcxResult_t flagcxDevCommCreate(flagcxComm_t comm,
                                   const flagcxDevCommRequirements *reqs,
                                   flagcxDevComm_t *devComm);

// Destroy a device communicator created by flagcxDevCommCreate.
flagcxResult_t flagcxDevCommDestroy(flagcxDevComm_t devComm);

// Intra-node AllReduce using FlagCX Device API (LSA peer pointers + barrier).
// The caller provides a window-registered buffer (via flagcxMemAlloc +
// flagcxCommWindowRegister) already containing the input data.  The kernel
// runs an in-place AllReduce across all intra-node GPUs.
// devComm must be created via flagcxDevCommCreate beforehand.
flagcxResult_t flagcxIntraAllReduceDemo(void *windowBuff, flagcxWindow_t win,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream);
#endif
