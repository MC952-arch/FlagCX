#ifdef USE_NVIDIA_ADAPTOR

#include "flagcx.h"
#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"

#define NCCL_ADAPTOR_DEVICE_CTA_COUNT 36
#define NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA 512
#define NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE (8 * 1024 * 1024)

struct stagedBuffer {
  void *buff;
  ncclWindow_t win;
};
typedef struct stagedBuffer *stagedBuffer_t;

#if defined(COMPILE_KERNEL_HOST)
extern "C" ncclResult_t
ncclAdaptorLocalAllReduce(const void *sendbuff, void *recvbuff,
                          ncclWindow_t sendwin, ncclWindow_t recvwin,
                          size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                          ncclDevComm &devComm, cudaStream_t stream);

extern "C" ncclResult_t ncclAdaptorInterleavedAllReduce(
    const void *sendbuff, void *recvbuff, ncclWindow_t sendwin,
    ncclWindow_t recvwin, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    ncclDevComm &devComm, cudaStream_t stream);
#endif // COMPILE_KERNEL_HOST

struct flagcxInnerDevComm {
  ncclDevComm base;
};

#else

typedef void *stagedBuffer_t;
typedef void ncclDevComm;
struct flagcxInnerDevComm {};

#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)

struct flagcxInnerComm {
  ncclComm_t base;
  ncclDevComm *devBase;
  stagedBuffer_t sendStagedBuff;
  stagedBuffer_t recvStagedBuff;
};

struct flagcxStream {
  cudaStream_t base;
};

struct flagcxEvent {
  cudaEvent_t base;
};

struct flagcxIpcMemHandle {
  cudaIpcMemHandle_t base;
};

#if NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
#define FLAGCX_SYM_WINDOW_DEFINED
struct flagcxSymWindow {
  ncclWindow_t base;
  int winFlags;

  // Non-homo symmetric path fields (unused on homo NCCL path)
  void *flatBase;     // flat VA base (NULL if IPC fallback)
  void *mcBase;       // multicast base (NULL if no NVLS)
  void **devPeerPtrs; // device-side peer pointer array
  int mrIndex;        // one-sided MR index (-1 if none)
  uintptr_t mrBase;   // MR base VA
  size_t heapSize;    // currently backed size per peer
  size_t maxHeapSize; // total reserved VA per peer (for growth)
  int localRanks;     // number of intra-node peers
  void *physHandle;   // for cleanup (symPhysFree)
  void *mcHandle;     // multicast handle (for growth + cleanup)

  // Growth tracking
  void **growthPhysHandles;
  int growthCount;
  int growthCapacity;

  // Cleanup state
  bool isSymmetricFallback; // true if non-homo symmetric path
  bool isVMM;               // true if VMM path (false = IPC fallback)
};
#else
#define FLAGCX_SYM_WINDOW_DEFINED
struct flagcxSymWindow {
  int winFlags;

  // Non-homo symmetric path fields
  void *flatBase;
  void *mcBase;
  void **devPeerPtrs;
  int mrIndex;
  uintptr_t mrBase;
  size_t heapSize;
  size_t maxHeapSize;
  int localRanks;
  void *physHandle;
  void *mcHandle;

  void **growthPhysHandles;
  int growthCount;
  int growthCapacity;

  bool isSymmetricFallback;
  bool isVMM;
};
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_NVIDIA_ADAPTOR