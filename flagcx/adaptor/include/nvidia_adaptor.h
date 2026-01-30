#ifdef USE_NVIDIA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include "flagcx.h"
#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include "nccl_device.h"


struct stagedBuffer {
  void *buff;
  ncclWindow_t win;
};
typedef struct stagedBuffer *stagedBuffer_t;

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

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_NVIDIA_ADAPTOR