/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Kernel Internal — Host-side functions and lifecycle management.
 *
 * This header contains host-only content that requires adaptor.h and
 * other host infrastructure. It includes:
 *   - flagcxFifo struct and methods
 *   - Host-side FIFO functions (dequeue, enqueue)
 *   - Device communicator lifecycle (flagcxDevCommCreate, etc.)
 *   - Device memory lifecycle (flagcxDevMemCreate, etc.)
 *   - Test kernels and AlltoAll implementations
 *
 * NOT safe for LLVM bitcode compilation.
 * For device-side types and constants, see flagcx_kernel_core.h.
 * For normal builds, include flagcx_kernel.h (umbrella header).
 ************************************************************************/

#ifndef FLAGCX_KERNEL_INTERNAL_H_
#define FLAGCX_KERNEL_INTERNAL_H_

#include "adaptor.h"
#include "flagcx_kernel_core.h"

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
FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(volatile uint64_t *buffer,
                                                      int *idx);

FLAGCX_DEVICE_DECORATOR size_t
getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype);

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer);
#endif // COMPILE_KERNEL

void flagcxLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, flagcxStream_t stream);

// ==========================================================================
// Device Communicator — Host-side lifecycle management
// ==========================================================================

// Requirements for creating a device communicator.
// Named fields map to NCCL ncclDevCommRequirements (Vendor).
// Naming: NCCL "lsa" → FlagCX "intra", "gin" → "inter", "multimem" →
// "multicast".
struct flagcxDevCommRequirements {
  bool intraMulticast; // → ncclReqs.lsaMultimem

  int barrierCount;      // → ncclReqs.barrierCount (world barrier)
  int intraBarrierCount; // → ncclReqs.lsaBarrierCount
  int interBarrierCount; // → ncclReqs.railGinBarrierCount

  int intraLLA2ABlockCount; // → ncclReqs.lsaLLA2ABlockCount
  int intraLLA2ASlotCount;  // → ncclReqs.lsaLLA2ASlotCount

  bool interForceEnable; // → ncclReqs.ginForceEnable
  int interContextCount; // → ncclReqs.ginContextCount (hint, default 4)
  int interSignalCount;  // → ncclReqs.ginSignalCount (start at id=0)
  int interCounterCount; // → ncclReqs.ginCounterCount (start at id=0)
};

#define FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER                               \
  {                                                                            \
    false,       /* intraMulticast */                                          \
        0, 0, 0, /* barrierCount, intraBarrierCount, interBarrierCount */      \
        0, 0,    /* intraLLA2ABlockCount, intraLLA2ASlotCount */               \
        false, 4, 0, 0 /* interForceEnable, interContextCount,                 \
                          interSignalCount, interCounterCount */               \
  }

// Network type enumeration (maps to ncclGinType_t on NVIDIA backend).
typedef enum {
  flagcxNetTypeNone = 0,  // → NCCL_GIN_TYPE_NONE
  flagcxNetTypeProxy = 2, // → NCCL_GIN_TYPE_PROXY
  flagcxNetTypeGdaki = 3, // → NCCL_GIN_TYPE_GDAKI
} flagcxNetType_t;

// Communicator properties — host-side queryable attributes.
struct flagcxCommProperties {
  int rank;
  int nRanks;
  int deviceId; // → ncclCommProperties.cudaDev (platform-neutral)
  bool vendorDeviceApiSupport; // → ncclCommProperties.deviceApiSupport
  bool multicastSupport;       // → ncclCommProperties.multimemSupport
  flagcxNetType_t netType;     // → ncclCommProperties.ginType
};
typedef struct flagcxCommProperties flagcxCommProperties_t;

// Query communicator properties.
// Currently returns placeholder defaults; will delegate to backend
// (e.g. ncclCommQueryProperties) when wired through the adaptor layer.
flagcxResult_t flagcxCommQueryProperties(flagcxComm_t comm,
                                         flagcxCommProperties_t *props);

// Forward declarations for types defined in flagcx_device.h.
struct flagcxTeam;
typedef struct flagcxTeam flagcxTeam_t;
struct flagcxDevCommRequirements;
struct flagcxIntraBarrierHandle;
typedef struct flagcxIntraBarrierHandle flagcxIntraBarrierHandle_t;
struct flagcxInterBarrierHandle;
typedef struct flagcxInterBarrierHandle flagcxInterBarrierHandle_t;

// Create barrier requirement handles (stub — returns flagcxNotSupported).
// FlagCX currently uses intraBarrierCount in DevCommCreate directly;
// the resource-handle model will be implemented when needed.
flagcxResult_t
flagcxIntraBarrierCreateRequirement(flagcxTeam_t team, int nBarriers,
                                    flagcxIntraBarrierHandle_t *outHandle,
                                    flagcxDevCommRequirements *outReq);

flagcxResult_t flagcxInterBarrierCreateRequirement(
    flagcxComm_t comm, flagcxTeam_t team, int nBarriers,
    flagcxInterBarrierHandle_t *outHandle, flagcxDevCommRequirements *outReq);

// Opaque handle to a device communicator (host-side lifetime management).
// Internally wraps ncclDevComm on NVIDIA backend (Vendor),
// or IPC barrier state on default path (Default).
typedef struct flagcxDevCommInternal *flagcxDevComm_t;

// Opaque handle to device memory (host-side lifetime management).
// Internally wraps ncclWindow_t on NVIDIA backend (Vendor),
// or IPC peer pointer table on default path (Default).
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

// Inter-node one-sided AlltoAll (put + waitSignal + flush).
flagcxResult_t flagcxInterOneSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

// Inter-node two-sided AlltoAll (send/recv + term/wait via FIFO).
flagcxResult_t flagcxInterTwoSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

// Inter-node Device API test kernels.
// Each kernel tests one API facet; host verifies after streamSynchronize.
flagcxResult_t flagcxInterTestPutSignalInc(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

flagcxResult_t flagcxInterTestPutSignalAddDecoupled(
    flagcxDevMem_t sendMem, flagcxDevMem_t recvMem, size_t count,
    flagcxDataType_t datatype, flagcxDevComm_t devComm, flagcxStream_t stream);

flagcxResult_t
flagcxInterTestCounterPipeline(flagcxDevMem_t sendMem, flagcxDevMem_t recvMem,
                               size_t count, flagcxDataType_t datatype,
                               flagcxDevComm_t devComm, flagcxStream_t stream,
                               uint64_t *resultBuf);

flagcxResult_t flagcxInterTestPutValue(flagcxDevMem_t recvMem,
                                       flagcxDevComm_t devComm,
                                       flagcxStream_t stream,
                                       size_t putValBase);

flagcxResult_t flagcxInterTestSignal(flagcxDevComm_t devComm,
                                     flagcxStream_t stream);

flagcxResult_t
flagcxInterTestFlushDecouple(flagcxDevMem_t sendMem, flagcxDevMem_t recvMem,
                             size_t count, flagcxDataType_t datatype,
                             flagcxDevComm_t devComm, flagcxStream_t stream);

flagcxResult_t flagcxInterTestFollowShadow(flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

flagcxResult_t flagcxInterTestMeetShadow(flagcxDevComm_t devComm,
                                         flagcxStream_t stream);

flagcxResult_t flagcxInterTestReset(flagcxDevComm_t devComm,
                                    flagcxStream_t stream, uint64_t *resultBuf);

flagcxResult_t flagcxInterTestGet(flagcxDevMem_t sendMem,
                                  flagcxDevMem_t recvMem, size_t count,
                                  flagcxDataType_t datatype,
                                  flagcxDevComm_t devComm,
                                  flagcxStream_t stream);

// Create a device communicator for custom kernel usage.
// On NVIDIA backend (Vendor), internally calls pncclDevCommCreate.
// On default path (Default), sets up IPC-based barrier across intra-node peers.
// The returned handle must be destroyed with flagcxDevCommDestroy(comm,
// devComm).
flagcxResult_t flagcxDevCommCreate(flagcxComm_t comm,
                                   const flagcxDevCommRequirements *reqs,
                                   flagcxDevComm_t *devComm);

// Destroy a device communicator created by flagcxDevCommCreate.
flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm, flagcxDevComm_t devComm);

// Create a device memory handle for a registered buffer.
// Registration is the caller's responsibility (Decision 7.16):
//   - IPC mode (win=NULL): caller calls flagcxCommRegister first.
//   - Window mode (win!=NULL): caller calls flagcxCommWindowRegister first.
// This function exchanges IPC handles to build peer pointer tables (both modes)
// and stores the window handle (window mode only).
flagcxResult_t flagcxDevMemCreate(flagcxComm_t comm, void *buff, size_t size,
                                  flagcxWindow_t win, flagcxDevMem_t *devMem);

// Destroy a device memory handle created by flagcxDevMemCreate.
flagcxResult_t flagcxDevMemDestroy(flagcxComm_t comm, flagcxDevMem_t devMem);

// Clean up IPC peer pointer table on comm.
// Must be called after homoComm destroy.
// so that cudaFree does not deadlock on device synchronization.
flagcxResult_t flagcxCommCleanupIpcTable(flagcxComm_t comm);

// Tear down inter-node signal relay stored on heteroComm.
// Must be called before flagcxHeteroCommDestroy (which frees proxyState and
// heteroComm). Internally drains FIFOs and performs a cross-rank barrier
// before closing RDMA connections.
flagcxResult_t flagcxCommRelayDestroy(flagcxComm_t comm);

// Deferred device/host-pinned memory free.
// Collects pointers during DevComm/DevMem cleanup.
void flagcxCommDeferFree(flagcxComm_t comm, void *ptr, int memType);
flagcxResult_t flagcxCommDrainDeferredFrees(flagcxComm_t comm);

// Release data buffer resources (MR, network connections, handle arrays).
flagcxResult_t flagcxOneSideDeregister(struct flagcxHeteroComm *heteroComm);

// Release signal buffer resources (MR, network connections, handle arrays).
// flagcxOneSideSignalRegister / flagcxOneSideStagingRegister /
// flagcxOneSideStagingDeregister are declared in flagcx.h (extern "C").
flagcxResult_t
flagcxOneSideSignalDeregister(struct flagcxHeteroComm *heteroComm);

// One-sided barrier MR registration (host-pinned memory for inter-node
// barrier). Collective: ALL ranks must call. Leaders pass recvComm+buff,
// non-leaders pass NULL.
flagcxResult_t
flagcxOneSideBarrierRegister(const flagcxComm_t comm, void *recvComm,
                             void *buff, size_t size,
                             struct flagcxOneSideHandleInfo **outInfo);
// Release barrier MR and free handle info.
flagcxResult_t
flagcxOneSideBarrierDeregister(const flagcxComm_t comm,
                               struct flagcxOneSideHandleInfo *info);

// Intra-node AllReduce using FlagCX Device API.
// The caller provides a registered buffer (via flagcxDevMemCreate)
// already containing the input data.  The kernel runs an in-place
// AllReduce across all intra-node GPUs.
// devComm must be created via flagcxDevCommCreate beforehand.
flagcxResult_t flagcxIntraAllReduce(flagcxDevMem_t devMem, size_t count,
                                    flagcxDataType_t datatype,
                                    flagcxDevComm_t devComm,
                                    flagcxStream_t stream);

#endif // FLAGCX_KERNEL_INTERNAL_H_
