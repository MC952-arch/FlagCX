#include "du_adaptor.h"

#ifdef USE_DU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "param.h"
#include <mutex>
#include <new>
#include <unordered_map>

struct DucudaVmmAllocation {
  size_t size;
  uint32_t mrCaps;
  bool mappingOwned;
  bool vaOwned;
};
static std::mutex gDucudaVmmAllocationMtx;
static std::unordered_map<void *, DucudaVmmAllocation> gDucudaVmmAllocations;

std::map<flagcxMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

flagcxResult_t ducudaAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                         flagcxMemcpyType_t type,
                                         flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(cudaMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        cudaMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMemset(ptr, value, size));
    } else {
      DEVCHECK(cudaMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDeviceMalloc(void **ptr, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(cudaMallocManaged(ptr, size, cudaMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMalloc(ptr, size));
    } else {
      DEVCHECK(cudaMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cudaFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(cudaFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaFree(ptr));
    } else {
      DEVCHECK(cudaFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "DU");
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                        void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  if (!flagcxParamVmmEnable()) {
    DEVCHECK(cudaMalloc(ptr, size));
    cudaPointerAttributes attrs;
    DEVCHECK(cudaPointerGetAttributes(&attrs, *ptr));
    unsigned flags = 1;
    DEVCHECK(cuPointerSetAttribute(&flags, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                   (CUdeviceptr)attrs.devicePointer));
    return flagcxSuccess;
  }

  int device = 0;
  CUdevice cuDevice;
  DEVCHECK(cudaGetDevice(&device));
  DEVCHECK(cuDeviceGet(&cuDevice, device));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cuDevice;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  int rdmaCapable = 0;
  CUresult attributeResult = cuDeviceGetAttribute(
      &rdmaCapable, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      cuDevice);
  if (attributeResult == CUDA_SUCCESS && rdmaCapable)
    prop.allocFlags.gpuDirectRDMACapable = 1;

  size_t granularity = 0;
  DEVCHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                         CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  size_t allocSize = size;
  ALIGN_SIZE(allocSize, granularity);

  CUmemGenericAllocationHandle handle;
  DEVCHECK(cuMemCreate(&handle, allocSize, &prop, 0));
  CUdeviceptr address = 0;
  CUresult result = cuMemAddressReserve(&address, allocSize, granularity, 0, 0);
  if (result != CUDA_SUCCESS) {
    cuMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  result = cuMemMap(address, allocSize, 0, handle, 0);
  if (result != CUDA_SUCCESS) {
    cuMemAddressFree(address, allocSize);
    cuMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = cuDevice;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  result = cuMemSetAccess(address, allocSize, &access, 1);
  if (result != CUDA_SUCCESS) {
    cuMemUnmap(address, allocSize);
    cuMemAddressFree(address, allocSize);
    cuMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  if (cuMemRelease(handle) != CUDA_SUCCESS) {
    cuMemUnmap(address, allocSize);
    cuMemAddressFree(address, allocSize);
    return flagcxUnhandledDeviceError;
  }
  *ptr = (void *)address;
  bool tracked = false;
  try {
    std::lock_guard<std::mutex> lock(gDucudaVmmAllocationMtx);
    const uint32_t mrCaps =
        FLAGCX_VMM_MR_CAP_DMABUF | (rdmaCapable ? FLAGCX_VMM_MR_CAP_VA : 0);
    tracked =
        gDucudaVmmAllocations
            .emplace(*ptr, DucudaVmmAllocation{allocSize, mrCaps, true, true})
            .second;
  } catch (const std::bad_alloc &) {
  }
  if (!tracked) {
    cuMemUnmap(address, allocSize);
    cuMemAddressFree(address, allocSize);
    *ptr = NULL;
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  std::lock_guard<std::mutex> lock(gDucudaVmmAllocationMtx);
  auto it = gDucudaVmmAllocations.find(ptr);
  if (it == gDucudaVmmAllocations.end()) {
    DEVCHECK(cudaFree(ptr));
    return flagcxSuccess;
  }
  DucudaVmmAllocation &allocation = it->second;
  if (allocation.mappingOwned) {
    if (cuMemUnmap((CUdeviceptr)ptr, allocation.size) != CUDA_SUCCESS)
      return flagcxUnhandledDeviceError;
    allocation.mappingOwned = false;
  }
  if (allocation.vaOwned) {
    if (cuMemAddressFree((CUdeviceptr)ptr, allocation.size) != CUDA_SUCCESS)
      return flagcxUnhandledDeviceError;
    allocation.vaOwned = false;
  }
  gDucudaVmmAllocations.erase(it);
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamCopy(flagcxStream_t *newStream,
                                       void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    cudaError error = cudaStreamQuery(stream->base);
    if (error == cudaSuccess) {
      res = flagcxSuccess;
    } else if (error == cudaErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t ducudaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                            flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        cudaStreamWaitEvent(stream->base, event->base, cudaEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventCreate(flagcxEvent_t *event,
                                        flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventRecord(flagcxEvent_t event,
                                        flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(cudaEventRecordWithFlags(event->base, stream->base,
                                        cudaEventRecordDefault));
    } else {
      DEVCHECK(cudaEventRecordWithFlags(event->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    cudaError error = cudaEventQuery(event->base);
    if (error == cudaSuccess) {
      res = flagcxSuccess;
    } else if (error == cudaErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t ducudaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                               size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                            void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                             void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                           void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return flagcxInvalidArgument;

  *dmaBufferSupport = false;
  int device = 0;
  CUdevice cuDevice;
  int supported = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cuDeviceGet(&cuDevice, device) != CUDA_SUCCESS)
    return flagcxSuccess;
  CUresult result = cuDeviceGetAttribute(
      &supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cuDevice);
  if (result == CUDA_SUCCESS)
    *dmaBufferSupport = supported != 0;
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorMemGetHandleForAddressRange(
    void *handleOut, void *buffer, size_t size, unsigned long long flags) {
  if (handleOut == NULL || buffer == NULL || size == 0)
    return flagcxInvalidArgument;
  CUresult result =
      cuMemGetHandleForAddressRange(handleOut, (CUdeviceptr)buffer, size,
                                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
  if (result == CUDA_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  return result == CUDA_SUCCESS ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t ducudaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                                int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  cudaDeviceProp devProp;
  DEVCHECK(cudaGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // TODO: see if there's another way to get this info. In some cuda versions,
  // cudaDeviceProp does not have `gpuDirectRDMASupported` field
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                              int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetDeviceByPciBusId(int *dev,
                                                const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamWaitValue64(flagcxStream_t stream, void *addr,
                                              uint64_t value, int flags) {
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  if (flags & ~FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxInvalidArgument;

  // The current DU driver supports ordinary stream memory waits but not the
  // acquire/flush guarantee required after a NIC or peer device publishes the
  // waited-on value. Report the missing capability without invoking a driver
  // flag that can surface as a generic error on this CUDA-compatible runtime.
  if (flags & FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxNotSupported;

  unsigned int waitFlags = CU_STREAM_WAIT_VALUE_GEQ;

  CUstream cuStream = (CUstream)(stream->base);
  CUresult err =
      cuStreamWaitValue64(cuStream, (CUdeviceptr)addr, value, waitFlags);
  if (err == CUDA_SUCCESS)
    return flagcxSuccess;
  if (err == CUDA_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  return flagcxUnhandledDeviceError;
}
flagcxResult_t ducudaAdaptorStreamWriteValue64(flagcxStream_t stream,
                                               void *addr, uint64_t value,
                                               int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWriteValue64(cuStream, (CUdeviceptr)addr, value,
                                      CU_STREAM_WRITE_VALUE_DEFAULT);
  return (err == CUDA_SUCCESS) ? flagcxSuccess : flagcxUnhandledDeviceError;
}
flagcxResult_t ducudaAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
                                             flagcxEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return flagcxInvalidArgument;
  }
  cudaError_t error = cudaEventElapsedTime(ms, start->base, end->base);
  if (error == cudaSuccess) {
    return flagcxSuccess;
  } else if (error == cudaErrorNotReady) {
    return flagcxInProgress;
  } else {
    return flagcxUnhandledDeviceError;
  }
}

flagcxResult_t ducudaAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostUnregister(ptr));
  return flagcxSuccess;
}

// Symmetric memory VMM handle exchange and flat mapping.
flagcxResult_t ducudaAdaptorSymPhysAlloc(void *ptr, size_t size,
                                         void **physHandle,
                                         void *shareableHandle,
                                         size_t *handleSize,
                                         size_t *allocSize) {
  if (ptr == NULL || physHandle == NULL || shareableHandle == NULL ||
      handleSize == NULL || allocSize == NULL)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)malloc(
          sizeof(CUmemGenericAllocationHandle));
  if (cuHandle == NULL)
    return flagcxSystemError;

  // Retain the physical allocation handle from the VMM-backed pointer
  CUresult result = cuMemRetainAllocationHandle(cuHandle, ptr);
  if (result != CUDA_SUCCESS) {
    free(cuHandle);
    return flagcxUnhandledDeviceError;
  }

  // Discover actual physical allocation size (already granularity-aligned)
  size_t actualAllocSize = 0;
  result = cuMemGetAddressRange(NULL, &actualAllocSize, (CUdeviceptr)ptr);
  if (result != CUDA_SUCCESS) {
    cuMemRelease(*cuHandle);
    free(cuHandle);
    return flagcxUnhandledDeviceError;
  }
  *allocSize = actualAllocSize;

  // Export as POSIX fd for IPC sharing
  if (*handleSize < sizeof(int)) {
    cuMemRelease(*cuHandle);
    free(cuHandle);
    return flagcxInvalidArgument;
  }
  result = cuMemExportToShareableHandle(
      shareableHandle, *cuHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (result != CUDA_SUCCESS) {
    cuMemRelease(*cuHandle);
    free(cuHandle);
    return flagcxUnhandledDeviceError;
  }
  *handleSize = sizeof(int); // POSIX fd is an int
  *physHandle = cuHandle;
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymPhysFree(void *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)physHandle;
  if (cuMemRelease(*cuHandle) != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  free(cuHandle);
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                       int selfIndex, void *selfPhysHandle,
                                       size_t allocSize, void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0 || allocSize == 0)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle selfHandle =
      *(CUmemGenericAllocationHandle *)selfPhysHandle;

  // allocSize is already granularity-aligned (from cuMemGetAddressRange)
  size_t totalSize = allocSize * nPeers;

  // Reserve the full VA range
  CUdeviceptr base = 0;
  CUresult result = cuMemAddressReserve(&base, totalSize, 0, 0, 0);
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;

  // Import and map each peer's physical memory
  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) {
    cuMemAddressFree(base, totalSize);
    return flagcxUnhandledDeviceError;
  }
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  int mappedPeers = 0;
  for (int i = 0; i < nPeers; i++) {
    CUmemGenericAllocationHandle peerHandle;
    bool imported = i != selfIndex;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      int fd = *(int *)peerHandles[i];
      result = cuMemImportFromShareableHandle(
          &peerHandle, (void *)(uintptr_t)fd,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
      if (result != CUDA_SUCCESS)
        goto rollback;
    }
    CUdeviceptr slot = base + (CUdeviceptr)i * allocSize;
    result = cuMemMap(slot, allocSize, 0, peerHandle, 0);
    if (result != CUDA_SUCCESS) {
      if (imported)
        cuMemRelease(peerHandle);
      goto rollback;
    }
    mappedPeers++;
    result = cuMemSetAccess(slot, allocSize, &accessDesc, 1);
    if (imported)
      cuMemRelease(peerHandle);
    if (result != CUDA_SUCCESS)
      goto rollback;
  }

  *flatBase = (void *)base;
  return flagcxSuccess;

rollback:
  for (int i = 0; i < mappedPeers; i++)
    cuMemUnmap(base + (CUdeviceptr)i * allocSize, allocSize);
  cuMemAddressFree(base, totalSize);
  return flagcxUnhandledDeviceError;
}
flagcxResult_t ducudaAdaptorSymFlatUnmap(void *flatBase, size_t allocSize,
                                         int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  CUdeviceptr base = (CUdeviceptr)flatBase;
  size_t totalSize = allocSize * nPeers;
  DEVCHECK(cuMemUnmap(base, totalSize));
  DEVCHECK(cuMemAddressFree(base, totalSize));
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymFlatMappingUnmap(void *flatBase,
                                                size_t allocSize, int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemUnmap((CUdeviceptr)flatBase, allocSize * nPeers));
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymFlatVaFree(void *flatBase, size_t allocSize,
                                          int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemAddressFree((CUdeviceptr)flatBase, allocSize * nPeers));
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastSupported(int *supported) {
  // not supported on dcu
  if (supported == NULL)
    return flagcxInvalidArgument;

  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastCreate(size_t allocSize,
                                               int nLocalDevices,
                                               const int *localDeviceOrdinals,
                                               void **mcHandle,
                                               int *shareableFd) {
  // not supported on dcu
  if (mcHandle)
    *mcHandle = NULL;

  if (shareableFd)
    *shareableFd = -1;

  return flagcxNotSupported;
}
flagcxResult_t ducudaAdaptorSymMulticastBind(void *mcHandle, int importFd,
                                             void *physHandle, size_t allocSize,
                                             int localRank, int nLocalDevices,
                                             void **mcBase, size_t *mcMapSize) {
  // not supported on dcu
  if (mcBase)
    *mcBase = NULL;

  if (mcMapSize)
    *mcMapSize = 0;

  return flagcxNotSupported;
}
flagcxResult_t ducudaAdaptorSymMulticastTeardown(void *mcBase,
                                                 size_t mcMapSize) {
  // not supported on dcu
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastMappingUnmap(void *mcBase,
                                                     size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemUnmap((CUdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastVaFree(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemAddressFree((CUdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastFree(void *mcHandle) {
  // not supported on dcu
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastImport(int, void **mcHandle) {
  if (mcHandle != NULL)
    *mcHandle = NULL;
  return flagcxNotSupported;
}

flagcxResult_t ducudaAdaptorGetAddressRange(const void *ptr, void **base,
                                            size_t *size) {
  if (ptr == NULL || base == NULL || size == NULL)
    return flagcxInvalidArgument;

  CUdeviceptr allocationBase = 0;
  CUresult result =
      cuMemGetAddressRange(&allocationBase, size, (CUdeviceptr)ptr);
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  *base = (void *)allocationBase;
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetAllocationVmmMrCaps(const void *ptr,
                                                   uint32_t *caps) {
  if (ptr == NULL || caps == NULL)
    return flagcxInvalidArgument;
  *caps = FLAGCX_VMM_MR_CAP_NONE;
  void *base = NULL;
  size_t size = 0;
  flagcxResult_t result = ducudaAdaptorGetAddressRange(ptr, &base, &size);
  if (result != flagcxSuccess)
    return result;
  std::lock_guard<std::mutex> lock(gDucudaVmmAllocationMtx);
  auto it = gDucudaVmmAllocations.find(base);
  if (it == gDucudaVmmAllocations.end())
    return flagcxNotSupported;
  *caps = it->second.mrCaps;
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor ducudaAdaptor {
  "DUCUDA",
      // Basic functions
      ducudaAdaptorDeviceSynchronize, ducudaAdaptorDeviceMemcpy,
      ducudaAdaptorDeviceMemset, ducudaAdaptorDeviceMalloc,
      ducudaAdaptorDeviceFree, ducudaAdaptorSetDevice, ducudaAdaptorGetDevice,
      ducudaAdaptorGetDeviceCount, ducudaAdaptorGetVendor,
      ducudaAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      ducudaAdaptorGdrMemAlloc, ducudaAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      ducudaAdaptorStreamCreate, ducudaAdaptorStreamDestroy,
      ducudaAdaptorStreamCopy, ducudaAdaptorStreamFree,
      ducudaAdaptorStreamSynchronize, ducudaAdaptorStreamQuery,
      ducudaAdaptorStreamWaitEvent, ducudaAdaptorStreamWaitValue64,
      ducudaAdaptorStreamWriteValue64,
      // Event functions
      ducudaAdaptorEventCreate, ducudaAdaptorEventDestroy,
      ducudaAdaptorEventRecord, ducudaAdaptorEventSynchronize,
      ducudaAdaptorEventQuery, ducudaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      ducudaAdaptorIpcMemHandleCreate, ducudaAdaptorIpcMemHandleGet,
      ducudaAdaptorIpcMemHandleOpen, ducudaAdaptorIpcMemHandleClose,
      ducudaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t
            // (*launchDeviceFunc)(flagcxStream_t stream,
            // void *args);
      // Others
      ducudaAdaptorGetDeviceProperties, // flagcxResult_t
                                        // (*getDeviceProperties)(struct
                                        // flagcxDevProps *props, int dev);
      ducudaAdaptorGetDevicePciBusId,   // flagcxResult_t
                                        // (*getDevicePciBusId)(char *pciBusId,
                                        // int len, int dev);
      ducudaAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                        // (*getDeviceByPciBusId)(
                                        // int
                                        // *dev, const char *pciBusId);
      ducudaAdaptorLaunchHostFunc,
      // DMA buffer
      ducudaAdaptorDmaSupport, // flagcxResult_t (*dmaSupport)(bool
                               // *dmaBufferSupport);
      ducudaAdaptorMemGetHandleForAddressRange, // flagcxResult_t
                                                // (*memGetHandleForAddressRange)(void
                                                // *handleOut, void *buffer,
                                                // size_t size, unsigned long
                                                // long flags);
      ducudaAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                   // size_t);
      ducudaAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions
      ducudaAdaptorSymPhysAlloc, ducudaAdaptorSymPhysFree,
      ducudaAdaptorSymFlatMap, ducudaAdaptorSymFlatUnmap,
      ducudaAdaptorSymMulticastSupported, ducudaAdaptorSymMulticastCreate,
      ducudaAdaptorSymMulticastBind, ducudaAdaptorSymMulticastTeardown,
      ducudaAdaptorSymMulticastFree,
      NULL, // flagcxResult_t (*getLastError)();
      flagcxDeviceAdaptorGetPointerTypeNotSupported,
      ducudaAdaptorGetAddressRange,
      FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA,
      FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, ducudaAdaptorSymMulticastImport,
      ducudaAdaptorSymFlatMappingUnmap, ducudaAdaptorSymFlatVaFree,
      ducudaAdaptorSymMulticastMappingUnmap, ducudaAdaptorSymMulticastVaFree,
      ducudaAdaptorGetAllocationVmmMrCaps,
};

#endif // USE_DU_ADAPTOR
