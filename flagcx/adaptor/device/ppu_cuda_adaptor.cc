#include "ppu_adaptor.h"

#ifdef USE_PPU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<flagcxMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

flagcxResult_t ppu_cudaAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
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

flagcxResult_t ppu_cudaAdaptorDeviceMemset(void *ptr, int value, size_t size,
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

flagcxResult_t ppu_cudaAdaptorDeviceMalloc(void **ptr, size_t size,
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

flagcxResult_t ppu_cudaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
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

flagcxResult_t ppu_cudaAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "PPU");
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorHostGetDevicePointer(void **pDevice,
                                                   void *pHost) {
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                          void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaMalloc(ptr, size));
  cudaPointerAttributes attrs;
  DEVCHECK(cudaPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(cuPointerSetAttribute(&flags, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (CUdeviceptr)attrs.devicePointer));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(cudaFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorStreamCopy(flagcxStream_t *newStream,
                                         void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorStreamQuery(flagcxStream_t stream) {
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

flagcxResult_t ppu_cudaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                              flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        cudaStreamWaitEvent(stream->base, event->base, cudaEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorStreamWaitValue64(flagcxStream_t stream,
                                                void *addr, uint64_t value,
                                                int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWaitValue64(cuStream, (CUdeviceptr)addr, value,
                                     CU_STREAM_WAIT_VALUE_GEQ);
  return (err == CUDA_SUCCESS) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t ppu_cudaAdaptorStreamWriteValue64(flagcxStream_t stream,
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

flagcxResult_t ppu_cudaAdaptorEventCreate(flagcxEvent_t *event,
                                          flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorEventRecord(flagcxEvent_t event,
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

flagcxResult_t ppu_cudaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorEventQuery(flagcxEvent_t event) {
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

flagcxResult_t ppu_cudaAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
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

flagcxResult_t ppu_cudaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                                 size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                              void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                               void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                             void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return flagcxInvalidArgument;

  *dmaBufferSupport = false;
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorMemGetHandleForAddressRange(
    void *handleOut, void *buffer, size_t size, unsigned long long flags) {
  return flagcxNotSupported;
}

flagcxResult_t ppu_cudaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
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

  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                                int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorGetDeviceByPciBusId(int *dev,
                                                  const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorHostRegister(void *ptr, size_t size) {
  DEVCHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorHostUnregister(void *ptr) {
  DEVCHECK(cudaHostUnregister(ptr));
  return flagcxSuccess;
}

// Symmetric memory VMM stubs (not supported on PPU)
flagcxResult_t ppu_cudaAdaptorSymPhysAlloc(void *ptr, size_t size,
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

  DEVCHECK(cuMemRetainAllocationHandle(cuHandle, ptr));

  size_t actualAllocSize = 0;
  DEVCHECK(cuMemGetAddressRange(NULL, &actualAllocSize, (CUdeviceptr)ptr));
  *allocSize = actualAllocSize;

  if (*handleSize < sizeof(int)) {
    free(cuHandle);
    return flagcxInvalidArgument;
  }
  DEVCHECK(cuMemExportToShareableHandle(
      shareableHandle, *cuHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  *handleSize = sizeof(int);
  *physHandle = cuHandle;
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorSymPhysFree(void *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)physHandle;
  cuMemRelease(*cuHandle);
  free(cuHandle);
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                         int selfIndex, void *selfPhysHandle,
                                         size_t allocSize, void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0 || allocSize == 0)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle selfHandle =
      *(CUmemGenericAllocationHandle *)selfPhysHandle;

  size_t totalSize = allocSize * nPeers;

  CUdeviceptr base = 0;
  DEVCHECK(cuMemAddressReserve(&base, totalSize, 0, 0, 0));

  int cudaDev;
  DEVCHECK(cudaGetDevice(&cudaDev));
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  for (int i = 0; i < nPeers; i++) {
    CUmemGenericAllocationHandle peerHandle;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      int fd = *(int *)peerHandles[i];
      DEVCHECK(cuMemImportFromShareableHandle(
          &peerHandle, (void *)(uintptr_t)fd,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }
    CUdeviceptr slot = base + (CUdeviceptr)i * allocSize;
    DEVCHECK(cuMemMap(slot, allocSize, 0, peerHandle, 0));
    DEVCHECK(cuMemSetAccess(slot, allocSize, &accessDesc, 1));
    if (i != selfIndex) {
      cuMemRelease(peerHandle);
    }
  }

  *flatBase = (void *)base;
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorSymFlatUnmap(void *flatBase, size_t allocSize,
                                           int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  CUdeviceptr base = (CUdeviceptr)flatBase;
  size_t totalSize = allocSize * nPeers;
  DEVCHECK(cuMemUnmap(base, totalSize));
  DEVCHECK(cuMemAddressFree(base, totalSize));
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorSymMulticastSupported(int *supported) {
  if (supported == NULL)
    return flagcxInvalidArgument;

  *supported = 0;
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorSymMulticastCreate(size_t allocSize,
                                                 int nLocalDevices,
                                                 const int *localDeviceOrdinals,
                                                 void **mcHandle,
                                                 int *shareableFd) {
  if (mcHandle)
    *mcHandle = NULL;
  if (shareableFd)
    *shareableFd = -1;
  return flagcxNotSupported;
}

flagcxResult_t ppu_cudaAdaptorSymMulticastBind(void *mcHandle, int importFd,
                                               void *physHandle,
                                               size_t allocSize, int localRank,
                                               int nLocalDevices, void **mcBase,
                                               size_t *mcMapSize) {
  if (mcBase)
    *mcBase = NULL;
  if (mcMapSize)
    *mcMapSize = 0;
  return flagcxNotSupported;
}

flagcxResult_t ppu_cudaAdaptorSymMulticastTeardown(void *mcBase,
                                                   size_t mcMapSize) {
  return flagcxSuccess;
}

flagcxResult_t ppu_cudaAdaptorSymMulticastFree(void *mcHandle) {
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor ppu_cudaAdaptor {
  "PPU_CUDA",
      // Basic functions
      ppu_cudaAdaptorDeviceSynchronize, ppu_cudaAdaptorDeviceMemcpy,
      ppu_cudaAdaptorDeviceMemset, ppu_cudaAdaptorDeviceMalloc,
      ppu_cudaAdaptorDeviceFree, ppu_cudaAdaptorSetDevice,
      ppu_cudaAdaptorGetDevice, ppu_cudaAdaptorGetDeviceCount,
      ppu_cudaAdaptorGetVendor, ppu_cudaAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // memHandleInit
      NULL, // memHandleDestroy
      ppu_cudaAdaptorGdrMemAlloc, ppu_cudaAdaptorGdrMemFree,
      NULL, // hostShareMemAlloc
      NULL, // hostShareMemFree
      NULL, // gdrPtrMmap
      NULL, // gdrPtrMunmap
      // Stream functions
      ppu_cudaAdaptorStreamCreate, ppu_cudaAdaptorStreamDestroy,
      ppu_cudaAdaptorStreamCopy, ppu_cudaAdaptorStreamFree,
      ppu_cudaAdaptorStreamSynchronize, ppu_cudaAdaptorStreamQuery,
      ppu_cudaAdaptorStreamWaitEvent, ppu_cudaAdaptorStreamWaitValue64,
      ppu_cudaAdaptorStreamWriteValue64,
      // Event functions
      ppu_cudaAdaptorEventCreate, ppu_cudaAdaptorEventDestroy,
      ppu_cudaAdaptorEventRecord, ppu_cudaAdaptorEventSynchronize,
      ppu_cudaAdaptorEventQuery, ppu_cudaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      ppu_cudaAdaptorIpcMemHandleCreate, ppu_cudaAdaptorIpcMemHandleGet,
      ppu_cudaAdaptorIpcMemHandleOpen, ppu_cudaAdaptorIpcMemHandleClose,
      ppu_cudaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // launchKernel
      NULL, // copyArgsInit
      NULL, // copyArgsFree
      NULL, // launchDeviceFunc
      // Others
      ppu_cudaAdaptorGetDeviceProperties, ppu_cudaAdaptorGetDevicePciBusId,
      ppu_cudaAdaptorGetDeviceByPciBusId, ppu_cudaAdaptorLaunchHostFunc,
      // DMA buffer
      ppu_cudaAdaptorDmaSupport, ppu_cudaAdaptorMemGetHandleForAddressRange,
      ppu_cudaAdaptorHostRegister, ppu_cudaAdaptorHostUnregister,
      // Symmetric memory VMM functions
      ppu_cudaAdaptorSymPhysAlloc, ppu_cudaAdaptorSymPhysFree,
      ppu_cudaAdaptorSymFlatMap, ppu_cudaAdaptorSymFlatUnmap,
      ppu_cudaAdaptorSymMulticastSupported, ppu_cudaAdaptorSymMulticastCreate,
      ppu_cudaAdaptorSymMulticastBind, ppu_cudaAdaptorSymMulticastTeardown,
      ppu_cudaAdaptorSymMulticastFree,
};

#endif // USE_PPU_ADAPTOR
