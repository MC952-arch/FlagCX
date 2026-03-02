/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Host-side lifecycle management for flagcxDevComm_t.
 * On NVIDIA (NCCL 2.28+): calls pncclDevCommCreate/Destroy via dlsym.
 * On other platforms: returns flagcxNotSupported.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "flagcx_kernel.h"

#ifdef FLAGCX_DEVICE_API_NCCL

#include "dlsymbols.h"
#include "nvidia_adaptor.h"
#include <dlfcn.h>

flagcxResult_t flagcxDevCommCreate(flagcxComm_t comm,
                                   const flagcxDevCommRequirements *reqs,
                                   flagcxDevComm_t *devComm) {
  if (comm == nullptr || reqs == nullptr || devComm == nullptr) {
    return flagcxInvalidArgument;
  }

  flagcxInnerComm_t innerComm = comm->homoComm;
  if (innerComm == nullptr) {
    return flagcxInternalError;
  }

  // Allocate the opaque handle
  flagcxDevComm_t handle =
      (flagcxDevComm_t)malloc(sizeof(struct flagcxDevCommInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevCommInternal));

  // Map opaque FlagCX requirements to NCCL requirements
  ncclDevCommRequirements ncclReqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ncclReqs.lsaBarrierCount = reqs->fields[0];
  ncclReqs.lsaMultimem = reqs->fields[1];
  ncclReqs.railGinBarrierCount = reqs->fields[2];
  ncclReqs.ginSignalCount = reqs->fields[3];

  // Load pncclDevCommCreate via dlsym (consistent with nccl_adaptor.cc)
  using pncclDevCommCreate_t =
      flagcxCustomOpFunc_t<ncclResult_t, ncclComm_t, ncclDevCommRequirements *,
                           ncclDevComm *>;

  void *libHandle = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
  if (!libHandle) {
    free(handle);
    return flagcxInternalError;
  }

  auto fn = reinterpret_cast<pncclDevCommCreate_t>(
      dlsym(libHandle, "pncclDevCommCreate"));
  if (!fn) {
    dlclose(libHandle);
    free(handle);
    return flagcxInternalError;
  }

  ncclResult_t ret = fn(innerComm->base, &ncclReqs, &handle->ncclDev);
  dlclose(libHandle);

  if (ret != ncclSuccess) {
    free(handle);
    return (flagcxResult_t)ret;
  }

  *devComm = handle;
  return flagcxSuccess;
}

flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm,
                                    flagcxDevComm_t devComm) {
  if (devComm == nullptr) {
    return flagcxSuccess;
  }
  if (comm == nullptr) {
    return flagcxInvalidArgument;
  }

  flagcxInnerComm_t innerComm = comm->homoComm;
  if (innerComm == nullptr) {
    free(devComm);
    return flagcxInternalError;
  }

  // Load pncclDevCommDestroy via dlsym
  using pncclDevCommDestroy_t =
      flagcxCustomOpFunc_t<ncclResult_t, ncclComm_t, const ncclDevComm *>;

  void *libHandle = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
  if (libHandle) {
    auto fn = reinterpret_cast<pncclDevCommDestroy_t>(
        dlsym(libHandle, "pncclDevCommDestroy"));
    if (fn) {
      fn(innerComm->base, &devComm->ncclDev);
    }
    dlclose(libHandle);
  }

  free(devComm);
  return flagcxSuccess;
}

#else // !FLAGCX_DEVICE_API_NCCL

flagcxResult_t flagcxDevCommCreate(flagcxComm_t comm,
                                   const flagcxDevCommRequirements *reqs,
                                   flagcxDevComm_t *devComm) {
  return flagcxNotSupported;
}

flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm,
                                    flagcxDevComm_t devComm) {
  return flagcxNotSupported;
}

#endif // FLAGCX_DEVICE_API_NCCL
