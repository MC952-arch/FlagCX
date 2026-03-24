/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Platform Traits - Compile-time dispatch for platform-level capabilities.
 *
 * PlatformTraits<P> provides:
 *   - Intrin: SIMT intrinsics (lane, activemask, syncwarp, popc, ...)
 *   - Atomic: Scoped atomic operations (load, store, fetchAdd, ...)
 *
 * Each platform (NVIDIA, Cambricon, ...) provides a specialization.
 * DeviceTraits<D> pulls in platform capabilities via using-aliases.
 ************************************************************************/

#ifndef FLAGCX_PLATFORM_TRAITS_H_
#define FLAGCX_PLATFORM_TRAITS_H_

#include "device_utils.h"

// Common enum types used as parameters to PlatformTraits::Atomic methods
typedef enum {
  flagcxDeviceMemoryOrderRelaxed = 0,
  flagcxDeviceMemoryOrderAcquire = 1,
  flagcxDeviceMemoryOrderRelease = 2,
  flagcxDeviceMemoryOrderAcqRel = 3,
  flagcxDeviceMemoryOrderSeqCst = 4
} flagcxDeviceMemoryOrder_t;

typedef enum {
  flagcxDeviceScopeSystem = 0,
  flagcxDeviceScopeDevice = 1,
  flagcxDeviceScopeBlock = 2,
  flagcxDeviceScopeThread = 3
} flagcxDeviceScope_t;

// Primary template — each platform provides a specialization
template <typename Platform>
struct PlatformTraits;

// Include platform specializations
#ifdef USE_NVIDIA_ADAPTOR
#include "nvidia_platform_traits.h"
#endif

// Future:
// #ifdef USE_CAMBRICON_ADAPTOR
// #include "cambricon_platform_traits.h"
// #endif

#endif // FLAGCX_PLATFORM_TRAITS_H_
