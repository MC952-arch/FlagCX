/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device Traits — Unified compile-time dispatch for device APIs.
 *
 * Architecture:
 *   PlatformTraits<P>         — platform-level: Intrin, Atomic
 *   DeviceTraits<D>           — backend-level:  Window, DevComm, Team, ...
 *   Fallback<PlatformTag>     — common IPC fallback (partial specialization)
 *
 * DeviceTraits pulls in platform capabilities via using-aliases (not
 * inheritance). Vendor specializations wrap vendor types with member
 * functions. The Fallback partial specialization provides IPC-based
 * types that work with any platform.
 *
 * Selection:
 *   NVIDIA + NCCL > 2.28:   DeviceAPI = DeviceTraits<NvidiaVendor>
 *   NVIDIA + fallback:       DeviceAPI = DeviceTraits<Fallback<NvidiaPlatform>>
 *
 * Kernel code uses DeviceAPI::* exclusively, no #ifdef branches.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_TRAITS_H_
#define FLAGCX_DEVICE_TRAITS_H_

#include "platform_traits.h"
#include <cstddef>
#include <cstdint>

// Primary template — each backend provides a specialization
template <typename Impl>
struct DeviceTraits;

// Fallback tag — parameterized by platform for the partial specialization
template <typename PlatformTag>
struct Fallback {};

// Common fallback partial specialization (IPC-based, works for any platform)
#include "fallback_device_traits.h"

// Vendor specializations + DeviceAPI selection
#ifdef USE_NVIDIA_ADAPTOR
#include "nvidia_device_traits.h"
#endif

// Future:
// #ifdef USE_CAMBRICON_ADAPTOR
// #include "cambricon_device_traits.h"
// #endif

#endif // FLAGCX_DEVICE_TRAITS_H_
