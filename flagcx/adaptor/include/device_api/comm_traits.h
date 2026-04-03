/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Comm Traits — Unified compile-time dispatch for device APIs.
 *
 * Architecture:
 *   PlatformTraits<P>         — platform-level: Intrin, Atomic
 *   CommTraits<D>             — backend-level:  Window, Comm, Team, ...
 *   Fallback<PlatformTag>     — common IPC fallback (partial specialization)
 *
 * CommTraits pulls in platform capabilities via using-aliases (not
 * inheritance). Vendor specializations wrap vendor types with member
 * functions. The Fallback partial specialization provides IPC-based
 * types that work with any platform.
 *
 * Selection:
 *   NVIDIA + NCCL > 2.28:    DeviceAPI = CommTraits<NvidiaVendor>
 *   NVIDIA + fallback:       DeviceAPI = CommTraits<Fallback<NvidiaPlatform>>
 *
 * Kernel code uses DeviceAPI::* exclusively, no #ifdef branches.
 ************************************************************************/

#ifndef FLAGCX_COMM_TRAITS_H_
#define FLAGCX_COMM_TRAITS_H_

#include "platform_traits.h"
#include <cstddef>
#include <cstdint>

// Primary template — each backend provides a specialization
template <typename Impl>
struct CommTraits;

// Fallback tag — parameterized by platform for the partial specialization
template <typename PlatformTag>
struct Fallback {};

// ============================================================
// Action types for one-sided operations (needed by traits Transport types).
// Pure POD structs with no device builtins.
// ============================================================
typedef uint32_t flagcxDevTransportSignal_t;
typedef uint32_t flagcxDevTransportCounter_t;

struct flagcxDevTransport_None {};
struct flagcxDevTransport_SignalInc {
  flagcxDevTransportSignal_t signal;
};
struct flagcxDevTransport_SignalAdd {
  flagcxDevTransportSignal_t signal;
  uint64_t value;
};
struct flagcxDevTransport_CounterInc {
  flagcxDevTransportCounter_t counter;
};

// Shared memory descriptor for NIC descriptor optimization.
// Uses void* on all paths; vendor Transport casts to native type in toNccl().
struct flagcxDescriptorSmem {
  void *_impl = nullptr;
};

struct flagcxDevTransport_DescriptorSmem {
  flagcxDescriptorSmem smem;
};

// Fence level enum — available on all tiers for unified barrier API
enum class flagcxTransportFenceLevel { Relaxed };

// ============================================================
// Unified team/barrier tag types.
// Used as both Barrier<Backend, Tag> template parameter
// and as ctor dispatch tags — eliminating the old two-tag redundancy.
// ============================================================
struct flagcxTeamTagIntra {};
struct flagcxTeamTagInter {};
struct flagcxTeamTagWorld {};

// Primary template — each backend provides specializations
template <typename Backend, typename Tag, typename Coop>
struct Barrier;

// Vendor specializations + DeviceAPI selection
#if defined(USE_NVIDIA_ADAPTOR)
#include "nvidia_comm_traits.h"
#elif defined(USE_DU_ADAPTOR)
#include "du_comm_traits.h"
#else
#include "fallback_comm_traits.h"
using DeviceAPI = CommTraits<Fallback<FallbackPlatform>>;
#endif

#endif // FLAGCX_COMM_TRAITS_H_
