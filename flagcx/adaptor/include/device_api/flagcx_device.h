/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API - Template wrappers and inline functions for
 * platform-agnostic device-side communication primitives.
 *
 * This is an umbrella header that includes both device-side types
 * (flagcx_device_core.h) and host-side internal structs
 * (flagcx_device_internal.h).
 *
 * For LLVM bitcode compilation, only flagcx_device_core.h is included.
 * For normal builds, both headers are included.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_API_H_
#define FLAGCX_DEVICE_API_H_

// Host-side internal structs (must come first — core constructors use them)
#ifndef __clang_llvm_bitcode_lib__
#include "flagcx_device_internal.h"
#endif

// Device-side types and inline functions (bitcode-safe)
#include "flagcx_device_core.h"

#endif // FLAGCX_DEVICE_API_H_
