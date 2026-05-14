/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Kernel API - Umbrella header for device triggers and host lifecycle.
 *
 * This header includes both device-side types (flagcx_kernel_core.h) and
 * host-side functions (flagcx_kernel_internal.h).
 *
 * For LLVM bitcode compilation, only flagcx_kernel_core.h is included.
 * For normal builds, both headers are included.
 ************************************************************************/

#ifndef FLAGCX_KERNEL_H_
#define FLAGCX_KERNEL_H_

// Device-side types and constants (bitcode-safe)
#include "flagcx_kernel_core.h"

// Host-side functions and lifecycle (needs adaptor.h)
#ifndef __clang_llvm_bitcode_lib__
#include "flagcx_kernel_internal.h"
#endif

#endif // FLAGCX_KERNEL_H_
