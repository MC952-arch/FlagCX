/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Platform-specific type list for NVIDIA CUDA.
 * FN(SUFFIX, TYPE) is expanded for each supported device type.
 * Used by flagcx_device_wrapper.h for typed load/store X-macro generation.
 ************************************************************************/
#ifndef FLAGCX_DEVICE_TYPES_H_
#define FLAGCX_DEVICE_TYPES_H_

#define FLAGCX_REPT_FOR_DEVICE_TYPES(FN)                                       \
  FN(F16, __half)                                                              \
  FN(BF16, __nv_bfloat16)                                                      \
  FN(F32, float)                                                               \
  FN(F64, double)                                                              \
  FN(I8, int8_t)                                                               \
  FN(I32, int32_t)                                                             \
  FN(U32, uint32_t)                                                            \
  FN(I64, int64_t)                                                             \
  FN(U64, uint64_t)

#endif /* FLAGCX_DEVICE_TYPES_H_ */
