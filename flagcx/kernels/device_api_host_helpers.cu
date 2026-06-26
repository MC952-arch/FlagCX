/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-callable helpers that require device-compiler visibility of
 * device-only types (e.g. flagcxDevNet). Compiled by nvcc/hipcc,
 * linked into libflagcx.so, called from flagcx_device.cc via extern "C".
 ************************************************************************/

#include "device_api/flagcx_device_core.h"
#include <new>

extern "C" size_t flagcxDevNetSizeOf() { return sizeof(flagcxDevNet); }

extern "C" void flagcxDevNetConstructArray(void *dst, const void *comm,
                                           int count) {
  const flagcxDevComm *devComm = (const flagcxDevComm *)comm;
  flagcxDevNet *nets = (flagcxDevNet *)dst;
  for (int i = 0; i < count; i++) {
    ::new (&nets[i]) flagcxDevNet(*devComm, i);
  }
}
