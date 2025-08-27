#ifndef FLAGCX_LAUNCH_KERNEL_H_
#define FLAGCX_LAUNCH_KERNEL_H_
#pragma once
#include "flagcx.h"
typedef void (*flagcxLaunchFunc_t)(flagcxStream_t, void *);
#include "adaptor.h"
#include "debug.h"
#include "param.h"
#include "topo.h"
#include "utils.h"
#include <dlfcn.h>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

flagcxResult_t loadKernelSymbol(const char *path, const char *name,
                                flagcxLaunchFunc_t *fn);

#ifdef __cplusplus
}
#endif

struct flagcxFuncArgs {
  flagcxStream_t stream;
  flagcxEvent_t event;
  bool *recorded; // Indicates if the op has been recorded by the event
  void *value;    // May point to either hlArgs or dlArgs
};

extern flagcxLaunchFunc_t deviceAsyncStore;
extern flagcxLaunchFunc_t deviceAsyncLoad;

void cpuAsyncStore(void *_args);
void cpuAsyncLoad(void *_args);

#endif