#include "perf_common.h"

static void collFn(PerfContext &ctx, size_t count) {
  flagcxAllReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype, ctx.op,
                  ctx.comm, ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  const char *envLocRed = getenv("FLAGCX_UNIRUNNER_USE_LOCRED");
  const char *envRingAG = getenv("FLAGCX_UNIRUNNER_USE_RINGAG");
  double factor = (double)(2 * (totalProcs - 1)) / (double)(totalProcs);
  if (envLocRed != NULL && atoi(envLocRed) == 1) {
    factor = 1;
  } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
    factor = (double)(totalProcs - 1) / (double)(totalProcs);
  }
  return factor;
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  size_t typeSize = getFlagcxDataTypeSize(ctx.datatype);
  for (size_t i = 0; i < count; i++) {
    float val = i % 10 * (1 << ctx.proc);
    memcpy((char *)ctx.hello + i * typeSize, &val,
           sizeof(float) < typeSize ? sizeof(float) : typeSize);
  }
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);
  ctx.devHandle->deviceMemcpy(ctx.recvbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, NULL);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  perfTeardown(ctx);
  return 0;
}
