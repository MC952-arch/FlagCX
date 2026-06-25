#include "perf_common.h"

static void collFn(PerfContext &ctx, size_t count) {
  int recvPeer = (ctx.proc - 1 + ctx.totalProcs) % ctx.totalProcs;
  int sendPeer = (ctx.proc + 1) % ctx.totalProcs;
  flagcxGroupStart(ctx.comm);
  flagcxSend(ctx.sendbuff, count, ctx.datatype, sendPeer, ctx.comm, ctx.stream);
  flagcxRecv(ctx.recvbuff, count, ctx.datatype, recvPeer, ctx.comm, ctx.stream);
  flagcxGroupEnd(ctx.comm);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn);
  perfTeardown(ctx);
  return 0;
}
