/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * P2P Engine performance benchmark.
 *
 * Exercises the FlagCX P2P Engine one-sided RDMA APIs directly:
 *   - flagcxP2pEngineRead  (RDMA GET)
 *   - flagcxP2pEngineWrite (RDMA PUT)
 *
 * Uses two MPI ranks with the RPC control-plane path:
 *   Rank 0 = server (target), Rank 1 = client (initiator).
 *   Both register GPU buffers, start RPC servers, connect, then the
 *   client initiates reads/writes against the server's buffer.
 *
 * Usage:
 *   mpirun -np 2 perf_p2p_engine -b 4K -e 256M -f 2 -n 20
 *
 * Environment:
 *   FLAGCX_P2P_PERF_OP=read|write|both  (default: both)
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_p2p.h"
#include "tools.h"

#include <cassert>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>

static void fatal(const char *msg, int rank) {
  fprintf(stderr, "[rank %d] FATAL: %s\n", rank, msg);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static void fatalIf(bool cond, const char *msg, int rank) {
  if (cond)
    fatal(msg, rank);
}

enum PerfOp { OP_READ = 1, OP_WRITE = 2, OP_BOTH = 3 };

static PerfOp getOpMode() {
  const char *env = getenv("FLAGCX_P2P_PERF_OP");
  if (env == nullptr)
    return OP_BOTH;
  if (strcmp(env, "read") == 0)
    return OP_READ;
  if (strcmp(env, "write") == 0)
    return OP_WRITE;
  return OP_BOTH;
}

static bool pollTransferDone(FlagcxP2pConn *conn, uint64_t transferId,
                             int timeoutMs) {
  if (transferId == 0)
    return true;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    if (flagcxP2pEngineXferStatus(conn, transferId))
      return true;
    std::this_thread::yield();
  }
  return flagcxP2pEngineXferStatus(conn, transferId);
}

static void printHeader(int rank, const char *opName) {
  if (rank == 0) {
    printf("\n");
    printf("# P2P Engine: %s\n", opName);
    printf("#%14s %12s %12s\n", "Size (bytes)", "Latency (us)", "BW (GB/s)");
  }
}

static void printResult(int rank, size_t size, double latencyUs,
                        double bandwidth) {
  if (rank == 0) {
    printf("%15zu %12.2f %12.3f\n", size, latencyUs, bandwidth);
  }
}

static void benchmarkOp(FlagcxP2pConn *conn, FlagcxP2pMr localMr,
                        void *localBuf, uint64_t remoteVa, size_t maxBytes,
                        size_t minBytes, int stepFactor, int numWarmupIters,
                        int numIters, int worldRank, bool isRead) {
  const char *opName = isRead ? "READ (RDMA GET)" : "WRITE (RDMA PUT)";
  printHeader(worldRank, opName);

  // Only rank 1 (client) initiates transfers
  const bool isInitiator = (worldRank == 1);

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    if (size > UINT32_MAX) {
      if (worldRank == 0)
        printf("# Skipping size %zu (exceeds uint32_t desc limit)\n", size);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      continue;
    }

    FlagcxP2pRdmaDesc desc = {};
    if (isInitiator) {
      int ret = flagcxP2pEngineMakeDesc(conn, remoteVa, (uint32_t)size, &desc);
      fatalIf(ret != 0, "flagcxP2pEngineMakeDesc failed", worldRank);

      // Warmup
      for (int i = 0; i < numWarmupIters; i++) {
        uint64_t transferId = 0;
        if (isRead) {
          ret = flagcxP2pEngineRead(conn, localMr, localBuf, size, desc,
                                    &transferId);
        } else {
          ret = flagcxP2pEngineWrite(conn, localMr, localBuf, size, desc,
                                     &transferId);
        }
        fatalIf(ret != 0, "warmup transfer failed", worldRank);
        fatalIf(!pollTransferDone(conn, transferId, 10000),
                "warmup transfer timed out", worldRank);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Timed iterations (only initiator)
    double elapsed = 0.0;
    if (isInitiator) {
      timer tim;
      for (int i = 0; i < numIters; i++) {
        uint64_t transferId = 0;
        int ret;
        if (isRead) {
          ret = flagcxP2pEngineRead(conn, localMr, localBuf, size, desc,
                                    &transferId);
        } else {
          ret = flagcxP2pEngineWrite(conn, localMr, localBuf, size, desc,
                                     &transferId);
        }
        fatalIf(ret != 0, "timed transfer failed", worldRank);
        fatalIf(!pollTransferDone(conn, transferId, 10000),
                "timed transfer timed out", worldRank);
      }
      elapsed = tim.elapsed();
    }

    // Sync timing across ranks
    double avgLatency = elapsed / (numIters > 0 ? numIters : 1);
    MPI_Allreduce(MPI_IN_PLACE, &avgLatency, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    double latencyUs = avgLatency * 1e6;
    double bandwidth = (double)size / avgLatency / 1e9;
    printResult(worldRank, size, latencyUs, bandwidth);

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int numIters = args.getTestIters();

  int worldRank = 0, worldSize = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  if (worldSize != 2) {
    if (worldRank == 0)
      fprintf(stderr, "perf_p2p_engine requires exactly 2 MPI ranks.\n");
    MPI_Finalize();
    return 1;
  }

  PerfOp opMode = getOpMode();

  // Initialize device handle and set GPU
  flagcxDeviceHandle_t devHandle = nullptr;
  fatalIf(flagcxDeviceHandleInit(&devHandle) != flagcxSuccess,
          "flagcxDeviceHandleInit failed", worldRank);

  int nGpu = 0;
  devHandle->getDeviceCount(&nGpu);
  fatalIf(nGpu <= 0, "No GPU devices found", worldRank);
  devHandle->setDevice(worldRank % nGpu);

  // Allocate GPU buffer using flagcxMemAlloc (GDR-capable)
  void *gpuBuf = nullptr;
  fatalIf(flagcxMemAlloc(&gpuBuf, maxBytes) != flagcxSuccess,
          "flagcxMemAlloc failed", worldRank);

  // Create P2P engine
  FlagcxP2pEngine *engine = flagcxP2pEngineCreate();
  fatalIf(engine == nullptr, "flagcxP2pEngineCreate failed", worldRank);

  // Register the GPU buffer
  FlagcxP2pMr mr = 0;
  fatalIf(flagcxP2pEngineReg(engine, reinterpret_cast<uintptr_t>(gpuBuf),
                             maxBytes, mr) != 0,
          "flagcxP2pEngineReg failed", worldRank);

  // Start RPC server
  fatalIf(flagcxP2pEngineStartRpcServer(engine) != 0,
          "flagcxP2pEngineStartRpcServer failed", worldRank);

  // Get RPC port
  int rpcPort = flagcxP2pEngineGetRpcPort(engine);
  fatalIf(rpcPort < 0, "flagcxP2pEngineGetRpcPort failed", worldRank);

  // Get metadata to extract local IP
  char *metadataRaw = nullptr;
  fatalIf(flagcxP2pEngineGetMetadata(engine, &metadataRaw) != 0,
          "flagcxP2pEngineGetMetadata failed", worldRank);

  // Parse IP from metadata format "ip:rdma_port?gpu_index?notif_port"
  std::string metaStr(metadataRaw);
  free(metadataRaw);
  size_t firstSep = metaStr.find('?');
  std::string endpoint = metaStr.substr(0, firstSep);
  size_t lastColon = endpoint.rfind(':');
  std::string localIp = endpoint.substr(0, lastColon);

  // Build session string "ip:rpc_port"
  char localSession[256];
  snprintf(localSession, sizeof(localSession), "%s:%d", localIp.c_str(),
           rpcPort);

  // Exchange session strings via MPI
  char remoteSession[256] = {};
  if (worldRank == 0) {
    MPI_Send(localSession, 256, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(remoteSession, 256, MPI_CHAR, 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(remoteSession, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Send(localSession, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }

  // Connect to peer
  FlagcxP2pConn *conn = flagcxP2pEngineGetConn(engine, remoteSession);
  fatalIf(conn == nullptr, "flagcxP2pEngineGetConn failed", worldRank);

  // Exchange remote buffer VA
  uint64_t localVa = reinterpret_cast<uint64_t>(gpuBuf);
  uint64_t remoteVa = 0;
  if (worldRank == 0) {
    MPI_Send(&localVa, 1, MPI_UINT64_T, 1, 1, MPI_COMM_WORLD);
    MPI_Recv(&remoteVa, 1, MPI_UINT64_T, 1, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(&remoteVa, 1, MPI_UINT64_T, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Send(&localVa, 1, MPI_UINT64_T, 0, 1, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (worldRank == 0) {
    printf("P2P Engine Perf Benchmark\n");
    printf("  Local session:  %s\n", localSession);
    printf("  Remote session: %s\n", remoteSession);
    printf("  Buffer size:    %zu bytes\n", maxBytes);
    printf("  Iterations:     %d (warmup: %d)\n", numIters, numWarmupIters);
  }

  // Both ranks run benchmarkOp; only rank 1 (client) initiates transfers,
  // rank 0 (server) participates in MPI collectives for timing sync.
  if (opMode & OP_READ) {
    benchmarkOp(conn, mr, gpuBuf, remoteVa, maxBytes, minBytes, stepFactor,
                numWarmupIters, numIters, worldRank, true);
  }
  if (opMode & OP_WRITE) {
    benchmarkOp(conn, mr, gpuBuf, remoteVa, maxBytes, minBytes, stepFactor,
                numWarmupIters, numIters, worldRank, false);
  }

  // Cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  flagcxP2pEngineMrDestroy(engine, mr);
  flagcxP2pEngineDestroy(engine);
  flagcxMemFree(gpuBuf);
  flagcxDeviceHandleFree(devHandle);
  MPI_Finalize();

  return 0;
}
