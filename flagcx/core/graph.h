/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_GRAPH_H_
#define FLAGCX_GRAPH_H_

#include "device.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <sched.h>

flagcxResult_t flagcxTopoCudaPath(int cudaDev, char** path);

struct flagcxTopoSystem;
// Build the topology
flagcxResult_t flagcxTopoSortSystem(struct flagcxTopoSystem* system);
flagcxResult_t flagcxTopoPrint(struct flagcxTopoSystem* system);

flagcxResult_t flagcxTopoComputePaths(struct flagcxTopoSystem* system, struct flagcxHeteroComm* comm);
void flagcxTopoFree(struct flagcxTopoSystem* system);
flagcxResult_t flagcxTopoTrimSystem(struct flagcxTopoSystem* system, struct flagcxHeteroComm* comm);
flagcxResult_t flagcxTopoComputeP2pChannels(struct flagcxHeteroComm* comm);
flagcxResult_t flagcxTopoGetNvbGpus(struct flagcxTopoSystem* system, int rank, int* nranks, int** ranks);
int flagcxTopoPathAllNVLink(struct flagcxTopoSystem* system);

// Query topology
flagcxResult_t flagcxTopoGetNetDev(struct flagcxHeteroComm* comm, int rank, struct flagcxTopoGraph* graph, int channelId, int peerRank, int64_t* id, int* dev, int* proxyRank);
flagcxResult_t flagcxTopoCheckP2p(struct flagcxTopoSystem* system, int64_t id1, int64_t id2, int* p2p, int *read, int* intermediateRank);
flagcxResult_t flagcxTopoCheckMNNVL(struct flagcxTopoSystem* system, struct flagcxPeerInfo* info1, struct flagcxPeerInfo* info2, int* ret);
flagcxResult_t flagcxTopoCheckGdr(struct flagcxTopoSystem* topo, int64_t busId, int64_t netId, int read, int* useGdr);
flagcxResult_t flagcxTopoNeedFlush(struct flagcxTopoSystem* system, int64_t busId, int* flush);
flagcxResult_t flagcxTopoCheckNet(struct flagcxTopoSystem* system, int64_t id1, int64_t id2, int* net);
int flagcxPxnDisable(struct flagcxHeteroComm* comm);
flagcxResult_t flagcxTopoGetPxnRanks(struct flagcxHeteroComm* comm, int** intermediateRanks, int* nranks);

// Find CPU affinity
flagcxResult_t flagcxTopoGetCpuAffinity(struct flagcxTopoSystem* system, int rank, cpu_set_t* affinity);

#define FLAGCX_TOPO_CPU_ARCH_X86 1
#define FLAGCX_TOPO_CPU_ARCH_POWER 2
#define FLAGCX_TOPO_CPU_ARCH_ARM 3
#define FLAGCX_TOPO_CPU_VENDOR_INTEL 1
#define FLAGCX_TOPO_CPU_VENDOR_AMD 2
#define FLAGCX_TOPO_CPU_VENDOR_ZHAOXIN 3
#define FLAGCX_TOPO_CPU_TYPE_BDW 1
#define FLAGCX_TOPO_CPU_TYPE_SKL 2
#define FLAGCX_TOPO_CPU_TYPE_YONGFENG 1
flagcxResult_t flagcxTopoCpuType(struct flagcxTopoSystem* system, int* arch, int* vendor, int* model);
flagcxResult_t flagcxTopoGetGpuCount(struct flagcxTopoSystem* system, int* count);
flagcxResult_t flagcxTopoGetNetCount(struct flagcxTopoSystem* system, int* count);
flagcxResult_t flagcxTopoGetNvsCount(struct flagcxTopoSystem* system, int* count);
flagcxResult_t flagcxGetLocalNetFromGpu(int gpu, int *dev);
flagcxResult_t flagcxTopoGetLocalGpu(struct flagcxTopoSystem* system, int64_t netId, int* gpuIndex);
flagcxResult_t getLocalNetCountByBw(struct flagcxTopoSystem* system, int gpu, int *count);

#define FLAGCX_TOPO_MAX_NODES 256

// Init search. Needs to be done before calling flagcxTopoCompute
flagcxResult_t flagcxTopoSearchInit(struct flagcxTopoSystem* system);

#define FLAGCX_TOPO_PATTERN_BALANCED_TREE 1   // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
#define FLAGCX_TOPO_PATTERN_SPLIT_TREE 2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
#define FLAGCX_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define FLAGCX_TOPO_PATTERN_RING 4            // Ring
#define FLAGCX_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
struct flagcxTopoGraph {
  // Input / output
  int id; // ring : 0, tree : 1, collnet : 2
  int pattern;
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // Output
  int nChannels;
  float bwIntra;
  float bwInter;
  float latencyInter;
  int typeIntra;
  int typeInter;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS*FLAGCX_TOPO_MAX_NODES];
  int64_t inter[MAXCHANNELS*2];
};
flagcxResult_t flagcxTopoCompute(struct flagcxTopoSystem* system, struct flagcxTopoGraph* graph);

flagcxResult_t flagcxTopoPrintGraph(struct flagcxTopoSystem* system, struct flagcxTopoGraph* graph);
flagcxResult_t flagcxTopoDumpGraphs(struct flagcxTopoSystem* system, int ngraphs, struct flagcxTopoGraph** graphs);

struct flagcxTopoRanks {
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
  int nvlsHeads[MAXCHANNELS];
  int nvlsHeadNum;
};

flagcxResult_t flagcxTopoPreset(struct flagcxHeteroComm* comm, struct flagcxTopoGraph** graphs, struct flagcxTopoRanks* topoRanks);

flagcxResult_t flagcxTopoPostset(struct flagcxHeteroComm* comm, int* firstRanks, int* treePatterns,
    struct flagcxTopoRanks** allTopoRanks, int* rings, struct flagcxTopoGraph** graphs, struct flagcxHeteroComm* parent);

flagcxResult_t flagcxTopoTuneModel(struct flagcxHeteroComm* comm, int minCompCap, int maxCompCap, struct flagcxTopoGraph** graphs);
#include "info.h"
flagcxResult_t flagcxTopoGetAlgoTime(struct flagcxInfo* info, int algorithm, int protocol, int numPipeOps, float* time, bool* backup = NULL);

#endif