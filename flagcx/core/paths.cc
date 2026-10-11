/*************************************************************************
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "comm.h"
#include "core.h"
#include "graph.h"
#include "net.h"
#include "topo.h"

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths

// Remain disabled until topology discovery and cross-process proxy submission
// are connected. An explicit opt-in must never silently rewrite NET paths.
FLAGCX_PARAM(PxnDisable, "PXN_DISABLE", 1);

int flagcxPxnDisable(struct flagcxHeteroComm *comm) {
  (void)comm;
  return flagcxParamPxnDisable();
}

static bool flagcxTopoApuCanUseNet(const flagcxTopoNode *apu,
                                   const flagcxTopoNode *net) {
  return apu->apu.gdrSupport && net->net.gdrSupport;
}

flagcxResult_t flagcxTopoSelectPxnRelay(struct flagcxTopoServer *topoServer,
                                        int apuIndex, int netIndex,
                                        int *relayRank,
                                        int logicalRanksPerNode) {
  if (topoServer == NULL || relayRank == NULL || apuIndex < 0 || netIndex < 0 ||
      apuIndex >= topoServer->nodes[APU].count ||
      netIndex >= topoServer->nodes[NET].count || logicalRanksPerNode < 0)
    return flagcxInvalidArgument;

  const flagcxTopoNode *source = topoServer->nodes[APU].nodes + apuIndex;
  if (source->paths[APU] == NULL || source->paths[NET] == NULL)
    return flagcxInvalidArgument;
  *relayRank = source->apu.rank;
  const flagcxTopoNode *net = topoServer->nodes[NET].nodes + netIndex;
  // NCCL first finds the GPU local to this NIC. Select that GPU by the NIC
  // path alone; the source-to-relay path is an eligibility check, not a score.
  int localGpuIndex = -1;
  float bestBw = 0;
  int bestType = PATH_DIS;
  for (int i = 0; i < topoServer->nodes[APU].count; i++) {
    const flagcxTopoNode *candidate = topoServer->nodes[APU].nodes + i;
    if (candidate->paths[NET] == NULL ||
        FLAGCX_TOPO_ID_SERVER_ID(candidate->id) !=
            FLAGCX_TOPO_ID_SERVER_ID(source->id))
      continue;
    if (logicalRanksPerNode > 0 && candidate->apu.rank / logicalRanksPerNode !=
                                       source->apu.rank / logicalRanksPerNode)
      continue;
    const flagcxTopoPath *netPath = candidate->paths[NET] + netIndex;
    if (netPath->bw > bestBw ||
        (netPath->bw == bestBw && netPath->type < bestType)) {
      localGpuIndex = i;
      bestBw = netPath->bw;
      bestType = netPath->type;
    }
  }
  if (localGpuIndex < 0 || localGpuIndex == apuIndex)
    return flagcxSuccess;
  const flagcxTopoNode *candidate =
      topoServer->nodes[APU].nodes + localGpuIndex;
  const flagcxTopoPath *netPath = candidate->paths[NET] + netIndex;
  const flagcxTopoPath *peerPath = source->paths[APU] + localGpuIndex;
  // Match NCCL's default P2P aggregation policy: the NIC-local GPU may
  // relay even when a direct PCI path has the same bandwidth.
  if (flagcxTopoApuCanUseNet(candidate, net) && peerPath->type <= PATH_CCI &&
      peerPath->bw > 0 && netPath->type <= PATH_PXB && netPath->bw > 0)
    *relayRank = candidate->apu.rank;
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoSelectNetRoute(
    struct flagcxTopoServer *local, struct flagcxTopoServer *remote,
    struct flagcxInterServerTopo *interServer, int sourceRank, int peerRank,
    int remoteNetDev, int *netDev, int *relayRank, int logicalRanksPerNode) {
  if (local == NULL || remote == NULL || interServer == NULL ||
      netDev == NULL || relayRank == NULL || logicalRanksPerNode < 0)
    return flagcxInvalidArgument;

  int sourceIndex = -1;
  for (int i = 0; i < local->nodes[APU].count; ++i) {
    if (local->nodes[APU].nodes[i].apu.rank == sourceRank) {
      sourceIndex = i;
      break;
    }
  }
  if (sourceIndex < 0)
    return flagcxInvalidArgument;

  const flagcxTopoNode *peer = NULL;
  for (int i = 0; i < remote->nodes[APU].count; ++i) {
    if (remote->nodes[APU].nodes[i].apu.rank == peerRank) {
      peer = remote->nodes[APU].nodes + i;
      break;
    }
  }
  if (peer == NULL)
    return flagcxInvalidArgument;

  // Use the NIC advertised by the receiver. Its topology-preferred NIC may
  // differ when FLAGCX_TOPO_FILE overrides the device choice.
  struct flagcxTopoNode *remoteNet = NULL;
  for (int i = 0; i < remote->nodes[NET].count; ++i) {
    if (remote->nodes[NET].nodes[i].net.dev == remoteNetDev) {
      remoteNet = remote->nodes[NET].nodes + i;
      break;
    }
  }
  if (remoteNet == NULL)
    return flagcxInvalidArgument;

  const flagcxTopoNode *source = local->nodes[APU].nodes + sourceIndex;
  if (source->paths[NET] == NULL)
    return flagcxInvalidArgument;

  // Match the receiver's GPU device number to a GPU on the sending node,
  // then use that GPU's preferred local NIC, as NCCL does for P2P NET.
  // CI places two logical nodes on one physical host, so device numbers are
  // normalized within each logical group only in that explicit test mode.
  const int peerDev = logicalRanksPerNode > 0
                          ? peer->apu.dev % logicalRanksPerNode
                          : peer->apu.dev;
  int preferredRank = -1;
  for (int i = 0; i < local->nodes[APU].count; ++i) {
    const flagcxTopoNode *candidate = local->nodes[APU].nodes + i;
    if (FLAGCX_TOPO_ID_SERVER_ID(candidate->id) !=
            FLAGCX_TOPO_ID_SERVER_ID(source->id) ||
        (logicalRanksPerNode > 0 && candidate->apu.rank / logicalRanksPerNode !=
                                        sourceRank / logicalRanksPerNode))
      continue;
    const int candidateDev = logicalRanksPerNode > 0
                                 ? candidate->apu.dev % logicalRanksPerNode
                                 : candidate->apu.dev;
    if (candidateDev == peerDev) {
      preferredRank = candidate->apu.rank;
      break;
    }
  }

  int candidateDevs[FLAGCX_TOPO_MAX_NODES] = {};
  int candidateDevCount = 0;
  auto addCandidateDev = [&](int dev) {
    if (dev < 0)
      return;
    for (int i = 0; i < candidateDevCount; ++i) {
      if (candidateDevs[i] == dev)
        return;
    }
    if (candidateDevCount < FLAGCX_TOPO_MAX_NODES)
      candidateDevs[candidateDevCount++] = dev;
  };
  if (preferredRank >= 0) {
    int preferredDev = -1;
    if (flagcxTopoGetLocalNet(local, preferredRank, &preferredDev) ==
        flagcxSuccess)
      addCandidateDev(preferredDev);
  }
  int sourceDev = -1;
  if (flagcxTopoGetLocalNet(local, sourceRank, &sourceDev) == flagcxSuccess)
    addCandidateDev(sourceDev);
  // FlagCX can describe asymmetric NIC reachability. Preserve a deterministic
  // route fallback after the NCCL-style preferred NIC; never score PXN relays
  // across all NICs by aggregate bandwidth.
  for (int n = 0; n < local->nodes[NET].count; ++n)
    addCandidateDev(local->nodes[NET].nodes[n].net.dev);
  for (int choice = 0; choice < candidateDevCount; ++choice) {
    for (int n = 0; n < local->nodes[NET].count; ++n) {
      const flagcxTopoNode *net = local->nodes[NET].nodes + n;
      if (net->net.dev != candidateDevs[choice])
        continue;
      bool reachable = false;
      if (local->serverId == remote->serverId && net->id == remoteNet->id) {
        reachable = net->net.bw > 0 && remoteNet->net.bw > 0;
      } else {
        auto localRoutes = interServer->routeMap.find(net->net.guid);
        if (localRoutes != interServer->routeMap.end()) {
          auto route = localRoutes->second.find(remoteNet->net.guid);
          reachable = route != localRoutes->second.end() &&
                      route->second != NULL && route->second->interBw > 0;
        }
        // CI's two logical nodes share a physical network. The forced-NET
        // baseline and this end-to-end test exercise the actual NIC pair.
        if (!reachable && logicalRanksPerNode > 0 &&
            local->serverId == remote->serverId)
          reachable = net->net.bw > 0 && remoteNet->net.bw > 0;
      }
      if (!reachable)
        continue;

      int candidateRank = sourceRank;
      FLAGCXCHECK(flagcxTopoSelectPxnRelay(
          local, sourceIndex, n, &candidateRank, logicalRanksPerNode));
      int candidateIndex = sourceIndex;
      for (int i = 0; i < local->nodes[APU].count; ++i) {
        if (local->nodes[APU].nodes[i].apu.rank == candidateRank) {
          candidateIndex = i;
          break;
        }
      }
      const flagcxTopoNode *sender = local->nodes[APU].nodes + candidateIndex;
      const flagcxTopoPath *netPath = sender->paths[NET] + n;
      if (!flagcxTopoApuCanUseNet(sender, net) || netPath->type > PATH_PXB ||
          netPath->bw <= 0)
        continue;
      *netDev = net->net.dev;
      *relayRank = candidateRank;
      return flagcxSuccess;
    }
  }
  return flagcxNotSupported;
}

flagcxResult_t flagcxTopoGetNetDev(struct flagcxHeteroComm *comm, int rank,
                                   struct flagcxTopoGraph *graph, int channelId,
                                   int peerRank, int64_t *id, int *dev,
                                   int *proxyRank) {
  (void)graph;
  (void)channelId;
  if (comm == NULL || id == NULL || dev == NULL || proxyRank == NULL)
    return flagcxInvalidArgument;
  *dev = comm->netDev;
  *proxyRank = rank;
  *id = -1;
  if (comm->topoServer == NULL)
    return flagcxSuccess;
  for (int n = 0; n < comm->topoServer->nodes[NET].count; ++n) {
    const flagcxTopoNode *net = comm->topoServer->nodes[NET].nodes + n;
    if (net->net.dev == *dev) {
      *id = net->id;
      break;
    }
  }
  // The transport still uses process-local operations and buffers. Exposing
  // another proxy rank here before remote submission is implemented would
  // silently corrupt a live send connection.
  (void)peerRank;
  return flagcxSuccess;
}

flagcxResult_t
flagcxTopoGetIntermediateRank(struct flagcxTopoServer *topoServer, int rank,
                              int64_t netId, int *intermediateRank) {
  if (topoServer == NULL || intermediateRank == NULL)
    return flagcxInvalidArgument;
  int apuIndex = -1;
  int netIndex = -1;
  for (int i = 0; i < topoServer->nodes[APU].count; ++i) {
    if (topoServer->nodes[APU].nodes[i].apu.rank == rank) {
      apuIndex = i;
      break;
    }
  }
  for (int i = 0; i < topoServer->nodes[NET].count; ++i) {
    if (topoServer->nodes[NET].nodes[i].id == netId) {
      netIndex = i;
      break;
    }
  }
  if (apuIndex < 0 || netIndex < 0)
    return flagcxInvalidArgument;
  const flagcxTopoNode *source = topoServer->nodes[APU].nodes + apuIndex;
  if (source->paths[NET] == NULL)
    return flagcxInvalidArgument;
  const flagcxTopoPath *path = source->paths[NET] + netIndex;
  if (path->type != PATH_PXN) {
    *intermediateRank = rank;
    return flagcxSuccess;
  }
  for (int i = 0; i < path->count; ++i) {
    if (path->list[i] == NULL || path->list[i]->remNode == NULL)
      return flagcxInternalError;
    const flagcxTopoNode *node = path->list[i]->remNode;
    if (node->type == APU && node != source &&
        FLAGCX_TOPO_ID_SERVER_ID(node->id) ==
            FLAGCX_TOPO_ID_SERVER_ID(source->id)) {
      *intermediateRank = node->apu.rank;
      return flagcxSuccess;
    }
  }
  return flagcxInternalError;
}

struct flagcxTopoNodeList {
  struct flagcxTopoNode *list[FLAGCX_TOPO_MAX_NODES];
  int count;
};

static flagcxResult_t getPath(struct flagcxTopoServer *topoServer,
                              struct flagcxTopoNode *node, int t, int64_t id,
                              struct flagcxTopoPath **path) {
  for (int i = 0; i < topoServer->nodes[t].count; i++) {
    if (topoServer->nodes[t].nodes[i].id == id) {
      *path = node->paths[t] + i;
      return flagcxSuccess;
    }
  }
  WARN("Could not find node of type %d id %lx", t, id);
  return flagcxInternalError;
}

static flagcxResult_t flagcxTopoSetPaths(struct flagcxTopoNode *baseNode,
                                         struct flagcxTopoServer *topoServer) {
  if (baseNode->paths[baseNode->type] == NULL) {
    FLAGCXCHECK(flagcxCalloc(baseNode->paths + baseNode->type,
                             topoServer->nodes[baseNode->type].count));
    for (int i = 0; i < topoServer->nodes[baseNode->type].count; i++)
      baseNode->paths[baseNode->type][i].type = PATH_DIS;
  }

  // breadth-first search to set all paths to that node in the system
  struct flagcxTopoNodeList nodeList;
  struct flagcxTopoNodeList nextNodeList = {{0}, 0};
  nodeList.count = 1;
  nodeList.list[0] = baseNode;
  struct flagcxTopoPath *basePath;
  FLAGCXCHECK(
      getPath(topoServer, baseNode, baseNode->type, baseNode->id, &basePath));
  basePath->count = 0;
  basePath->bw = LOC_BW;
  basePath->type = PATH_LOC;

  while (nodeList.count) {
    nextNodeList.count = 0;
    for (int n = 0; n < nodeList.count; n++) {
      struct flagcxTopoNode *node = nodeList.list[n];
      // APU paths may end at a peer APU, but a peer must not become an
      // implicit transit hop when computing the receiver's local NIC path.
      if (node->type == APU && node != baseNode)
        continue;
      struct flagcxTopoPath *path;
      FLAGCXCHECK(
          getPath(topoServer, node, baseNode->type, baseNode->id, &path));
      for (int l = 0; l < node->nlinks; l++) {
        struct flagcxTopoLink *link = node->links + l;
        struct flagcxTopoNode *remNode = link->remNode;
        if (remNode->paths[baseNode->type] == NULL) {
          FLAGCXCHECK(flagcxCalloc(remNode->paths + baseNode->type,
                                   topoServer->nodes[baseNode->type].count));
          for (int i = 0; i < topoServer->nodes[baseNode->type].count; i++)
            remNode->paths[baseNode->type][i].type = PATH_DIS;
        }
        struct flagcxTopoPath *remPath;
        FLAGCXCHECK(getPath(topoServer, remNode, baseNode->type, baseNode->id,
                            &remPath));
        float bw = std::min(path->bw, link->bw);

        // allow routing through a APU only as 1 hop (not supported)

        if ((remPath->bw == 0 || remPath->count > path->count) &&
            remPath->bw < bw) {
          // Find reverse link
          for (int l = 0; l < remNode->nlinks; l++) {
            if (remNode->links[l].remNode == node &&
                remNode->links[l].type == link->type) {
              remPath->list[0] = remNode->links + l;
              break;
            }
          }
          if (remPath->list[0] == NULL) {
            WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to "
                 "node %d/%lx",
                 remNode->type, remNode->id, remNode->nlinks, node->type,
                 node->id);
            return flagcxInternalError;
          }
          // Copy the rest of the path
          for (int i = 0; i < path->count; i++)
            remPath->list[i + 1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->bw = bw;

          // Start with path type = link type. PATH and LINK types are supposed
          // to match. Don't consider LINK_NET as we only care about the
          // NIC->APU path.
          int type = link->type == LINK_NET ? LINK_LOC : link->type;
          // Differentiate between one and multiple PCI switches
          if (node->type == PCI && remNode->type == PCI)
            type = PATH_PXB;
          // Consider a path going through the CPU as PATH_PHB
          if (link->type == LINK_PCI &&
              (node->type == CPU || link->remNode->type == CPU))
            type = PATH_PHB;
          // Set 1 hop CCI as CCB
          // if (node->type == APU && path->type == PATH_CCI && type == PATH_CCI
          // && remPath->count > 1) type = PATH_CCB;

          remPath->type = std::max(path->type, type);

          // Add to the list for the next iteration if not already in the list
          int i;
          for (i = 0; i < nextNodeList.count; i++)
            if (nextNodeList.list[i] == remNode)
              break;
          if (i == nextNodeList.count)
            nextNodeList.list[nextNodeList.count++] = remNode;
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return flagcxSuccess;
}

// Remove/free all paths
static void flagcxTopoRemovePaths(struct flagcxTopoServer *topoServer) {
  for (int t1 = 0; t1 < FLAGCX_TOPO_NODE_TYPES; t1++) {
    for (int n = 0; n < topoServer->nodes[t1].count; n++) {
      struct flagcxTopoNode *node = topoServer->nodes[t1].nodes + n;
      for (int t2 = 0; t2 < FLAGCX_TOPO_NODE_TYPES; t2++) {
        if (node->paths[t2])
          free(node->paths[t2]);
        node->paths[t2] = NULL;
      }
    }
  }
}

// This is a tailored version of the original one.
flagcxResult_t flagcxTopoComputePaths(struct flagcxTopoServer *topoServer,
                                      struct flagcxHeteroComm *comm) {
  // Precompute paths between GPUs/NICs.

  // Remove everything in case we're re-computing
  INFO(FLAGCX_GRAPH, "Removing paths");
  flagcxTopoRemovePaths(topoServer);

  // Set direct paths to CPUs. We need them in many cases.
  INFO(FLAGCX_GRAPH, "Setting paths to CPUs");
  for (int c = 0; c < topoServer->nodes[CPU].count; c++) {
    FLAGCXCHECK(
        flagcxTopoSetPaths(topoServer->nodes[CPU].nodes + c, topoServer));
  }

  // Set direct paths to GPUs.
  INFO(FLAGCX_GRAPH, "Setting paths to APUs");
  for (int g = 0; g < topoServer->nodes[APU].count; g++) {
    FLAGCXCHECK(
        flagcxTopoSetPaths(topoServer->nodes[APU].nodes + g, topoServer));
  }

  // Set direct paths to NICs.
  INFO(FLAGCX_GRAPH, "Setting paths to NICs");
  for (int n = 0; n < topoServer->nodes[NET].count; n++) {
    INFO(FLAGCX_GRAPH, "setting paths to net node [%d]", n);
    FLAGCXCHECK(
        flagcxTopoSetPaths(topoServer->nodes[NET].nodes + n, topoServer));
  }

  // TODO: Update paths for NICs (no GPU Direct, PXN, ...)
  return flagcxSuccess;
}

static void printNodePaths(struct flagcxTopoServer *topoServer,
                           struct flagcxTopoNode *node) {
  const int linesize = 1024;
  char line[linesize];
#ifdef ENABLE_TRACE
  INFO(FLAGCX_GRAPH, "Paths from %s/%lx-%lx :", topoNodeTypeStr[node->type],
       FLAGCX_TOPO_ID_SERVER_ID(node->id), FLAGCX_TOPO_ID_LOCAL_ID(node->id));
#else
  snprintf(line, linesize, "%s/%lx-%lx :", topoNodeTypeStr[node->type],
           FLAGCX_TOPO_ID_SERVER_ID(node->id),
           FLAGCX_TOPO_ID_LOCAL_ID(node->id));
  int offset = strlen(line);
#endif
  for (int t = 0; t < FLAGCX_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL)
      continue;
    for (int n = 0; n < topoServer->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i = 0; i < node->paths[t][n].count; i++) {
        struct flagcxTopoLink *link = node->paths[t][n].list[i];
        struct flagcxTopoNode *remNode = link->remNode;
        snprintf(line + offset, linesize - offset, "--%s(%g)->%s/%lx-%lx",
                 topoLinkTypeStr[link->type], link->bw,
                 topoNodeTypeStr[remNode->type],
                 FLAGCX_TOPO_ID_SERVER_ID(remNode->id),
                 FLAGCX_TOPO_ID_LOCAL_ID(remNode->id));
        offset = strlen(line);
      }
      INFO(FLAGCX_GRAPH, "%s (%f)", line, node->paths[t][n].bw);
#else
      snprintf(line + offset, linesize - offset, "%s/%lx-%lx (%d/%.1f/%s) ",
               topoNodeTypeStr[t],
               FLAGCX_TOPO_ID_SERVER_ID(system->nodes[t].nodes[n].id),
               FLAGCX_TOPO_ID_LOCAL_ID(system->nodes[t].nodes[n].id),
               node->paths[t][n].count, node->paths[t][n].bw,
               topoPathTypeStr[node->paths[t][n].type]);
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

flagcxResult_t flagcxTopoPrintPaths(struct flagcxTopoServer *topoServer) {
  for (int i = 0; i < topoServer->nodes[APU].count; i++) {
    printNodePaths(topoServer, topoServer->nodes[APU].nodes + i);
  }
  for (int i = 0; i < topoServer->nodes[NET].count; i++) {
    printNodePaths(topoServer, topoServer->nodes[NET].nodes + i);
  }
  return flagcxSuccess;
}

void flagcxTopoFree(struct flagcxTopoServer *topoServer) {
  flagcxTopoRemovePaths(topoServer);
  free(topoServer);
}

void flagcxInterServerTopoFree(struct flagcxInterServerTopo *interServerTopo) {
  for (int i = 0; i < interServerTopo->numServers; i++) {
    flagcxTopoRemovePaths(interServerTopo->servers + i);
  }
  free(interServerTopo->servers);
  // free interserver routes
  for (auto localRankIter = interServerTopo->routeMap.begin();
       localRankIter != interServerTopo->routeMap.end(); ++localRankIter) {
    auto remoteRoutes = localRankIter->second;
    for (auto remoteRankIter = remoteRoutes.begin();
         remoteRankIter != remoteRoutes.end(); ++remoteRankIter) {
      free(remoteRankIter->second);
    }
  }
  delete interServerTopo;
}
