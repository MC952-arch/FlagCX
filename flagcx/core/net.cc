#include "net.h"
#include "adaptor.h"
#include "device.h"
#include "proxy.h"
#include "reg_pool.h"

#include <errno.h>
#include <string.h>
#include <string>

int64_t flagcxNetBufferSize;
int64_t flagcxNetChunkSize;
int64_t flagcxNetChunks;

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
// Use adaptor system for all network types
struct flagcxNetAdaptor *flagcxNetAdaptors[3] = {
    nullptr, getUnifiedNetAdaptor(IBRC), getUnifiedNetAdaptor(SOCKET)};
enum flagcxNetState {
  flagcxNetStateInit = 0,
  flagcxNetStateEnabled = 1,
  flagcxNetStateDisabled = 2
};
enum flagcxNetState flagcxNetStates[3] = {
    flagcxNetStateInit, flagcxNetStateInit, flagcxNetStateInit};

flagcxResult_t flagcxNetCheckDeviceVersion(struct flagcxHeteroComm *comm,
                                           struct flagcxNetAdaptor *net,
                                           int dev) {
  flagcxNetProperties_v8_t props;

  FLAGCXCHECK(net->getProperties(dev, (void *)&props));
  flagcxNetDeviceType type = props.netDeviceType;
  if (type)
    switch (type) {
      case FLAGCX_NET_DEVICE_UNPACK:
        if (props.netDeviceVersion == FLAGCX_NET_DEVICE_UNPACK_VERSION) {
          INFO(FLAGCX_INIT,
               "Using FLAGCX_NET_DEVICE_UNPACK net plugin version %d",
               props.netDeviceVersion);
          return flagcxSuccess;
        } else {
          WARN("FLAGCX_DEVICE_UNPACK plugin has incompatible version %d, this "
               "flagcx build is compatible with %d, not using it",
               props.netDeviceVersion, FLAGCX_NET_DEVICE_UNPACK_VERSION);
          return flagcxInternalError;
        }
      default:
        WARN("Unknown device code index");
        return flagcxInternalError;
    }

  INFO(FLAGCX_INIT, "Using non-device net plugin version %d",
       props.netDeviceVersion);
  return flagcxSuccess;
}

static flagcxResult_t netGetState(int i, enum flagcxNetState *state) {
  pthread_mutex_lock(&netLock);
  if (flagcxNetStates[i] == flagcxNetStateInit) {
    int ndev;
    if (flagcxNetAdaptors[i] == nullptr) {
      flagcxNetStates[i] = flagcxNetStateDisabled;
    } else if (flagcxNetAdaptors[i]->init() != flagcxSuccess) {
      flagcxNetStates[i] = flagcxNetStateDisabled;
    } else if (flagcxNetAdaptors[i]->devices(&ndev) != flagcxSuccess ||
               ndev <= 0) {
      flagcxNetStates[i] = flagcxNetStateDisabled;
    } else {
      flagcxNetStates[i] = flagcxNetStateEnabled;
    }
  }
  *state = flagcxNetStates[i];
  pthread_mutex_unlock(&netLock);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetInit(struct flagcxHeteroComm *comm) {
  // Initialize main communication network
  const char *netName;
  bool ok = false;

  const char *forceSocketEnv = getenv("FLAGCX_FORCE_NET_SOCKET");
  bool forceSocket = (forceSocketEnv && atoi(forceSocketEnv) == 1);

  netName = comm->config.netName;

  if (forceSocket) {
    // Force socket network usage
    for (int i = 2; i >= 0; i--) {
      if (flagcxNetAdaptors[i] == nullptr)
        continue;
      if (flagcxNetAdaptors[i] != getUnifiedNetAdaptor(SOCKET))
        continue;
      enum flagcxNetState state;
      FLAGCXCHECK(netGetState(i, &state));
      if (state != flagcxNetStateEnabled)
        continue;
      if (netName && strcasecmp(netName, flagcxNetAdaptors[i]->name) != 0)
        continue;
      if (flagcxSuccess !=
          flagcxNetCheckDeviceVersion(comm, flagcxNetAdaptors[i], 0)) {
        continue;
      }

      comm->netAdaptor = flagcxNetAdaptors[i];
      ok = true;

      break;
    }
  } else {
    // Normal network selection order (IBUC first when enabled, then IBRC, then
    // socket)
    for (int i = 0; i < 3; i++) {
      if (flagcxNetAdaptors[i] == nullptr)
        continue;
      enum flagcxNetState state;
      FLAGCXCHECK(netGetState(i, &state));
      if (state != flagcxNetStateEnabled)
        continue;
      if (netName && strcasecmp(netName, flagcxNetAdaptors[i]->name) != 0)
        continue;
      if (flagcxSuccess !=
          flagcxNetCheckDeviceVersion(comm, flagcxNetAdaptors[i], 0)) {
        continue;
      }

      comm->netAdaptor = flagcxNetAdaptors[i];
      ok = true;

      break;
    }
  }

  if (!ok) {
    WARN("Error: network %s not found.", netName ? netName : "");
    return flagcxInvalidUsage;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxProxySend(sendNetResources *resources, void *data,
                               size_t size, flagcxProxyArgs *args) {
  if (!args->semaphore->pollStart()) {
    return flagcxSuccess;
  }
  if (args->done) {
    return flagcxSuccess;
  }
  if (args->transmitted < args->chunkSteps) {
    int stepMask = args->sendStepMask;
    if (args->waitCopy < args->chunkSteps &&
        args->waitCopy - args->transmitted < flagcxNetChunks) {
      TRACE(FLAGCX_PROXY,
            "ProxySend BP1 waitCopy %d, copied %d, posted %d, transmitted %d",
            args->waitCopy, args->copied, args->posted, args->transmitted);
      int step = args->waitCopy & stepMask;
      args->subs[step].stepSize =
          std::min(args->chunkSize, size - args->totalCopySize);
      if (!args->regBufFlag) {
        args->subs[step].stepBuff =
            resources->buffers[0] + (flagcxNetChunkSize * step);
        if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
          FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
              args->subs[step].stepBuff, (char *)data + args->totalCopySize,
              args->subs[step].stepSize, flagcxMemcpyDeviceToDevice,
              resources->cpStream, args->subs[step].copyArgs));
        } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
          FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
              args->subs[step].stepBuff, (char *)data + args->totalCopySize,
              args->subs[step].stepSize, flagcxMemcpyDeviceToHost,
              resources->cpStream, args->subs[step].copyArgs));
        }
        FLAGCXCHECK(deviceAdaptor->eventRecord(resources->cpEvents[step],
                                               resources->cpStream));
      } else {
        args->subs[step].stepBuff =
            (void *)((char *)data + (flagcxNetChunkSize * args->waitCopy));
      }
      args->totalCopySize += args->subs[step].stepSize;
      args->waitCopy++;
    }

    if (args->copied < args->waitCopy) {
      TRACE(FLAGCX_PROXY,
            "ProxySend BP2 waitCopy %d, copied %d, posted %d, transmitted %d",
            args->waitCopy, args->copied, args->posted, args->transmitted);
      int step = args->copied & stepMask;
      if (!args->regBufFlag) {
        if (deviceAdaptor->eventQuery(resources->cpEvents[step]) ==
            flagcxSuccess) {
          args->copied++;
        }
      } else {
        args->copied++;
      }
    }

    if (args->posted < args->copied) {
      void *req = NULL;
      int tag = args->posted % flagcxNetChunks;
      TRACE(FLAGCX_PROXY,
            "ProxySend BP3 waitCopy %d, copied %d, posted %d, transmitted %d, "
            "tag %d",
            args->waitCopy, args->copied, args->posted, args->transmitted, tag);
      resources->netAdaptor->isend(
          resources->netSendComm, args->subs[args->posted & stepMask].stepBuff,
          args->subs[args->posted & stepMask].stepSize, tag,
          args->regBufFlag ? args->regHandle : resources->mhandles[0], NULL,
          &req);
      if (req) {
        args->subs[args->posted++ & stepMask].requests[0] = req;
      }
    }

    if (args->transmitted < args->posted) {
      TRACE(FLAGCX_PROXY,
            "ProxySend BP4 waitCopy %d, copied %d, posted %d, transmitted %d",
            args->waitCopy, args->copied, args->posted, args->transmitted);
      void *req = args->subs[args->transmitted & stepMask].requests[0];
      int done = 0, sizes;
      resources->netAdaptor->test(req, &done, &sizes);
      if (done) {
        args->transmitted++;
      }
    }
  } else {
    TRACE(FLAGCX_PROXY,
          "ProxySend BP5 waitCopy %d, copied %d, posted %d, transmitted %d",
          args->waitCopy, args->copied, args->posted, args->transmitted);
    if (args->done != 1) {
      args->semaphore->subCounter(1);
      args->done = 1;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyRecv(recvNetResources *resources, void *data,
                               size_t size, flagcxProxyArgs *args) {
  if (!args->semaphore->pollStart()) {
    return flagcxSuccess;
  }
  if (args->done) {
    return flagcxSuccess;
  }
  int res = args->chunkSteps % flagcxNetChunks;
  int niters = args->chunkSteps / flagcxNetChunks;
  if (res > 0)
    niters++;
  TRACE(FLAGCX_PROXY,
        "ProxyRecv BP0 bytes %d, buffSize %d, chunkSize %d, chunkSteps %d, "
        "totalChunks %d, res %d, niters %d",
        size, flagcxNetBufferSize, flagcxNetChunkSize, args->chunkSteps,
        flagcxNetChunks, res, niters);
  if (args->copied < niters) {
    if (args->posted < niters && (args->posted - args->copied) < 1) {
      int nreqs = (args->posted < niters - 1)
                      ? (int)flagcxNetChunks
                      : ((res > 0) ? res : (int)flagcxNetChunks);
      TRACE(FLAGCX_PROXY,
            "ProxyRecv BP1 posted %d, transmitted %d, postFlush %d, flushed "
            "%d, waitCopy %d, copied %d, nreqs %d",
            args->posted, args->transmitted, args->postFlush, args->flushed,
            args->waitCopy, args->copied, nreqs);
      void *req = NULL;
      int tags[FLAGCX_NET_MAX_STEPS] = {0};
      void *batchData[FLAGCX_NET_MAX_STEPS] = {NULL};
      void *batchMhandles[FLAGCX_NET_MAX_STEPS] = {NULL};
      size_t batchSize[FLAGCX_NET_MAX_STEPS] = {0};
      int newPostSize = 0;
      for (int i = 0; i < nreqs; ++i) {
        tags[i] = i;
        if (!args->regBufFlag) {
          args->subs[i].stepBuff =
              resources->buffers[0] + flagcxNetChunkSize * i;
        } else {
          args->subs[i].stepBuff =
              (void *)((char *)data + flagcxNetBufferSize * args->posted +
                       flagcxNetChunkSize * i);
        }
        batchData[i] = args->subs[i].stepBuff;
        args->subs[i].stepSize = flagcxNetChunkSize;
        if (args->posted == niters - 1 && i == nreqs - 1) {
          args->subs[i].stepSize =
              (size <= flagcxNetChunkSize)
                  ? size
                  : size - args->totalPostSize - newPostSize;
        }
        batchSize[i] = (size_t)args->subs[i].stepSize;
        batchMhandles[i] =
            args->regBufFlag ? args->regHandle : resources->mhandles[0];
        newPostSize += batchSize[i];
      }
      resources->netAdaptor->irecv(resources->netRecvComm, nreqs, batchData,
                                   batchSize, tags, batchMhandles, NULL, &req);
      if (req) {
        args->subs[0].requests[0] = req;
        args->totalPostSize += newPostSize;
        args->posted += 1;
      }
    }

    if (args->transmitted < args->posted) {
      TRACE(FLAGCX_PROXY,
            "ProxyRecv BP2 posted %d, transmitted %d, postFlush %d, flushed "
            "%d, waitCopy %d, copied %d",
            args->posted, args->transmitted, args->postFlush, args->flushed,
            args->waitCopy, args->copied);
      void *req = args->subs[0].requests[0];
      int done = 0;
      int sizes[FLAGCX_NET_MAX_STEPS] = {0};
      resources->netAdaptor->test(req, &done, sizes);
      if (done) {
        // for (int i = 0; i < FLAGCX_NET_MAX_STEPS; ++i) nreqs += sizes[i];
        args->transmitted += 1;
      }
    }

    if (args->postFlush < args->transmitted) {
      int nreqs = (args->postFlush < niters - 1)
                      ? (int)flagcxNetChunks
                      : ((res > 0) ? res : (int)flagcxNetChunks);
      TRACE(FLAGCX_PROXY,
            "ProxyRecv BP3 posted %d, transmitted %d, postFlush %d, flushed "
            "%d, waitCopy %d, copied %d, nreqs %d",
            args->posted, args->transmitted, args->postFlush, args->flushed,
            args->waitCopy, args->copied, nreqs);
      if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
        void *req = NULL;
        void *batchData[FLAGCX_NET_MAX_STEPS] = {NULL};
        void *batchMhandles[FLAGCX_NET_MAX_STEPS] = {NULL};
        int batchSize[FLAGCX_NET_MAX_STEPS] = {0};
        for (int i = 0; i < nreqs; ++i) {
          batchData[i] = args->subs[i].stepBuff;
          batchSize[i] = args->subs[i].stepSize;
          batchMhandles[i] =
              args->regBufFlag ? args->regHandle : resources->mhandles[0];
        }
        resources->netAdaptor->iflush(resources->netRecvComm, nreqs, batchData,
                                      batchSize, batchMhandles, &req);
        if (req) {
          args->subs[0].requests[0] = req;
          args->postFlush += 1;
        }
      } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
        args->subs[0].requests[0] = (void *)0x1;
        args->postFlush += 1;
      }
    }

    if (args->flushed < args->postFlush) {
      TRACE(FLAGCX_PROXY,
            "ProxyRecv BP4 posted %d, transmitted %d, postFlush %d, flushed "
            "%d, waitCopy %d, copied %d",
            args->posted, args->transmitted, args->postFlush, args->flushed,
            args->waitCopy, args->copied);
      void *req = args->subs[0].requests[0];
      int done = 0;
      int sizes[FLAGCX_NET_MAX_STEPS] = {0};
      if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET) &&
          req == (void *)0x1) {
        done = 1;
      } else {
        resources->netAdaptor->test(req, &done, sizes);
      }
      if (done) {
        // for (int i = 0; i < FLAGCX_NET_MAX_STEPS; ++i) nreqs += sizes[i];
        args->flushed += 1;
      }
    }

    if (args->waitCopy < args->flushed) {
      int nreqs = (args->waitCopy < niters - 1)
                      ? (int)flagcxNetChunks
                      : ((res > 0) ? res : (int)flagcxNetChunks);
      TRACE(FLAGCX_PROXY,
            "ProxyRecv BP5 posted %d, transmitted %d, postFlush %d, flushed "
            "%d, waitCopy %d, copied %d, nreqs %d",
            args->posted, args->transmitted, args->postFlush, args->flushed,
            args->waitCopy, args->copied, nreqs);
      for (int i = 0; i < nreqs; ++i) {
        if (!args->regBufFlag) {
          if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
            FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
                (char *)data + flagcxNetBufferSize * args->waitCopy +
                    flagcxNetChunkSize * i,
                args->subs[i].stepBuff, args->subs[i].stepSize,
                flagcxMemcpyDeviceToDevice, resources->cpStream,
                args->subs[i].copyArgs));
          } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
            FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
                (char *)data + flagcxNetBufferSize * args->waitCopy +
                    flagcxNetChunkSize * i,
                args->subs[i].stepBuff, args->subs[i].stepSize,
                flagcxMemcpyHostToDevice, resources->cpStream,
                args->subs[i].copyArgs));
          }
          FLAGCXCHECK(deviceAdaptor->eventRecord(resources->cpEvents[i],
                                                 resources->cpStream));
        }
      }
      // args->totalCopySize += args->subs[step].stepSize;
      args->waitCopy++;
    }

    if (args->copied < args->waitCopy) {
      int nreqs = (args->copied < niters - 1)
                      ? (int)flagcxNetChunks
                      : ((res > 0) ? res : (int)flagcxNetChunks);
      int npass = 0;
      TRACE(FLAGCX_PROXY,
            "ProxyRecv BP6 posted %d, transmitted %d, postFlush %d, flushed "
            "%d, waitCopy %d, copied %d, nreqs %d",
            args->posted, args->transmitted, args->postFlush, args->flushed,
            args->waitCopy, args->copied, nreqs);
      for (int i = 0; i < nreqs; ++i) {
        if (!args->regBufFlag) {
          if (deviceAdaptor->eventQuery(resources->cpEvents[i]) ==
              flagcxSuccess) {
            npass++;
          }
        } else {
          npass++;
        }
      }
      if (npass == nreqs) {
        args->copied++;
      }
    }
  } else {
    TRACE(FLAGCX_PROXY,
          "ProxyRecv BP7 posted %d, transmitted %d, postFlush %d, flushed %d, "
          "waitCopy %d, copied %d",
          args->posted, args->transmitted, args->postFlush, args->flushed,
          args->waitCopy, args->copied);
    if (args->done != 1) {
      args->semaphore->subCounter(1);
      args->done = 1;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSendProxyFree(sendNetResources *resources) {
  for (int s = 0; s < flagcxNetChunks; s++) {
    FLAGCXCHECK(deviceAdaptor->eventDestroy(resources->cpEvents[s]));
  }
  FLAGCXCHECK(deviceAdaptor->streamDestroy(resources->cpStream));
  resources->netAdaptor->deregMr(resources->netSendComm,
                                 resources->mhandles[0]);
  resources->netAdaptor->closeSend(resources->netSendComm);
  if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
    free(resources->buffers[0]);
  } else if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
    FLAGCXCHECK(deviceAdaptor->gdrMemFree(resources->buffers[0], NULL));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRecvProxyFree(recvNetResources *resources) {
  for (int s = 0; s < flagcxNetChunks; s++) {
    FLAGCXCHECK(deviceAdaptor->eventDestroy(resources->cpEvents[s]));
  }
  FLAGCXCHECK(deviceAdaptor->streamDestroy(resources->cpStream));
  resources->netAdaptor->deregMr(resources->netRecvComm,
                                 resources->mhandles[0]);
  resources->netAdaptor->closeRecv(resources->netRecvComm);
  resources->netAdaptor->closeListen(resources->netListenComm);
  if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
    free(resources->buffers[0]);
  } else if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
    FLAGCXCHECK(deviceAdaptor->gdrMemFree(resources->buffers[0], NULL));
  }
  return flagcxSuccess;
}

static flagcxResult_t netRegisterBuffer(flagcxHeteroComm *comm,
                                        const void *userbuff, size_t buffSize,
                                        struct flagcxConnector **peerConns,
                                        int nPeers, flagcxRegItem *regRecord,
                                        int *outRegBufFlag, void **outHandle) {
  *outRegBufFlag = 0;
  if (regRecord) {
    for (int p = 0; p < nPeers; ++p) {
      struct flagcxConnector *peerConn = peerConns[p];
      struct flagcxProxyConnector *peerProxyConn = NULL;
      bool found = false;
      if (peerConn == NULL)
        continue;
      peerProxyConn = &peerConn->proxyConn;
      for (auto it = regRecord->handles.begin(); it != regRecord->handles.end();
           it++) {
        if (it->first.proxyConn == peerProxyConn && it->first.handle) {
          found = true;
          outHandle[p] = it->first.handle;
          *outRegBufFlag = 1;
          INFO(FLAGCX_REG,
               "rank %d - NET reuse buffer %p size %ld (baseAddr %p size %ld) "
               "handle %p",
               comm->rank, userbuff, buffSize, (void *)regRecord->beginAddr,
               regRecord->endAddr - regRecord->beginAddr, it->first.handle);
          break;
        }
      }
      if (!found) {
        struct netRegInfo info = {regRecord->beginAddr,
                                  regRecord->endAddr - regRecord->beginAddr};
        void *handle = NULL;
        FLAGCXCHECK(flagcxProxyCallBlocking(
            (flagcxHeteroComm *)comm, peerProxyConn, flagcxProxyMsgRegister,
            &info, sizeof(struct netRegInfo), &handle, sizeof(void *)));
        if (handle) {
          FLAGCXCHECK(globalRegPool.addNetHandle(comm, regRecord, handle,
                                                 peerProxyConn));
          outHandle[p] = handle;
          *outRegBufFlag = 1;
          INFO(FLAGCX_REG,
               "rank %d - NET register userbuff %p (handle %p), buffSize %ld",
               comm->rank, userbuff, handle, buffSize);
        } else {
          INFO(FLAGCX_REG,
               "rank %d failed to NET register userbuff %p buffSize %ld",
               comm->rank, userbuff, buffSize);
        }
      }
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetRegisterBuffer(flagcxHeteroComm *comm,
                                       const void *userbuff, size_t buffSize,
                                       struct flagcxConnector **peerConns,
                                       int nPeers, int *outRegBufFlag,
                                       void **outHandle) {
  INFO(FLAGCX_REG, "comm = %p, userbuff = %p, buffSize = %ld, nPeers = %d",
       comm, userbuff, buffSize, nPeers);
  *outRegBufFlag = 0;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    flagcxRegItem *reg = globalRegPool.getItem(reinterpret_cast<void *>(comm),
                                               const_cast<void *>(userbuff));
    if (reg != NULL && reg->refCount > 0) {
      FLAGCXCHECK(netRegisterBuffer(comm, userbuff, buffSize, peerConns, nPeers,
                                    reg, outRegBufFlag, outHandle));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetDeregisterBuffer(void *comm,
                                         struct flagcxProxyConnector *proxyConn,
                                         void *handle) {
  INFO(FLAGCX_REG, "rank %d - deregister net buffer handle %p",
       reinterpret_cast<flagcxHeteroComm *>(comm)->rank, handle);
  FLAGCXCHECK(flagcxProxyCallBlocking(
      reinterpret_cast<flagcxHeteroComm *>(comm), proxyConn,
      flagcxProxyMsgDeregister, &handle, sizeof(void *), NULL, 0));
  return flagcxSuccess;
}