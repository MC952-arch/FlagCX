#include "ppu_adaptor.h"

#ifdef USE_PPU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

flagcxResult_t ppu_ncclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)ncclGetVersion(version);
}

flagcxResult_t ppu_ncclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

flagcxResult_t ppu_ncclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                              void **buff, size_t size,
                                              int isRecv) {
  return flagcxNotSupported;
}

const char *ppu_ncclAdaptorGetErrorString(flagcxResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *ppu_ncclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

flagcxResult_t
ppu_ncclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                            flagcxUniqueId_t commId, int rank,
                            struct bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                          *(ncclUniqueId *)commId, rank);
}

flagcxResult_t ppu_ncclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommFinalize(comm->base);
}

flagcxResult_t ppu_ncclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommDestroy(comm->base);
}

flagcxResult_t ppu_ncclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommAbort(comm->base);
}

flagcxResult_t ppu_ncclAdaptorCommResume(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t ppu_ncclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t ppu_ncclAdaptorCommCount(const flagcxInnerComm_t comm,
                                        int *count) {
  return (flagcxResult_t)ncclCommCount(comm->base, count);
}

flagcxResult_t ppu_ncclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                           int *device) {
  return (flagcxResult_t)ncclCommCuDevice(comm->base, device);
}

flagcxResult_t ppu_ncclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                           int *rank) {
  return (flagcxResult_t)ncclCommUserRank(comm->base, rank);
}

flagcxResult_t ppu_ncclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                                flagcxResult_t *asyncError) {
  return (flagcxResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

flagcxResult_t ppu_ncclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

flagcxResult_t ppu_ncclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

flagcxResult_t ppu_ncclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                           size_t size, void **handle) {
  return flagcxNotSupported;
}

flagcxResult_t ppu_ncclAdaptorCommDeregister(flagcxInnerComm_t comm,
                                             void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t ppu_ncclAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                                 void *buff, size_t size,
                                                 flagcxInnerWindow_t *win,
                                                 int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t ppu_ncclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                                   flagcxInnerWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t ppu_ncclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                     size_t count, flagcxDataType_t datatype,
                                     flagcxRedOp_t op, int root,
                                     flagcxInnerComm_t comm,
                                     flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t ppu_ncclAdaptorGather(const void *sendbuff, void *recvbuff,
                                     size_t count, flagcxDataType_t datatype,
                                     int root, flagcxInnerComm_t comm,
                                     flagcxStream_t stream) {
  int rank, nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommUserRank(comm->base, &rank);
  res = ncclCommCount(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = ncclRecv(static_cast<void *>(buffer + r * size), size, ncclChar, r,
                     comm->base, stream->base);
    }
  }
  res = ncclSend(sendbuff, size, ncclChar, root, comm->base, stream->base);
  res = ncclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t ppu_ncclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      int root, flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  int rank, nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommUserRank(comm->base, &rank);
  res = ncclCommCount(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  res = ncclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = ncclSend(static_cast<const void *>(buffer + r * size), size,
                     ncclChar, r, comm->base, stream->base);
    }
  }
  res = ncclRecv(recvbuff, size, ncclChar, root, comm->base, stream->base);
  res = ncclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t ppu_ncclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        int root, flagcxInnerComm_t comm,
                                        flagcxStream_t stream) {
  return (flagcxResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t ppu_ncclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op,
                                        flagcxInnerComm_t comm,
                                        flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t
ppu_ncclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t count,
                             flagcxDataType_t datatype, flagcxRedOp_t op,
                             flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduceScatter(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t ppu_ncclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxInnerComm_t comm,
                                        flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllGather(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t ppu_ncclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  int nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommCount(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = static_cast<const char *>(sendbuff);
  char *recvBuffer = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  for (int r = 0; r < nranks; r++) {
    res = ncclSend(static_cast<const void *>(sendBuffer + r * size), size,
                   ncclChar, r, comm->base, stream->base);
    res = ncclRecv(static_cast<void *>(recvBuffer + r * size), size, ncclChar,
                   r, comm->base, stream->base);
  }
  res = ncclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t
ppu_ncclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                         size_t *sdispls, void *recvbuff, size_t *recvcounts,
                         size_t *rdispls, flagcxDataType_t datatype,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  int nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommCount(comm->base, &nranks);

  size_t typeSize = getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = static_cast<const char *>(sendbuff);
  char *recvBuffer = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      res = ncclSend(
          static_cast<const void *>(sendBuffer + sdispls[r] * typeSize),
          sendcounts[r] * typeSize, ncclChar, r, comm->base, stream->base);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      res = ncclRecv(static_cast<void *>(recvBuffer + rdispls[r] * typeSize),
                     recvcounts[r] * typeSize, ncclChar, r, comm->base,
                     stream->base);
    }
  }
  res = ncclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t ppu_ncclAdaptorSend(const void *sendbuff, size_t count,
                                   flagcxDataType_t datatype, int peer,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return (flagcxResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ppu_ncclAdaptorRecv(void *recvbuff, size_t count,
                                   flagcxDataType_t datatype, int peer,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return (flagcxResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ppu_ncclAdaptorGroupStart() {
  return (flagcxResult_t)ncclGroupStart();
}

flagcxResult_t ppu_ncclAdaptorGroupEnd() {
  return (flagcxResult_t)ncclGroupEnd();
}

flagcxResult_t
ppu_ncclAdaptorDevCommReqsInit(flagcxInnerComm_t /*comm*/,
                               flagcxDevCommRequirements * /*reqs*/) {
  return flagcxNotSupported;
}

flagcxResult_t
ppu_ncclAdaptorDevCommCreate(flagcxInnerComm_t /*comm*/,
                             const flagcxDevCommRequirements * /*reqs*/,
                             flagcxInnerDevComm_t * /*devComm*/) {
  return flagcxNotSupported;
}

flagcxResult_t ppu_ncclAdaptorDevCommDestroy(flagcxInnerComm_t /*comm*/,
                                             flagcxInnerDevComm_t /*devComm*/) {
  return flagcxNotSupported;
}

struct flagcxCCLAdaptor ppu_ncclAdaptor = {
    "PPU_NCCL",
    // Basic functions
    ppu_ncclAdaptorGetVersion, ppu_ncclAdaptorGetUniqueId,
    ppu_ncclAdaptorGetErrorString, ppu_ncclAdaptorGetLastError,
    ppu_ncclAdaptorGetStagedBuffer,
    // Communicator functions
    ppu_ncclAdaptorCommInitRank, ppu_ncclAdaptorCommFinalize,
    ppu_ncclAdaptorCommDestroy, ppu_ncclAdaptorCommAbort,
    ppu_ncclAdaptorCommResume, ppu_ncclAdaptorCommSuspend,
    ppu_ncclAdaptorCommCount, ppu_ncclAdaptorCommCuDevice,
    ppu_ncclAdaptorCommUserRank, ppu_ncclAdaptorCommGetAsyncError,
    ppu_ncclAdaptorMemAlloc, ppu_ncclAdaptorMemFree,
    ppu_ncclAdaptorCommRegister, ppu_ncclAdaptorCommDeregister,
    // Symmetric functions
    ppu_ncclAdaptorCommWindowRegister, ppu_ncclAdaptorCommWindowDeregister,
    // Communication functions
    ppu_ncclAdaptorReduce, ppu_ncclAdaptorGather, ppu_ncclAdaptorScatter,
    ppu_ncclAdaptorBroadcast, ppu_ncclAdaptorAllReduce,
    ppu_ncclAdaptorReduceScatter, ppu_ncclAdaptorAllGather,
    ppu_ncclAdaptorAlltoAll, ppu_ncclAdaptorAlltoAllv, ppu_ncclAdaptorSend,
    ppu_ncclAdaptorRecv,
    // Group semantics
    ppu_ncclAdaptorGroupStart, ppu_ncclAdaptorGroupEnd,
    // Device API
    ppu_ncclAdaptorDevCommReqsInit, ppu_ncclAdaptorDevCommCreate,
    ppu_ncclAdaptorDevCommDestroy};

#endif // USE_PPU_ADAPTOR
