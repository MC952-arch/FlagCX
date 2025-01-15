#include "gloo_adaptor.h"

#ifdef USE_GLOO_ADAPTOR

//TODO: unsupported
flagcxResult_t glooAdaptorGetVersion(int *version) {
    return flagcxNotSupported;
}

//TODO: unsupported
flagcxResult_t glooAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
    return flagcxNotSupported;
}

//TODO: unsupported
const char* glooAdaptorGetErrorString(flagcxResult_t result) {
    return "Not Implemented";
}

//TODO: unsupported
const char* glooAdaptorGetLastError(flagcxHomoComm_t comm) {
    return "Not Implemented";
}

flagcxResult_t glooAdaptorCommInitRank(flagcxHomoComm_t *comm, int nranks, flagcxUniqueId_t /*commId*/, int rank, bootstrapState *bootstrap) {
    if (*comm == NULL) {
        FLAGCXCHECK(flagcxCalloc(comm, 1));
    }
    // Create gloo context
    (*comm)->base = std::make_shared<flagcxGlooContext>(rank, nranks, bootstrap);
    // Create gloo transport device
    std::shared_ptr<::gloo::transport::Device> dev;
    try {
        // Firstly, try ibverbs
        ::gloo::transport::ibverbs::attr attr;
        dev = ::gloo::transport::ibverbs::CreateDevice(attr);
    } catch (const std::exception& e) {
        std::cout << "Caught an exception during the creation of ibverbs transport device: " << e.what() << ". Try tcp transport device alternatively." << std::endl;
        // Alternatively, try tcp
        char line[1024];
        FLAGCXCHECK(getHostName(line, 1024, '.'));
        std::string hostname(line);
        ::gloo::transport::tcp::attr attr;
        attr.hostname = hostname;
        dev = ::gloo::transport::tcp::CreateDevice(attr);
    }
    (*comm)->base->connectFullMesh(dev);
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommFinalize(flagcxHomoComm_t comm) {
    comm->base.reset();
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommDestroy(flagcxHomoComm_t comm) {
    comm->base.reset();
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommAbort(flagcxHomoComm_t comm) {
    comm->base.reset();
    return flagcxSuccess;
}

//TODO: unsupported
flagcxResult_t glooAdaptorCommResume(flagcxHomoComm_t comm) {
    return flagcxNotSupported;
}

//TODO: unsupported
flagcxResult_t glooAdaptorCommSuspend(flagcxHomoComm_t comm) {
    return flagcxNotSupported;
}

flagcxResult_t glooAdaptorCommCount(const flagcxHomoComm_t comm, int* count) {
    *count = comm->base->size;
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommCuDevice(const flagcxHomoComm_t comm, int* device) {
    device = NULL;
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommUserRank(const flagcxHomoComm_t comm, int* rank) {
    *rank = comm->base->rank;
    return flagcxSuccess;
}

//TODO: unsupported
flagcxResult_t glooAdaptorCommGetAsyncError(flagcxHomoComm_t comm, flagcxResult_t asyncError) {
    return flagcxNotSupported;
}

//TODO: unsupported
flagcxResult_t glooAdaptorReduce(const void* sendbuff, void* recvbuff, size_t count,
                                 flagcxDataType_t datatype, flagcxRedOp_t op, int root,
                                 flagcxHomoComm_t comm, flagcxStream_t /*stream*/) {
    return flagcxNotSupported;
}

//TODO: unsupported
flagcxResult_t glooAdaptorGather(const void* sendbuff, void* recvbuff, size_t count,
                                 flagcxDataType_t datatype, int root, flagcxHomoComm_t comm,
                                 flagcxStream_t /*stream*/) {
    return flagcxNotSupported;
}

//TODO: unsupported
flagcxResult_t glooAdaptorScatter(const void* sendbuff, void* recvbuff, size_t count,
                                  flagcxDataType_t datatype, int root, flagcxHomoComm_t comm,
                                  flagcxStream_t /*stream*/) {
    return flagcxNotSupported;
}

//TODO: unsupported
flagcxResult_t glooAdaptorBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                                    flagcxDataType_t datatype, int root, flagcxHomoComm_t comm,
                                    flagcxStream_t /*stream*/) {
    return flagcxNotSupported;
}

flagcxResult_t glooAdaptorAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                                    flagcxDataType_t datatype, flagcxRedOp_t op,
                                    flagcxHomoComm_t comm, flagcxStream_t /*stream*/) {
    ::gloo::AllreduceOptions opts(comm->base);
    opts.setReduceFunction(getFunction<::gloo::AllreduceOptions::Func>(datatype, op));
    GENERATE_ALL_TYPES(datatype, setInput, opts, const_cast<void*>(sendbuff), count);
    GENERATE_ALL_TYPES(datatype, setOutput, opts, recvbuff, count);
    ::gloo::allreduce(opts);
    return flagcxSuccess;
}

//TODO: unsupported
flagcxResult_t glooAdaptorReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                        flagcxDataType_t datatype, flagcxRedOp_t op,
                                        flagcxHomoComm_t comm, flagcxStream_t /*stream*/) {
    return flagcxNotSupported;
}

flagcxResult_t glooAdaptorAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                                    flagcxDataType_t datatype, flagcxHomoComm_t comm,
                                    flagcxStream_t /*stream*/) {
    ::gloo::AllgatherOptions opts(comm->base);
    GENERATE_ALL_TYPES(datatype, setInput, opts, const_cast<void*>(sendbuff), sendcount);
    GENERATE_ALL_TYPES(datatype, setOutput, opts, recvbuff, comm->base->size * sendcount);
    ::gloo::allgather(opts);
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
                                   flagcxDataType_t datatype, flagcxHomoComm_t comm,
                                   flagcxStream_t /*stream*/) {
    ::gloo::AlltoallOptions opts(comm->base);
    GENERATE_ALL_TYPES(datatype, setInput, opts, const_cast<void*>(sendbuff), comm->base->size * count);
    GENERATE_ALL_TYPES(datatype, setOutput, opts, recvbuff, comm->base->size * count);
    ::gloo::alltoall(opts);
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorSend(const void* sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxHomoComm_t comm, flagcxStream_t /*stream*/) {
    size_t size = count * getFlagcxDataTypeSize(datatype);
    inputBuffers.push(comm->base->createUnboundBuffer(const_cast<void*>(sendbuff), size));
    inputBuffers.back()->send(peer, comm->base->rank);
    if (!groupStarted) {
        inputBuffers.back()->waitSend(flagcxGlooDefaultTimeout);
    }
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorRecv(void* recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxHomoComm_t comm, flagcxStream_t /*stream*/) {
    size_t size = count * getFlagcxDataTypeSize(datatype);
    auto buf = comm->base->createUnboundBuffer(const_cast<void*>(recvbuff), size);
    buf->recv(peer, peer);
    buf->waitRecv(flagcxGlooDefaultTimeout);
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorGroupStart() {
    groupStarted = true;
    return flagcxSuccess;
}

flagcxResult_t glooAdaptorGroupEnd() {
    if (groupStarted) {
        while (!inputBuffers.empty()) {
            inputBuffers.front()->waitSend(flagcxGlooDefaultTimeout);
            inputBuffers.pop();
        }
        groupStarted = false;
    }
    return flagcxSuccess;
}

struct flagcxCCLAdaptor glooAdaptor = {
  "GLOO",
  // Basic functions
  glooAdaptorGetVersion,
  glooAdaptorGetUniqueId,
  glooAdaptorGetErrorString,
  glooAdaptorGetLastError,
  // Communicator functions
  glooAdaptorCommInitRank,
  glooAdaptorCommFinalize,
  glooAdaptorCommDestroy,
  glooAdaptorCommAbort,
  glooAdaptorCommResume,
  glooAdaptorCommSuspend,
  glooAdaptorCommCount,
  glooAdaptorCommCuDevice,
  glooAdaptorCommUserRank,
  glooAdaptorCommGetAsyncError,
  // Communication functions
  glooAdaptorReduce,
  glooAdaptorGather,
  glooAdaptorScatter,
  glooAdaptorBroadcast,
  glooAdaptorAllReduce,
  glooAdaptorReduceScatter,
  glooAdaptorAllGather,
  glooAdaptorAlltoAll,
  glooAdaptorSend,
  glooAdaptorRecv,
  // Group semantics
  glooAdaptorGroupStart,
  glooAdaptorGroupEnd
};

#endif // USE_GLOO_ADAPTOR