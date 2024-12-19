#include "mpi.h"
#include "flagcx.h"
#include "flagcx_hetero.h"
#include "tools.h"
#include <iostream>
#include <cstring>

#define DATATYPE flagcxFloat

int main(int argc, char *argv[]){
    parser args(argc, argv);
    size_t min_bytes = args.getMinBytes();
    size_t max_bytes = args.getMaxBytes();
    int step_factor = args.getStepFactor();
    int num_warmup_iters = args.getWarmupIters();
    int num_iters = args.getTestIters();
    int print_buffer = args.isPrintBuffer();

    int totalProcs, proc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    printf("I am %d of %d\n", proc, totalProcs);

    flagcxHandlerGroup_t handler;
    flagcxHandleInit(&handler);
    flagcxUniqueId_t& uniqueId = handler->uniqueId;
    flagcxDeviceHandle_t& devHandle = handler->devHandle;

    int nGpu;
    devHandle->getDeviceCount(&nGpu);
    devHandle->setDevice(proc % nGpu);

    if(proc == 0)
        flagcxHeteroGetUniqueId(uniqueId);

    MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    flagcxHeteroComm_t comm;
    flagcxHeteroCommInitRank(&comm, totalProcs, *uniqueId, proc);

    flagcxStream_t stream;
    devHandle->streamCreate(&stream);

    void *sendbuff, *recvbuff, *hello;
    timer tim;
    int peerSend = (proc + 1) % totalProcs;
    int peerRecv = (proc - 1 + totalProcs) % totalProcs;
    
    for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
        devHandle->deviceMalloc(&sendbuff, size, flagcxMemDevice);
        devHandle->deviceMalloc(&recvbuff, size, flagcxMemDevice);
        devHandle->deviceMalloc(&hello, size, flagcxMemHost);
        devHandle->deviceMemset(hello, 0, size, flagcxMemHost);

        for(size_t i=0;i+13<=size;i+=13){
            strcpy((char *)hello + i, std::to_string(i/(13)).c_str());
        }

        devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice, NULL);

        if(proc == 0 && print_buffer){
            printf("sendbuff = ");
            for(size_t i=0;i+13<=50;i+=13){
                printf("%c", ((char *)hello)[i]);
            }
            printf("\n");
        }

        for (int i = 0; i < num_warmup_iters; i++) {
            flagcxHeteroGroupStart();
            flagcxHeteroSend(sendbuff, size, flagcxChar, peerSend, comm, stream);
            flagcxHeteroRecv(recvbuff, size, flagcxChar, peerRecv, comm, stream);
            flagcxHeteroGroupEnd();
        }
        devHandle->streamSynchronize(stream);

        MPI_Barrier(MPI_COMM_WORLD);

        tim.reset();
        for (int i = 0; i < num_iters; i++) {
            flagcxHeteroGroupStart();
            flagcxHeteroSend(sendbuff, size, flagcxChar, peerSend, comm, stream);
            flagcxHeteroRecv(recvbuff, size, flagcxChar, peerRecv, comm, stream);
            flagcxHeteroGroupEnd();
        }
        devHandle->streamSynchronize(stream);

        double elapsed_time = tim.elapsed() / num_iters;
        double base_bw =(double)(size) / 1024 / 1024 / 1024 / elapsed_time;
        double alg_bw = base_bw;
        double factor = 1;
        double bus_bw = base_bw * factor;
        if (proc == 0) {
            printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: %lf GB/s; Bus bandwidth: %lf GB/s\n", size, elapsed_time, alg_bw, bus_bw);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        devHandle->deviceMemset(hello, 0, size, flagcxMemHost);
        devHandle->deviceMemcpy(hello, recvbuff, size, flagcxMemcpyDeviceToHost, NULL);
        if(proc == 0 && print_buffer){
            printf("recvbuff = ");
            for(size_t i=0;i+13<=50;i+=13){
                printf("%c", ((char *)hello)[i]);
            }
            printf("\n");
        }

        devHandle->deviceFree(sendbuff, flagcxMemDevice);
        devHandle->deviceFree(recvbuff, flagcxMemDevice);
        devHandle->deviceFree(hello, flagcxMemHost);
    }

    devHandle->streamDestroy(stream);
    flagcxHeteroCommDestroy(comm);
    flagcxHandleFree(handler);

    MPI_Finalize();
    return 0;
}