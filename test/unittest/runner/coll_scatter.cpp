// Scatter correctness test (migrated from test/unittest/main.cpp)
#include "global_comm.h"
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <iostream>
#include <vector>

TEST_F(FlagCXCollTest, Scatter) {
  int root = 0;
  if (comm->commType == flagcxCommunicatorHybrid) {
    // Prefer inter-rank 0 outside cluster 0. On a general multi-NIC topology
    // this exercises the root scratch seed used when the inter-rank self-send
    // is intentionally omitted.
    int candidate =
        comm->clusterIds[rank] > 0 && comm->homoInterMyRank == 0 ? rank : -1;
    MPI_Allreduce(&candidate, &root, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (root < 0)
      root = 0;
  }

  if (rank == root) {
    for (size_t i = 0; i < count; i++) {
      ((float *)hostsendbuff)[i] = static_cast<float>(i);
    }

    devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                            flagcxMemcpyHostToDevice, stream);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t localResult = flagcxScatter(sendbuff, recvbuff, count / nranks,
                                             flagcxFloat, root, comm, stream);
  int localResultCode = static_cast<int>(localResult);
  int globalResultCode = static_cast<int>(flagcxSuccess);
  MPI_Allreduce(&localResultCode, &globalResultCode, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  ASSERT_EQ(globalResultCode, static_cast<int>(flagcxSuccess));

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size / nranks,
                          flagcxMemcpyDeviceToHost, stream);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  // Each rank receives chunk[rank] of the selected root's send buffer.
  size_t chunkCount = count / nranks;
  size_t chunkStart = rank * chunkCount;
  std::vector<float> expected(chunkCount);
  for (size_t i = 0; i < chunkCount; i++) {
    expected[i] = static_cast<float>(chunkStart + i);
  }

  EXPECT_TRUE(verifyBuffer(static_cast<float *>(hostrecvbuff), expected.data(),
                           chunkCount));
}
