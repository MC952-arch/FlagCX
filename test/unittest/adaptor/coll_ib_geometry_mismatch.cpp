/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>
#include <mpi.h>

#include "comm.h"
#include "flagcx.h"
#include "global_comm.h"

#include <cstdlib>
#include <cstring>

namespace {

void recordFirstPermanentError(flagcxResult_t candidate,
                               flagcxResult_t *first) {
  if (candidate != flagcxSuccess && candidate != flagcxInProgress &&
      *first == flagcxSuccess)
    *first = candidate;
}

TEST(IbConnectionGeometryMpiTest,
     MismatchedPeersFailConsistentlyWithoutPublishingConnectors) {
  const char *enabled = std::getenv("FLAGCX_CI_EXPECT_IB_GEOMETRY_MISMATCH");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0)
    GTEST_SKIP() << "Runs only in the CUDA IB geometry mismatch job";

  int rank = -1;
  int nranks = 0;
  ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
  ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &nranks), MPI_SUCCESS);
  ASSERT_EQ(nranks, 2);

  flagcxUniqueId uniqueId{};
  int idResult = static_cast<int>(flagcxSuccess);
  if (rank == 0)
    idResult = static_cast<int>(flagcxGetUniqueId(&uniqueId));
  ASSERT_EQ(MPI_Bcast(&idResult, 1, MPI_INT, 0, MPI_COMM_WORLD), MPI_SUCCESS);
  ASSERT_EQ(idResult, static_cast<int>(flagcxSuccess));
  ASSERT_EQ(MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD),
            MPI_SUCCESS);

  flagcxComm_t comm = nullptr;
  flagcxResult_t initResult =
      flagcxCommInitRank(&comm, nranks, &uniqueId, rank);
  int localInitOk = initResult == flagcxSuccess && comm != nullptr;
  int allInitOk = 0;
  ASSERT_EQ(MPI_Allreduce(&localInitOk, &allInitOk, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(allInitOk, 1) << "communicator initialization failed before the "
                             "connection geometry check";

  int localUsesIb = comm->heteroComm != nullptr &&
                    comm->heteroComm->netAdaptor != nullptr &&
                    std::strcmp(comm->heteroComm->netAdaptor->name, "IB") == 0;
  int allUseIb = 0;
  ASSERT_EQ(MPI_Allreduce(&localUsesIb, &allUseIb, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(allUseIb, 1);

  const int peer = 1 - rank;
  int payload = rank;
  flagcxResult_t firstError = flagcxSuccess;
  flagcxResult_t result = flagcxGroupStart(comm);
  recordFirstPermanentError(result, &firstError);
  if (result == flagcxSuccess || result == flagcxInProgress) {
    result = flagcxSend(&payload, 1, flagcxInt, peer, comm, nullptr);
    recordFirstPermanentError(result, &firstError);
    result = flagcxRecv(&payload, 1, flagcxInt, peer, comm, nullptr);
    recordFirstPermanentError(result, &firstError);
  }
  result = flagcxGroupEnd(comm);
  recordFirstPermanentError(result, &firstError);

  int localError = static_cast<int>(firstError);
  int minError = 0;
  int maxError = 0;
  ASSERT_EQ(MPI_Allreduce(&localError, &minError, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(&localError, &maxError, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  EXPECT_NE(minError, static_cast<int>(flagcxSuccess));
  EXPECT_EQ(minError, maxError)
      << "all ranks must converge on the same geometry mismatch error";

  const flagcxConnector *sendConnector =
      &comm->heteroComm->channels[0].peers[peer]->send[0];
  const flagcxConnector *recvConnector =
      &comm->heteroComm->channels[0].peers[peer]->recv[0];
  int localPublished = sendConnector->connected || recvConnector->connected;
  int anyPublished = 0;
  ASSERT_EQ(MPI_Allreduce(&localPublished, &anyPublished, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  EXPECT_EQ(anyPublished, 0)
      << "a failed geometry handshake must not publish a connector";

  // The enclosing MPI timeout makes a stalled cleanup a hard CI failure.
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
}

} // namespace
