// Two-rank fault-injection coverage for collective one-sided registration.

#include "adaptor.h"
#include "comm.h"
#include "flagcx_hetero.h"
#include "flagcx_net_adaptor.h"
#include "global_comm.h"
#include "rma_test.hpp"

#include <cstdint>

namespace {

flagcxResult_t failRegMr(void *, void *, size_t, int, int, void **mhandle) {
  if (mhandle != nullptr)
    *mhandle = nullptr;
  return flagcxSystemError;
}

flagcxResult_t failRegMrDmaBuf(void *, void *, size_t, int, uint64_t, int, int,
                               void **mhandle) {
  if (mhandle != nullptr)
    *mhandle = nullptr;
  return flagcxSystemError;
}

void allRankResultRange(flagcxResult_t result, int *minimum, int *maximum) {
  int local = static_cast<int>(result);
  MPI_Allreduce(&local, minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local, maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
}

bool allRanksSucceeded(flagcxResult_t result) {
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  return minimum == static_cast<int>(flagcxSuccess) && minimum == maximum;
}

} // namespace

TEST_F(RmaTest, RankLocalRegistrationFailureConvergesBeforeMetadataExchange) {
  if (!networkRmaAvailable)
    GTEST_SKIP() << "Requires the explicit network RMA invocation";
  ASSERT_EQ(nranks, 2) << "Fault injection requires exactly two ranks";

  flagcxUniqueId uniqueId = {};
  flagcxResult_t setupResult = flagcxSuccess;
  if (rank == 0)
    setupResult = flagcxGetUniqueId(&uniqueId);
  ASSERT_TRUE(allRanksSucceeded(setupResult));
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

  flagcxComm_t faultComm = nullptr;
  setupResult = flagcxCommInitRank(&faultComm, nranks, &uniqueId, rank);
  ASSERT_TRUE(allRanksSucceeded(setupResult));
  ASSERT_NE(faultComm, nullptr);
  ASSERT_NE(faultComm->heteroComm, nullptr);

  constexpr size_t registrationSize = 4096;
  void *registrationBuffer = nullptr;
  setupResult = flagcxMemAlloc(&registrationBuffer, registrationSize);
  if (!allRanksSucceeded(setupResult)) {
    if (faultComm != nullptr)
      flagcxCommDestroy(faultComm);
    if (registrationBuffer != nullptr)
      flagcxMemFree(registrationBuffer);
    FAIL() << "Failed to allocate the registration fault-injection buffer";
  }

  flagcxHeteroComm *hetero = faultComm->heteroComm;
  ASSERT_NE(hetero->netAdaptor, nullptr);
  auto originalRegMr = hetero->netAdaptor->regMr;
  auto originalRegMrDmaBuf = hetero->netAdaptor->regMrDmaBuf;
  ASSERT_NE(originalRegMr, nullptr);

  // Only rank 1 fails local MR preparation. Both callbacks are replaced so
  // the test remains valid if a platform enables DMA-BUF registration.
  if (rank == 1) {
    hetero->netAdaptor->regMr = failRegMr;
    hetero->netAdaptor->regMrDmaBuf = failRegMrDmaBuf;
  }

  flagcxResult_t registerResult =
      flagcxOneSideRegister(faultComm, registrationBuffer, registrationSize);

  if (rank == 1) {
    hetero->netAdaptor->regMr = originalRegMr;
    hetero->netAdaptor->regMrDmaBuf = originalRegMrDmaBuf;
  }

  int minimumResult = 0;
  int maximumResult = 0;
  allRankResultRange(registerResult, &minimumResult, &maximumResult);
  EXPECT_EQ(minimumResult, maximumResult);
  EXPECT_EQ(minimumResult, static_cast<int>(flagcxSystemError));

  // Local status convergence must happen before metadata exchange and before
  // publishing either the handle array entry or proxy sendComm table.
  EXPECT_EQ(hetero->oneSideDataMetadataExchangeCount, 0u);
  EXPECT_EQ(hetero->oneSideHandleCount, 0);
  EXPECT_EQ(hetero->oneSideHandleCapacity, 0);
  EXPECT_EQ(hetero->oneSideHandles, nullptr);
  EXPECT_EQ(hetero->pendingOneSideCleanup, nullptr);
  ASSERT_NE(hetero->rmaProxy, nullptr);
  EXPECT_EQ(__atomic_load_n(&hetero->rmaProxy->fullSendComms, __ATOMIC_ACQUIRE),
            nullptr);

  MPI_Barrier(MPI_COMM_WORLD);
  flagcxResult_t retryResult =
      flagcxOneSideRegister(faultComm, registrationBuffer, registrationSize);
  allRankResultRange(retryResult, &minimumResult, &maximumResult);
  EXPECT_EQ(minimumResult, maximumResult);
  EXPECT_EQ(minimumResult, static_cast<int>(flagcxSuccess));
  EXPECT_EQ(hetero->oneSideDataMetadataExchangeCount, 1u);
  EXPECT_EQ(hetero->oneSideHandleCount, 1);
  EXPECT_NE(hetero->oneSideHandles, nullptr);
  ASSERT_NE(hetero->rmaProxy, nullptr);
  EXPECT_NE(__atomic_load_n(&hetero->rmaProxy->fullSendComms, __ATOMIC_ACQUIRE),
            nullptr);

  flagcxResult_t destroyResult = flagcxCommDestroy(faultComm);
  int minimumDestroy = 0;
  int maximumDestroy = 0;
  allRankResultRange(destroyResult, &minimumDestroy, &maximumDestroy);
  EXPECT_EQ(minimumDestroy, maximumDestroy);
  EXPECT_EQ(minimumDestroy, static_cast<int>(flagcxSuccess));
  if (destroyResult == flagcxSuccess) {
    EXPECT_EQ(flagcxMemFree(registrationBuffer), flagcxSuccess);
  }
}
