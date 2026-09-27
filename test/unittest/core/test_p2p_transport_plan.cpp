/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net_transport.h"
#include "p2p_transport.h"
#include <gtest/gtest.h>

TEST(P2pTransportPlan, ResolvesAllRegistrationCombinations) {
  flagcxP2pTransferPlan plan = flagcxP2pPlanPending;

  ASSERT_EQ(flagcxP2pResolveTransferPlan(0, 0, 0, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanFifo);

  ASSERT_EQ(flagcxP2pResolveTransferPlan(1, 0, 0, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanRead);

  ASSERT_EQ(flagcxP2pResolveTransferPlan(0, 1, 0, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanWrite);

  ASSERT_EQ(flagcxP2pResolveTransferPlan(1, 1, 0, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanWrite);
}

TEST(P2pTransportPlan, RejectsInvalidOffers) {
  flagcxP2pTransferPlan plan = flagcxP2pPlanPending;
  EXPECT_EQ(flagcxP2pResolveTransferPlan(2, 0, 0, 0, &plan),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxP2pResolveTransferPlan(0, -1, 0, 0, &plan),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxP2pResolveTransferPlan(0, 0, 0, 0, nullptr),
            flagcxInvalidArgument);
}

TEST(P2pTransportPlan, IpcFailureForcesACommonFifoFallback) {
  flagcxP2pTransferPlan plan = flagcxP2pPlanPending;
  ASSERT_EQ(flagcxP2pResolveTransferPlan(1, 1, 1, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanFifo);
  ASSERT_EQ(flagcxP2pResolveTransferPlan(1, 1, 0, 1, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanFifo);
  EXPECT_EQ(flagcxP2pResolveTransferPlan(1, 1, 2, 0, &plan),
            flagcxInvalidArgument);
}

TEST(TransportCompletion, DistinguishesPendingFromPermanentFailure) {
  int completed = -1;
  EXPECT_EQ(flagcxTransportClassifyCompletion(flagcxSuccess, &completed),
            flagcxSuccess);
  EXPECT_EQ(completed, 1);

  completed = -1;
  EXPECT_EQ(flagcxTransportClassifyCompletion(flagcxInProgress, &completed),
            flagcxSuccess);
  EXPECT_EQ(completed, 0);

  completed = -1;
  EXPECT_EQ(flagcxTransportClassifyCompletion(flagcxSystemError, &completed),
            flagcxSystemError);
  EXPECT_EQ(completed, 0);
  EXPECT_EQ(flagcxTransportClassifyCompletion(flagcxSuccess, nullptr),
            flagcxInvalidArgument);
}

TEST(TransportNeutralFoundation, RequestAndCreditShareCanonicalState) {
  flagcxTransportCredit credit = {};
  ASSERT_EQ(flagcxTransportCreditInit(&credit, 2), flagcxSuccess);
  ASSERT_EQ(flagcxTransportCreditAcquire(&credit, 2), flagcxSuccess);
  EXPECT_EQ(flagcxTransportCreditAcquire(&credit, 1), flagcxInProgress);
  ASSERT_EQ(flagcxTransportCreditRelease(&credit, 2), flagcxSuccess);

  flagcxTransportRequest request = {};
  flagcxTransportRequestInit(&request);
  ASSERT_EQ(flagcxTransportRequestAcquire(&request), flagcxSuccess);
  ASSERT_EQ(flagcxTransportRequestAddPending(&request, 2), flagcxSuccess);
  ASSERT_EQ(flagcxTransportRequestComplete(&request, 1, flagcxSuccess),
            flagcxSuccess);
  ASSERT_EQ(flagcxTransportRequestComplete(&request, 1, flagcxRemoteError),
            flagcxSuccess);
  int done = 0;
  EXPECT_EQ(flagcxTransportRequestTest(&request, &done), flagcxRemoteError);
  EXPECT_EQ(done, 1);
}
