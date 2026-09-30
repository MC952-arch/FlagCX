/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "coll_proxy_transport.h"

TEST(CollProxyTransportTest, OrderingKeyIsStableForWireDirection) {
  const uint64_t key = flagcxCollProxyOrderingKey(3, 7, 11);
  EXPECT_EQ(key, flagcxCollProxyOrderingKey(3, 7, 11));
  EXPECT_NE(key, flagcxCollProxyOrderingKey(4, 7, 11));
  EXPECT_NE(key, flagcxCollProxyOrderingKey(3, 11, 7));
  EXPECT_NE(key, 0u);
}

TEST(CollProxyTransportTest, EdgeChannelIsStableAndNeverUsesPublicDomain) {
  EXPECT_EQ(flagcxCollProxyChannelForEdge(2, 7, 1),
            flagcxCollProxyChannelForEdge(2, 7, 1));
  EXPECT_GT(flagcxCollProxyChannelForEdge(2, 7, 1), 0);
  EXPECT_LT(flagcxCollProxyChannelForEdge(2, 7, 1),
            FLAGCX_COLL_PROXY_MAX_CHANNELS);
}

TEST(CollProxyTransportTest, OutOfOrderCompletionRetiresContiguousPrefix) {
  flagcxCollProxyTransport transport = {};
  ASSERT_EQ(flagcxCollProxyTransportInit(&transport, 4, 9,
                                         flagcxCollProxyOrderingKey(2, 0, 1),
                                         FLAGCX_NET_SUBMIT_INDEPENDENT),
            flagcxSuccess);

  flagcxNetSubmitContext *first = nullptr;
  flagcxNetSubmitContext *second = nullptr;
  ASSERT_EQ(flagcxCollProxyTrackNext(&transport, &first), flagcxSuccess);
  ASSERT_EQ(flagcxCollProxyTrackNext(&transport, &second), flagcxSuccess);
  EXPECT_EQ(first->sequence, 0u);
  EXPECT_EQ(second->sequence, 1u);
  EXPECT_NE(first->flags & FLAGCX_NET_SUBMIT_INDEPENDENT, 0u);

  uint32_t advanced = UINT32_MAX;
  ASSERT_EQ(
      flagcxCollProxyComplete(&transport, second, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 0u);

  ASSERT_EQ(
      flagcxCollProxyComplete(&transport, first, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 2u);
}

TEST(CollProxyTransportTest, CreditBackpressureAndCancelledPostAreRetryable) {
  flagcxCollProxyTransport transport = {};
  ASSERT_EQ(flagcxCollProxyTransportInit(&transport, 2, 1, 0, 0),
            flagcxSuccess);

  flagcxNetSubmitContext *first = nullptr;
  flagcxNetSubmitContext *second = nullptr;
  flagcxNetSubmitContext *blocked = nullptr;
  ASSERT_EQ(flagcxCollProxyTrackNext(&transport, &first), flagcxSuccess);
  ASSERT_EQ(flagcxCollProxyTrackNext(&transport, &second), flagcxSuccess);
  EXPECT_EQ(flagcxCollProxyTrackNext(&transport, &blocked), flagcxInProgress);

  ASSERT_EQ(flagcxCollProxyCancel(&transport, second), flagcxSuccess);
  ASSERT_EQ(flagcxCollProxyTrackNext(&transport, &second), flagcxSuccess);
  EXPECT_EQ(second->sequence, 1u);

  uint32_t advanced = 0;
  ASSERT_EQ(
      flagcxCollProxyComplete(&transport, first, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  ASSERT_EQ(
      flagcxCollProxyComplete(&transport, second, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
}

TEST(CollProxyTransportTest, FirstErrorIsRetainedWhilePrefixDrains) {
  flagcxCollProxyTransport transport = {};
  ASSERT_EQ(flagcxCollProxyTransportInit(&transport, 2, 4, 17,
                                         FLAGCX_NET_SUBMIT_INDEPENDENT),
            flagcxSuccess);
  flagcxNetSubmitContext *first = nullptr;
  flagcxNetSubmitContext *second = nullptr;
  ASSERT_EQ(flagcxCollProxyTrackNext(&transport, &first), flagcxSuccess);
  ASSERT_EQ(flagcxCollProxyTrackNext(&transport, &second), flagcxSuccess);

  uint32_t advanced = 0;
  ASSERT_EQ(
      flagcxCollProxyComplete(&transport, second, flagcxRemoteError, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 0u);
  ASSERT_EQ(
      flagcxCollProxyComplete(&transport, first, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 2u);

  uint64_t next = 0;
  uint32_t inFlight = 0;
  flagcxResult_t firstError = flagcxSuccess;
  ASSERT_EQ(flagcxNetCompletionScoreboardQuery(&transport.scoreboard, &next,
                                               &inFlight, &firstError),
            flagcxSuccess);
  EXPECT_EQ(next, 2u);
  EXPECT_EQ(inFlight, 0u);
  EXPECT_EQ(firstError, flagcxRemoteError);
}
