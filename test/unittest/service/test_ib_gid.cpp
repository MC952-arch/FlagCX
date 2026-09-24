/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "ib_gid.h"

#include <gtest/gtest.h>

namespace {

flagcxIbAutoGidCandidate
candidate(int index, uint32_t type, bool hasNetworkDevice = false,
          bool isIpv4Mapped = false, bool isLinkLocalIpv6 = false,
          bool isOverlayNetwork = false, bool isPrivateIpv4 = false,
          bool isLinkLocalIpv4 = false) {
  flagcxIbAutoGidCandidate value = {};
  value.gidIndex = index;
  value.gidType = type;
  value.hasNetworkDevice = hasNetworkDevice;
  value.isIpv4Mapped = isIpv4Mapped;
  value.isLinkLocalIpv4 = isLinkLocalIpv4;
  value.isLinkLocalIpv6 = isLinkLocalIpv6;
  value.isOverlayNetwork = isOverlayNetwork;
  value.isPrivateIpv4 = isPrivateIpv4;
  value.querySucceeded = true;
  return value;
}

} // namespace

TEST(IbAutoGid, UsesAutoSelectionOnlyWithoutExplicitLegacySelectors) {
  EXPECT_TRUE(flagcxIbShouldUseAutoGidSelection(false, false, false));
  EXPECT_FALSE(flagcxIbShouldUseAutoGidSelection(true, false, false));
  EXPECT_FALSE(flagcxIbShouldUseAutoGidSelection(false, true, false));
  EXPECT_FALSE(flagcxIbShouldUseAutoGidSelection(false, false, true));
}

TEST(IbAutoGid, ClassifiesIpv4AddressScopes) {
  EXPECT_EQ(flagcxIbClassifyIpv4Address(0x0a000001U), flagcxIbIpv4Private);
  EXPECT_EQ(flagcxIbClassifyIpv4Address(0xac100001U), flagcxIbIpv4Private);
  EXPECT_EQ(flagcxIbClassifyIpv4Address(0xc0a80101U), flagcxIbIpv4Private);
  EXPECT_EQ(flagcxIbClassifyIpv4Address(0x64400001U), flagcxIbIpv4Private);
  EXPECT_EQ(flagcxIbClassifyIpv4Address(0xa9fe0001U), flagcxIbIpv4LinkLocal);
  EXPECT_EQ(flagcxIbClassifyIpv4Address(0x08080808U), flagcxIbIpv4Routable);
}

TEST(IbAutoGid, PrefersRoutableNetworkCandidate) {
  flagcxIbAutoGidCandidate candidates[] = {
      candidate(0, FLAGCX_IB_GID_TYPE_ROCE_V2, false),
      candidate(1, FLAGCX_IB_GID_TYPE_ROCE_V2, true, true, false, false, true),
      candidate(2, FLAGCX_IB_GID_TYPE_ROCE_V2, true),
  };
  flagcxIbAutoGidSelection selection = {};

  ASSERT_TRUE(flagcxIbSelectBestAutoGidCandidate(
      candidates, sizeof(candidates) / sizeof(candidates[0]), &selection));
  EXPECT_EQ(selection.gidIndex, 2);
  EXPECT_EQ(selection.candidateClass, flagcxIbGidNetworkRoutable);
}

TEST(IbAutoGid, PrefersPrivateFabricAddressOverDegradedCandidate) {
  flagcxIbAutoGidCandidate candidates[] = {
      candidate(0, FLAGCX_IB_GID_TYPE_ROCE_V2, true, false, true),
      candidate(1, FLAGCX_IB_GID_TYPE_ROCE_V2, true, true, false, false, true),
      candidate(2, FLAGCX_IB_GID_TYPE_ROCE_V2, true, true, false, true),
  };
  flagcxIbAutoGidSelection selection = {};

  ASSERT_TRUE(flagcxIbSelectBestAutoGidCandidate(
      candidates, sizeof(candidates) / sizeof(candidates[0]), &selection));
  EXPECT_EQ(selection.gidIndex, 1);
  EXPECT_EQ(selection.candidateClass, flagcxIbGidNetworkPrivateV4);
}

TEST(IbAutoGid, DegradesIpv4LinkLocalCandidate) {
  flagcxIbAutoGidCandidate candidates[] = {
      candidate(0, FLAGCX_IB_GID_TYPE_ROCE_V2, true, true, false, false, false,
                true),
      candidate(1, FLAGCX_IB_GID_TYPE_ROCE_V2, true, true, false, false, true),
  };
  flagcxIbAutoGidSelection selection = {};

  ASSERT_TRUE(flagcxIbSelectBestAutoGidCandidate(
      candidates, sizeof(candidates) / sizeof(candidates[0]), &selection));
  EXPECT_EQ(selection.gidIndex, 1);
  EXPECT_EQ(selection.candidateClass, flagcxIbGidNetworkPrivateV4);
}

TEST(IbAutoGid, DoesNotDemoteInfiniBandGidForIpv6Shape) {
  flagcxIbAutoGidCandidate candidates[] = {
      candidate(3, FLAGCX_IB_GID_TYPE_IB, true, false, true),
      candidate(2, FLAGCX_IB_GID_TYPE_ROCE_V2, true, false, true),
  };
  flagcxIbAutoGidSelection selection = {};

  ASSERT_TRUE(flagcxIbSelectBestAutoGidCandidate(
      candidates, sizeof(candidates) / sizeof(candidates[0]), &selection));
  EXPECT_EQ(selection.gidIndex, 3);
  EXPECT_EQ(selection.candidateClass, flagcxIbGidNetworkRoutable);
}

TEST(IbAutoGid, UsesLowestIndexWithinSameClass) {
  flagcxIbAutoGidCandidate candidates[] = {
      candidate(7, FLAGCX_IB_GID_TYPE_ROCE_V2, true),
      candidate(2, FLAGCX_IB_GID_TYPE_ROCE_V2, true),
  };
  flagcxIbAutoGidSelection selection = {};

  ASSERT_TRUE(flagcxIbSelectBestAutoGidCandidate(
      candidates, sizeof(candidates) / sizeof(candidates[0]), &selection));
  EXPECT_EQ(selection.gidIndex, 2);
}

TEST(IbAutoGid, FallsBackToFirstNonzeroUnsupportedType) {
  flagcxIbAutoGidCandidate candidates[] = {
      candidate(4, 1, true),
      candidate(6, 1, true),
  };
  flagcxIbAutoGidSelection selection = {};

  ASSERT_TRUE(flagcxIbSelectBestAutoGidCandidate(
      candidates, sizeof(candidates) / sizeof(candidates[0]), &selection));
  EXPECT_EQ(selection.gidIndex, 4);
  EXPECT_EQ(selection.candidateClass, flagcxIbGidFallbackNonzero);
}

TEST(IbAutoGid, RejectsFailedAndNullQueries) {
  flagcxIbAutoGidCandidate candidates[] = {
      candidate(0, FLAGCX_IB_GID_TYPE_IB, true),
      candidate(1, FLAGCX_IB_GID_TYPE_ROCE_V2, true),
  };
  candidates[0].querySucceeded = false;
  candidates[1].isNullGid = true;
  flagcxIbAutoGidSelection selection = {};

  EXPECT_FALSE(flagcxIbSelectBestAutoGidCandidate(
      candidates, sizeof(candidates) / sizeof(candidates[0]), &selection));
}
