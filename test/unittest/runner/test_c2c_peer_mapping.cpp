#include <gtest/gtest.h>

#include "c2c_algo.h"

TEST(C2cPeerMapping, TwoClustersKeepRemoteResultOffRepresentative) {
  EXPECT_EQ(flagcxC2cGetPeerHomoRank(0, 1, 0, 2), 1);
  EXPECT_EQ(flagcxC2cGetPeerHomoRank(1, 0, 1, 2), 0);
}

TEST(C2cPeerMapping, ThreeClustersUseDistinctReceiveSlots) {
  EXPECT_EQ(flagcxC2cGetPeerHomoRank(0, 1, 0, 3), 2);
  EXPECT_EQ(flagcxC2cGetPeerHomoRank(0, 2, 0, 3), 1);

  EXPECT_EQ(flagcxC2cGetPeerHomoRank(1, 0, 1, 3), 0);
  EXPECT_EQ(flagcxC2cGetPeerHomoRank(1, 2, 1, 3), 2);
}

TEST(C2cPeerMapping, RejectsLayoutsWithoutEnoughReceiveSlots) {
  EXPECT_EQ(flagcxC2cGetPeerHomoRank(0, 2, 0, 2), -1);
  EXPECT_EQ(flagcxC2cGetPeerHomoRank(0, 0, 0, 2), -1);
}
