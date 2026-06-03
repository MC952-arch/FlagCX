// Unit tests for bootstrap accessor APIs.
// Source: flagcx/service/include/bootstrap.h, flagcx/service/bootstrap.cc
// Links against libflagcx.

#include <cstring>
#include <gtest/gtest.h>

#include "bootstrap.h"

// =============================================================================
// bootstrapGetRank / bootstrapGetNranks
// =============================================================================

TEST(BootstrapAccessors, GetRankNullState) {
  EXPECT_EQ(bootstrapGetRank(nullptr), -1);
  EXPECT_EQ(bootstrapGetNranks(nullptr), -1);
}

TEST(BootstrapAccessors, GetRankP2pModeReturnsNegOne) {
  // P2P mode has no rank/nranks concept — accessors should return -1
  struct bootstrapP2pState p2p;
  memset(&p2p, 0, sizeof(p2p));

  struct bootstrapState state;
  state.mode = FLAGCX_BOOTSTRAP_P2P;
  state.p2p = &p2p;

  EXPECT_EQ(bootstrapGetRank(&state), -1);
  EXPECT_EQ(bootstrapGetNranks(&state), -1);
}

TEST(BootstrapAccessors, GetRankCollMode) {
  struct bootstrapCollState coll;
  memset(&coll, 0, sizeof(coll));
  coll.rank = 3;
  coll.nranks = 8;

  struct bootstrapState state;
  state.mode = FLAGCX_BOOTSTRAP_COLL;
  state.coll = &coll;

  EXPECT_EQ(bootstrapGetRank(&state), 3);
  EXPECT_EQ(bootstrapGetNranks(&state), 8);
}

TEST(BootstrapAccessors, GetRankCollNullInner) {
  struct bootstrapState state;
  state.mode = FLAGCX_BOOTSTRAP_COLL;
  state.coll = nullptr;

  EXPECT_EQ(bootstrapGetRank(&state), -1);
  EXPECT_EQ(bootstrapGetNranks(&state), -1);
}

// =============================================================================
// bootstrapGetNetProperties / bootstrapGetNetIfName / bootstrapGetNetIfAddr
// =============================================================================

TEST(BootstrapAccessors, GetNetPropertiesNotNull) {
  // The global properties struct always exists (static storage)
  flagcxNetProperties_t *props = bootstrapGetNetProperties();
  ASSERT_NE(props, nullptr);
}

TEST(BootstrapAccessors, GetNetIfNameNotNull) {
  const char *name = bootstrapGetNetIfName();
  ASSERT_NE(name, nullptr);
  // Before bootstrapNetInit is called, name may be empty string
  // but the pointer itself must be valid
}

TEST(BootstrapAccessors, GetNetIfAddrNotNull) {
  union flagcxSocketAddress *addr = bootstrapGetNetIfAddr();
  ASSERT_NE(addr, nullptr);
}

TEST(BootstrapAccessors, NetInitPopulatesIfName) {
  // After bootstrapNetInit succeeds, ifName should be non-empty
  flagcxResult_t res = bootstrapNetInit();
  if (res == flagcxSuccess) {
    const char *name = bootstrapGetNetIfName();
    EXPECT_GT(strlen(name), 0u);

    union flagcxSocketAddress *addr = bootstrapGetNetIfAddr();
    // Should have a valid address family
    EXPECT_TRUE(addr->sa.sa_family == AF_INET ||
                addr->sa.sa_family == AF_INET6);
  } else {
    // If no network interface is available, skip gracefully
    GTEST_SKIP() << "bootstrapNetInit failed (no usable interface)";
  }
}
