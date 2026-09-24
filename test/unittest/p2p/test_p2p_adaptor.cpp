// Unit tests for the IB P2P net adaptor.
// Tests that don't require IB hardware always run. The device requirement test
// fails when the adaptor exposes no usable IB device; individual data-path
// tests may still skip to avoid repeating the same environment failure.
// Links against libflagcx.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <sstream>
#include <thread>
#include <vector>

#include "../adaptor/net_test_utils.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "ibv_compat.h"

// The P2P adaptor struct is non-static in ibrc_p2p_adaptor.cc
extern struct flagcxNetAdaptor flagcxNetIbP2p;
extern flagcxResult_t flagcxNetIbP2pAbortListen(void *listenComm);

// ---------------------------------------------------------------------------
// Fixture: initializes the adaptor once, caches device count
// ---------------------------------------------------------------------------
class P2pAdaptorTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initResult = flagcxNetIbP2p.init();
    if (initResult == flagcxSuccess) {
      flagcxNetIbP2p.devices(&nDevs);
      if (nDevs > 0) {
        selectionResult =
            flagcx_test::getLocalNetDevice(&flagcxNetIbP2p, nDevs, &netDev);
      }
    }
  }

  static bool hasIbDevices() {
    return initResult == flagcxSuccess && nDevs > 0;
  }

  static flagcxResult_t initResult;
  static flagcxResult_t selectionResult;
  static int nDevs;
  static int netDev;
};

flagcxResult_t P2pAdaptorTest::initResult = flagcxInternalError;
flagcxResult_t P2pAdaptorTest::selectionResult = flagcxInternalError;
int P2pAdaptorTest::nDevs = 0;
int P2pAdaptorTest::netDev = -1;

// ---------------------------------------------------------------------------
// 1. Adaptor struct completeness — always runs, no hardware needed
// ---------------------------------------------------------------------------
TEST(P2pAdaptorStruct, AllFunctionPointersSet) {
  EXPECT_NE(flagcxNetIbP2p.name, nullptr);
  EXPECT_STREQ(flagcxNetIbP2p.name, "IB_P2P");

  // Basic
  EXPECT_NE(flagcxNetIbP2p.init, nullptr);
  EXPECT_NE(flagcxNetIbP2p.devices, nullptr);
  EXPECT_NE(flagcxNetIbP2p.getProperties, nullptr);

  // Connection setup
  EXPECT_NE(flagcxNetIbP2p.listen, nullptr);
  EXPECT_NE(flagcxNetIbP2p.connect, nullptr);
  EXPECT_NE(flagcxNetIbP2p.accept, nullptr);
  EXPECT_NE(flagcxNetIbP2p.closeSend, nullptr);
  EXPECT_NE(flagcxNetIbP2p.closeRecv, nullptr);
  EXPECT_NE(flagcxNetIbP2p.closeListen, nullptr);

  // Memory registration
  EXPECT_NE(flagcxNetIbP2p.regMr, nullptr);
  EXPECT_NE(flagcxNetIbP2p.regMrDmaBuf, nullptr);
  EXPECT_NE(flagcxNetIbP2p.deregMr, nullptr);

  // Two-sided (stubs)
  EXPECT_NE(flagcxNetIbP2p.isend, nullptr);
  EXPECT_NE(flagcxNetIbP2p.irecv, nullptr);
  EXPECT_NE(flagcxNetIbP2p.iflush, nullptr);
  EXPECT_NE(flagcxNetIbP2p.test, nullptr);

  // One-sided
  EXPECT_NE(flagcxNetIbP2p.iput, nullptr);
  EXPECT_NE(flagcxNetIbP2p.iget, nullptr);
  EXPECT_NE(flagcxNetIbP2p.iputSignal, nullptr);

  // Device lookup
  EXPECT_NE(flagcxNetIbP2p.getDevFromName, nullptr);
}

// Two-sided stubs should return errors
TEST(P2pAdaptorStruct, TwoSidedStubsReturnError) {
  void *dummy = nullptr;
  EXPECT_NE(
      flagcxNetIbP2p.isend(nullptr, nullptr, 0, 0, nullptr, nullptr, &dummy),
      flagcxSuccess);
  EXPECT_NE(flagcxNetIbP2p.irecv(nullptr, 0, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, &dummy),
            flagcxSuccess);
  EXPECT_NE(
      flagcxNetIbP2p.iflush(nullptr, 0, nullptr, nullptr, nullptr, &dummy),
      flagcxSuccess);
}

// ---------------------------------------------------------------------------
// 2. Init + Devices — requires IB hardware
// ---------------------------------------------------------------------------
TEST_F(P2pAdaptorTest, InitSucceeds) { EXPECT_EQ(initResult, flagcxSuccess); }

TEST_F(P2pAdaptorTest, DevicesReturnsPositive) {
  ASSERT_EQ(initResult, flagcxSuccess);
  ASSERT_GT(nDevs, 0)
      << "The IB P2P integration suite requires at least one usable RDMA "
         "device";
  EXPECT_EQ(selectionResult, flagcxSuccess)
      << "Failed to select an RDMA device using the current GPU topology";
  EXPECT_GE(netDev, 0);
  EXPECT_LT(netDev, nDevs);
}

TEST_F(P2pAdaptorTest, InitIsIdempotent) {
  // Calling init again should succeed without side effects
  EXPECT_EQ(flagcxNetIbP2p.init(), flagcxSuccess);
  int nDevs2 = 0;
  EXPECT_EQ(flagcxNetIbP2p.devices(&nDevs2), flagcxSuccess);
  EXPECT_EQ(nDevs2, nDevs);
}

// ---------------------------------------------------------------------------
// 3. GetProperties — requires IB hardware
// ---------------------------------------------------------------------------
TEST_F(P2pAdaptorTest, GetPropertiesForEachDevice) {
  if (!hasIbDevices())
    GTEST_SKIP() << "No IB devices available after the requirement failed";

  for (int d = 0; d < nDevs; d++) {
    flagcxNetProperties_t props;
    memset(&props, 0, sizeof(props));
    EXPECT_EQ(flagcxNetIbP2p.getProperties(d, &props), flagcxSuccess);
    EXPECT_GT(props.speed, 0);
    EXPECT_NE(props.name, nullptr);
  }
}

// ---------------------------------------------------------------------------
// 4. Listen + Connect + Accept loopback — requires IB hardware
// ---------------------------------------------------------------------------
class P2pLoopbackTest : public P2pAdaptorTest {
protected:
  void SetUp() override {
    if (!hasIbDevices())
      GTEST_SKIP() << "No IB devices available after the requirement failed";
    ASSERT_EQ(selectionResult, flagcxSuccess)
        << "Failed to select a topology-local RDMA device";
  }
};

TEST_F(P2pAdaptorTest, ListenConnectAcceptCloseForEachDevice) {
  if (!hasIbDevices())
    GTEST_SKIP() << "No IB devices available after the requirement failed";

  for (int dev = 0; dev < nDevs; ++dev) {
    flagcxNetProperties_t props = {};
    ASSERT_EQ(flagcxNetIbP2p.getProperties(dev, &props), flagcxSuccess);
    SCOPED_TRACE(::testing::Message()
                 << "netDev=" << dev << " name="
                 << (props.name != nullptr ? props.name : "<unnamed>")
                 << " pciPath="
                 << (props.pciPath != nullptr ? props.pciPath : "<unknown>")
                 << " port=" << props.port << " speed=" << props.speed);

    // Listen
    char handle[FLAGCX_NET_HANDLE_MAXSIZE];
    void *listenComm = nullptr;
    ASSERT_EQ(flagcxNetIbP2p.listen(dev, handle, &listenComm), flagcxSuccess);
    ASSERT_NE(listenComm, nullptr);

    // Connect + Accept in parallel using std::async with timeout
    auto acceptFuture = std::async(std::launch::async, [&]() {
      void *comm = nullptr;
      flagcxResult_t r = flagcxNetIbP2p.accept(listenComm, &comm);
      return std::make_pair(r, comm);
    });

    auto connectFuture = std::async(std::launch::async, [&]() {
      void *comm = nullptr;
      flagcxResult_t r = flagcxNetIbP2p.connect(dev, handle, &comm);
      return std::make_pair(r, comm);
    });

    // Wait with timeout to avoid hanging forever
    auto timeout = std::chrono::seconds(10);

    ASSERT_EQ(connectFuture.wait_for(timeout), std::future_status::ready)
        << "connect() timed out after 10s";
    auto [connectResult, sendComm] = connectFuture.get();

    ASSERT_EQ(acceptFuture.wait_for(timeout), std::future_status::ready)
        << "accept() timed out after 10s";
    auto [acceptResult, recvComm] = acceptFuture.get();

    ASSERT_EQ(connectResult, flagcxSuccess) << "connect() failed";
    ASSERT_EQ(acceptResult, flagcxSuccess) << "accept() failed";
    ASSERT_NE(sendComm, nullptr);
    ASSERT_NE(recvComm, nullptr);

    // Close
    EXPECT_EQ(flagcxNetIbP2p.closeSend(sendComm), flagcxSuccess);
    EXPECT_EQ(flagcxNetIbP2p.closeRecv(recvComm), flagcxSuccess);
    EXPECT_EQ(flagcxNetIbP2p.closeListen(listenComm), flagcxSuccess);
  }
}

namespace {

struct P2pPairResources {
  void *listenComm = nullptr;
  void *sendComm = nullptr;
  void *recvComm = nullptr;
  void *localSrcMr = nullptr;
  void *remoteDstMr = nullptr;
  void *remoteSrcMr = nullptr;
  void *localDstMr = nullptr;

  ~P2pPairResources() {
    if (localSrcMr != nullptr)
      flagcxNetIbP2p.deregMr(sendComm, localSrcMr);
    if (localDstMr != nullptr)
      flagcxNetIbP2p.deregMr(sendComm, localDstMr);
    if (remoteDstMr != nullptr)
      flagcxNetIbP2p.deregMr(recvComm, remoteDstMr);
    if (remoteSrcMr != nullptr)
      flagcxNetIbP2p.deregMr(recvComm, remoteSrcMr);
    if (sendComm != nullptr)
      flagcxNetIbP2p.closeSend(sendComm);
    if (recvComm != nullptr)
      flagcxNetIbP2p.closeRecv(recvComm);
    if (listenComm != nullptr)
      flagcxNetIbP2p.closeListen(listenComm);
  }
};

static flagcxResult_t waitP2pRequest(void *request) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    int done = 0;
    flagcxResult_t result = flagcxNetIbP2p.test(request, &done, nullptr);
    if (result != flagcxSuccess || done)
      return result;
    std::this_thread::yield();
  }
  return flagcxInProgress;
}

enum class P2pPairOperation { Write, Read };

static ::testing::AssertionResult
runHostPairOperation(int sendDev, int recvDev, P2pPairOperation operation) {
  flagcxNetProperties_t sendProps = {};
  flagcxNetProperties_t recvProps = {};
  flagcxResult_t result = flagcxNetIbP2p.getProperties(sendDev, &sendProps);
  if (result != flagcxSuccess)
    return ::testing::AssertionFailure()
           << "getProperties(sendDev=" << sendDev << ") returned " << result;
  result = flagcxNetIbP2p.getProperties(recvDev, &recvProps);
  if (result != flagcxSuccess)
    return ::testing::AssertionFailure()
           << "getProperties(recvDev=" << recvDev << ") returned " << result;

  std::ostringstream pair;
  pair << (operation == P2pPairOperation::Write ? "WRITE " : "READ ")
       << "sendDev=" << sendDev << "("
       << (sendProps.name != nullptr ? sendProps.name : "<unnamed>") << ", "
       << (sendProps.pciPath != nullptr ? sendProps.pciPath : "<unknown>")
       << ") recvDev=" << recvDev << "("
       << (recvProps.name != nullptr ? recvProps.name : "<unnamed>") << ", "
       << (recvProps.pciPath != nullptr ? recvProps.pciPath : "<unknown>")
       << ")";

  constexpr size_t bufferSize = 4096;
  std::vector<unsigned char> localSrc(bufferSize, 0xA5);
  std::vector<unsigned char> remoteDst(bufferSize, 0);
  std::vector<unsigned char> remoteSrc(bufferSize, 0x5A);
  std::vector<unsigned char> localDst(bufferSize, 0);

  // Keep the registered storage alive until after P2pPairResources has
  // deregistered every MR and closed the associated communicators.
  P2pPairResources resources;
  char handle[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  result = flagcxNetIbP2p.listen(recvDev, handle, &resources.listenComm);
  if (result != flagcxSuccess || resources.listenComm == nullptr)
    return ::testing::AssertionFailure()
           << pair.str() << ": listen returned " << result;

  auto acceptFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxResult_t acceptResult =
        flagcxNetIbP2p.accept(resources.listenComm, &comm);
    return std::make_pair(acceptResult, comm);
  });
  auto connectFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxResult_t connectResult =
        flagcxNetIbP2p.connect(sendDev, handle, &comm);
    return std::make_pair(connectResult, comm);
  });

  const auto connectDeadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  const bool connectReady =
      connectFuture.wait_until(connectDeadline) == std::future_status::ready;
  const bool acceptReady =
      acceptFuture.wait_until(connectDeadline) == std::future_status::ready;
  if (!connectReady || !acceptReady) {
    // std::future destruction may wait for std::launch::async work. Abort the
    // shared listener first so accept and its peer socket operations can exit,
    // then join both tasks and preserve any returned comms for RAII cleanup.
    flagcxResult_t abortResult =
        flagcxNetIbP2pAbortListen(resources.listenComm);
    auto connectResult = connectFuture.get();
    auto acceptResult = acceptFuture.get();
    resources.sendComm = connectResult.second;
    resources.recvComm = acceptResult.second;
    return ::testing::AssertionFailure()
           << pair.str() << ": connection did not finish before the 10-second "
           << "abort deadline (connectReady=" << connectReady
           << ", acceptReady=" << acceptReady << ", abortResult=" << abortResult
           << ", connectResult=" << connectResult.first
           << ", acceptResult=" << acceptResult.first << ")";
  }
  auto connectResult = connectFuture.get();
  auto acceptResult = acceptFuture.get();
  resources.sendComm = connectResult.second;
  resources.recvComm = acceptResult.second;
  if (connectResult.first != flagcxSuccess || resources.sendComm == nullptr ||
      acceptResult.first != flagcxSuccess || resources.recvComm == nullptr) {
    return ::testing::AssertionFailure()
           << pair.str() << ": connect returned " << connectResult.first
           << ", accept returned " << acceptResult.first;
  }

  const int mrFlags = FLAGCX_NET_MR_FLAG_NONE;

  void *request = nullptr;
  if (operation == P2pPairOperation::Write) {
    result =
        flagcxNetIbP2p.regMr(resources.sendComm, localSrc.data(), bufferSize,
                             FLAGCX_PTR_HOST, mrFlags, &resources.localSrcMr);
    if (result != flagcxSuccess)
      return ::testing::AssertionFailure()
             << pair.str() << ": local source registration returned " << result;
    result =
        flagcxNetIbP2p.regMr(resources.recvComm, remoteDst.data(), bufferSize,
                             FLAGCX_PTR_HOST, mrFlags, &resources.remoteDstMr);
    if (result != flagcxSuccess)
      return ::testing::AssertionFailure()
             << pair.str() << ": remote destination registration returned "
             << result;

    result = flagcxNetIbP2p.iput(
        resources.sendComm, 0, 0, bufferSize, 0, 0,
        reinterpret_cast<void **>(resources.localSrcMr),
        reinterpret_cast<void **>(resources.remoteDstMr), &request);
    if (result != flagcxSuccess || request == nullptr)
      return ::testing::AssertionFailure()
             << pair.str() << ": iput returned " << result;
    result = waitP2pRequest(request);
    if (result != flagcxSuccess)
      return ::testing::AssertionFailure()
             << pair.str() << ": completion returned " << result;
    if (localSrc != remoteDst)
      return ::testing::AssertionFailure() << pair.str() << ": data mismatch";
  } else {
    result =
        flagcxNetIbP2p.regMr(resources.recvComm, remoteSrc.data(), bufferSize,
                             FLAGCX_PTR_HOST, mrFlags, &resources.remoteSrcMr);
    if (result != flagcxSuccess)
      return ::testing::AssertionFailure()
             << pair.str() << ": remote source registration returned "
             << result;
    result =
        flagcxNetIbP2p.regMr(resources.sendComm, localDst.data(), bufferSize,
                             FLAGCX_PTR_HOST, mrFlags, &resources.localDstMr);
    if (result != flagcxSuccess)
      return ::testing::AssertionFailure()
             << pair.str() << ": local destination registration returned "
             << result;

    result = flagcxNetIbP2p.iget(
        resources.sendComm, 0, 0, bufferSize, 0, 0,
        reinterpret_cast<void **>(resources.remoteSrcMr),
        reinterpret_cast<void **>(resources.localDstMr), &request);
    if (result != flagcxSuccess || request == nullptr)
      return ::testing::AssertionFailure()
             << pair.str() << ": iget returned " << result;
    result = waitP2pRequest(request);
    if (result != flagcxSuccess)
      return ::testing::AssertionFailure()
             << pair.str() << ": completion returned " << result;
    if (remoteSrc != localDst)
      return ::testing::AssertionFailure() << pair.str() << ": data mismatch";
  }

  return ::testing::AssertionSuccess();
}

} // namespace

// Exercise every ordered local-HCA pair. This is opt-in because it is a
// hardware route diagnostic rather than a topology-selected production path.
// CI uses a short RC retry budget so a broken pair reports its WC error instead
// of consuming the suite timeout.
TEST_F(P2pAdaptorTest, HostWriteReadForEveryOrderedHcaPair) {
  const char *enabled = std::getenv("FLAGCX_CI_P2P_HCA_PAIR_MATRIX");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0)
    GTEST_SKIP() << "HCA pair matrix is enabled only in its dedicated CI run";
  ASSERT_TRUE(hasIbDevices());

  for (int sendDev = 0; sendDev < nDevs; ++sendDev) {
    for (int recvDev = 0; recvDev < nDevs; ++recvDev) {
      EXPECT_TRUE(
          runHostPairOperation(sendDev, recvDev, P2pPairOperation::Write));
      EXPECT_TRUE(
          runHostPairOperation(sendDev, recvDev, P2pPairOperation::Read));
    }
  }
}

// ---------------------------------------------------------------------------
// 5. RegMr + DeregMr — requires IB hardware + a loopback connection
// ---------------------------------------------------------------------------
TEST_F(P2pLoopbackTest, RegMrDeregMr) {
  // Set up loopback connection
  char handle[FLAGCX_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.listen(netDev, handle, &listenComm), flagcxSuccess)
      << "topology-selected netDev=" << netDev;

  auto acceptFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxNetIbP2p.accept(listenComm, &comm);
    return comm;
  });
  auto connectFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxNetIbP2p.connect(netDev, handle, &comm);
    return comm;
  });

  auto timeout = std::chrono::seconds(10);
  ASSERT_EQ(connectFuture.wait_for(timeout), std::future_status::ready)
      << "connect() timed out";
  void *sendComm = connectFuture.get();
  ASSERT_EQ(acceptFuture.wait_for(timeout), std::future_status::ready)
      << "accept() timed out";
  void *recvComm = acceptFuture.get();
  ASSERT_NE(sendComm, nullptr);
  ASSERT_NE(recvComm, nullptr);

  // Register MR on send side
  const size_t bufSize = 4096;
  void *buf = malloc(bufSize);
  ASSERT_NE(buf, nullptr);
  memset(buf, 0, bufSize);

  void *mhandle = nullptr;
  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;
  ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm, buf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &mhandle),
            flagcxSuccess);
  ASSERT_NE(mhandle, nullptr);

  // Deregister
  EXPECT_EQ(flagcxNetIbP2p.deregMr(sendComm, mhandle), flagcxSuccess);

  // Register on recv side too (symmetric)
  void *mhandle2 = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.regMr(recvComm, buf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &mhandle2),
            flagcxSuccess);
  ASSERT_NE(mhandle2, nullptr);
  EXPECT_EQ(flagcxNetIbP2p.deregMr(recvComm, mhandle2), flagcxSuccess);

  free(buf);
  flagcxNetIbP2p.closeSend(sendComm);
  flagcxNetIbP2p.closeRecv(recvComm);
  flagcxNetIbP2p.closeListen(listenComm);
}

// ---------------------------------------------------------------------------
// 6. Iput + Test — requires IB hardware + loopback
// ---------------------------------------------------------------------------
TEST_F(P2pLoopbackTest, IputAndTest) {
  // Set up loopback connection
  char handle[FLAGCX_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.listen(netDev, handle, &listenComm), flagcxSuccess)
      << "topology-selected netDev=" << netDev;

  auto acceptFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxNetIbP2p.accept(listenComm, &comm);
    return comm;
  });
  auto connectFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxNetIbP2p.connect(netDev, handle, &comm);
    return comm;
  });

  auto timeout = std::chrono::seconds(10);
  ASSERT_EQ(connectFuture.wait_for(timeout), std::future_status::ready)
      << "connect() timed out";
  void *sendComm = connectFuture.get();
  ASSERT_EQ(acceptFuture.wait_for(timeout), std::future_status::ready)
      << "accept() timed out";
  void *recvComm = acceptFuture.get();
  ASSERT_NE(sendComm, nullptr) << "topology-selected netDev=" << netDev;
  ASSERT_NE(recvComm, nullptr) << "topology-selected netDev=" << netDev;

  // Allocate and register src + dst buffers
  const size_t bufSize = 4096;
  void *srcBuf = malloc(bufSize);
  void *dstBuf = malloc(bufSize);
  ASSERT_NE(srcBuf, nullptr);
  ASSERT_NE(dstBuf, nullptr);

  // Fill src with pattern, dst with zeros
  memset(srcBuf, 0xAB, bufSize);
  memset(dstBuf, 0, bufSize);

  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;

  void *srcMr = nullptr;
  void *dstMr = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm, srcBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &srcMr),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm, dstBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &dstMr),
            flagcxSuccess);

  // Iput: write srcBuf -> dstBuf via RDMA
  void *request = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.iput(sendComm, 0, 0, bufSize, 0, 0, (void **)srcMr,
                                (void **)dstMr, &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);

  // Poll until done
  int done = 0;
  int sizes = 0;
  int polls = 0;
  while (!done && polls < 1000000) {
    ASSERT_EQ(flagcxNetIbP2p.test(request, &done, &sizes), flagcxSuccess);
    polls++;
  }
  EXPECT_TRUE(done) << "iput did not complete within poll limit";

  // Verify data was written
  EXPECT_EQ(memcmp(srcBuf, dstBuf, bufSize), 0)
      << "RDMA write did not transfer data correctly";

  // Cleanup
  flagcxNetIbP2p.deregMr(sendComm, srcMr);
  flagcxNetIbP2p.deregMr(sendComm, dstMr);
  free(srcBuf);
  free(dstBuf);
  flagcxNetIbP2p.closeSend(sendComm);
  flagcxNetIbP2p.closeRecv(recvComm);
  flagcxNetIbP2p.closeListen(listenComm);
}

// ---------------------------------------------------------------------------
// 7. Close with NULL is safe
// ---------------------------------------------------------------------------
TEST(P2pAdaptorStruct, CloseNullIsSafe) {
  EXPECT_EQ(flagcxNetIbP2p.closeSend(nullptr), flagcxSuccess);
  EXPECT_EQ(flagcxNetIbP2p.closeRecv(nullptr), flagcxSuccess);
  EXPECT_EQ(flagcxNetIbP2p.closeListen(nullptr), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// 8. Test with NULL request returns done immediately
// ---------------------------------------------------------------------------
TEST(P2pAdaptorStruct, TestNullRequestIsDone) {
  int done = 0;
  int sizes = 0;
  EXPECT_EQ(flagcxNetIbP2p.test(nullptr, &done, &sizes), flagcxSuccess);
  EXPECT_EQ(done, 1);
}
