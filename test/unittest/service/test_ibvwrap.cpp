#include "ib_common.h"
#include "ib_gid.h"
#include "ib_retrans.h"
#include "ibvsymbols.h"
#include "ibvwrap.h"

#include <cerrno>
#include <gtest/gtest.h>

namespace {

int postResult = IBV_SUCCESS;
int postCalls = 0;
ibv_send_wr *firstRejected = nullptr;

int queryGidExResult = 0;
int queryGidExErrno = 0;

int fakeQueryGidEx(ibv_context *, uint32_t portNum, uint32_t gidIndex,
                   flagcxIbGidEntry *entry, uint32_t, size_t entrySize) {
  if (entrySize != sizeof(*entry))
    return EINVAL;
  errno = queryGidExErrno;
  if (queryGidExResult != 0)
    return queryGidExResult;
  entry->gidIndex = gidIndex;
  entry->portNum = portNum;
  entry->gidType = FLAGCX_IB_GID_TYPE_ROCE_V2;
  entry->ndevIfindex = 7;
  return 0;
}

class ScopedQueryGidExSymbol {
public:
  explicit ScopedQueryGidExSymbol(
      decltype(ibvSymbols.ibv_internal_query_gid_ex) replacement)
      : saved_(ibvSymbols.ibv_internal_query_gid_ex) {
    ibvSymbols.ibv_internal_query_gid_ex = replacement;
  }

  ~ScopedQueryGidExSymbol() { ibvSymbols.ibv_internal_query_gid_ex = saved_; }

private:
  decltype(ibvSymbols.ibv_internal_query_gid_ex) saved_;
};

int fakePostSend(ibv_qp *, ibv_send_wr *, ibv_send_wr **badWr) {
  ++postCalls;
  if (badWr != nullptr)
    *badWr = firstRejected;
  return postResult;
}

} // namespace

TEST(IbvWrapOneSidedPost, ClassifiesOnlySendQueuePressureAsRetryable) {
  EXPECT_EQ(flagcxIbOneSidedPostResult(IBV_SUCCESS), flagcxSuccess);
  EXPECT_EQ(flagcxIbOneSidedPostResult(ENOMEM), flagcxInProgress);
  EXPECT_EQ(flagcxIbOneSidedPostResult(EINVAL), flagcxSystemError);
  EXPECT_EQ(flagcxIbOneSidedPostResult(EIO), flagcxSystemError);
}

TEST(IbvWrapOneSidedPost, ReportsSendQueuePressureAfterOnePostAttempt) {
  ibv_context context = {};
  ibv_qp qp = {};
  ibv_send_wr wr = {};
  ibv_send_wr *badWr = nullptr;
  context.ops.post_send = fakePostSend;
  qp.context = &context;
  postResult = ENOMEM;
  postCalls = 0;
  firstRejected = &wr;

  EXPECT_EQ(flagcxWrapIbvPostSendOneSided(&qp, &wr, &badWr), flagcxInProgress);
  EXPECT_EQ(postCalls, 1);
  EXPECT_EQ(badWr, &wr);
}

TEST(IbvWrapOneSidedPost, ReportsPermanentFailureWithoutRetrying) {
  ibv_context context = {};
  ibv_qp qp = {};
  ibv_send_wr wrs[2] = {};
  ibv_send_wr *badWr = nullptr;
  wrs[0].next = &wrs[1];
  context.ops.post_send = fakePostSend;
  qp.context = &context;
  postResult = EINVAL;
  postCalls = 0;
  firstRejected = &wrs[1];

  EXPECT_EQ(flagcxWrapIbvPostSendOneSided(&qp, wrs, &badWr), flagcxSystemError);
  EXPECT_EQ(postCalls, 1);
  EXPECT_EQ(badWr, &wrs[1]);
}

TEST(IbvCompatLid, ConvertsPortLidToPortableMetadata) {
  ibv_port_attr portAttr = {};
#ifdef USE_SHCA
  constexpr uint32_t lid = 91972;
  portAttr.lid = u32_to_17(lid);
#else
  constexpr uint32_t lid = 1234;
  portAttr.lid = static_cast<uint16_t>(lid);
#endif

  EXPECT_EQ(flagcxIbPortLid(&portAttr), lid);
}

TEST(IbvCompatLid, ConvertsPortableMetadataToAhDlid) {
  ibv_ah_attr ahAttr = {};
#ifdef USE_SHCA
  const uint32_t lids[] = {0, 65535, 65536, 91972, 91984, 91986, 131071};
  for (uint32_t lid : lids) {
    ahAttr = {};
    ASSERT_EQ(flagcxIbSetAhDlid(&ahAttr, lid), flagcxSuccess) << "lid=" << lid;
    EXPECT_EQ(u17_to_32(ahAttr.dlid), lid) << "lid=" << lid;
  }
#else
  constexpr uint32_t lid = 1234;
  ASSERT_EQ(flagcxIbSetAhDlid(&ahAttr, lid), flagcxSuccess);
  EXPECT_EQ(ahAttr.dlid, lid);
  EXPECT_EQ(flagcxIbSetAhDlid(&ahAttr, UINT16_MAX + 1U), flagcxInvalidArgument);
#endif
}

TEST(IbvCompatRoute, SelectsGlobalRouteForTheActiveVerbsAbi) {
  EXPECT_TRUE(flagcxIbUseGlobalRoute(IBV_LINK_LAYER_ETHERNET));
#ifdef USE_SHCA
  EXPECT_TRUE(flagcxIbUseGlobalRoute(IBV_LINK_LAYER_INFINIBAND));
#else
  EXPECT_FALSE(flagcxIbUseGlobalRoute(IBV_LINK_LAYER_INFINIBAND));
#endif
}

TEST(IbvCompatLid, ConnectionMetadataPreservesExtendedControlLid) {
  flagcxIbConnectionMetadata metadata = {};
  metadata.ctrlLid[0] = 91972;
  EXPECT_EQ(metadata.ctrlLid[0], 91972U);
}

TEST(IbvCompatRetrans, ReportsUdControlChannelCapability) {
#ifdef USE_SHCA
  EXPECT_FALSE(flagcxIbRetransUdSupported());
#else
  EXPECT_TRUE(flagcxIbRetransUdSupported());
#endif
}

TEST(IbvAtomicDepth, ClampsResponderDepthToDeviceCapability) {
  EXPECT_EQ(flagcxIbResponderAtomicDepth(16, 32), 16);
  EXPECT_EQ(flagcxIbResponderAtomicDepth(16, 8), 8);
  EXPECT_EQ(flagcxIbResponderAtomicDepth(512, 512), UINT8_MAX);
  EXPECT_EQ(flagcxIbResponderAtomicDepth(0, 16), 0);
  EXPECT_EQ(flagcxIbResponderAtomicDepth(16, 0), 0);
}

TEST(IbvAtomicDepth, NegotiatesInitiatorDepthWithRemoteResponder) {
  EXPECT_EQ(flagcxIbInitiatorAtomicDepth(16, 32, 16), 16);
  EXPECT_EQ(flagcxIbInitiatorAtomicDepth(16, 8, 16), 8);
  EXPECT_EQ(flagcxIbInitiatorAtomicDepth(16, 32, 4), 4);
  EXPECT_EQ(flagcxIbInitiatorAtomicDepth(512, 512, 512), UINT8_MAX);
  EXPECT_EQ(flagcxIbInitiatorAtomicDepth(16, 16, 0), 0);
}

TEST(IbvCompatGid, ExtendedQueryWithoutCallbackIsUnsupported) {
  ibv_context context = {};
  flagcxIbGidEntry entry = {};

  ScopedQueryGidExSymbol symbol(nullptr);
  EXPECT_EQ(flagcxWrapIbvQueryGidEx(&context, 1, 2, &entry, 0),
            flagcxNotSupported);
}

TEST(IbvCompatGid, ExtendedQueryRejectsNullArguments) {
  ibv_context context = {};
  flagcxIbGidEntry entry = {};
  ScopedQueryGidExSymbol symbol(fakeQueryGidEx);

  EXPECT_EQ(flagcxWrapIbvQueryGidEx(nullptr, 1, 2, &entry, 0),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxWrapIbvQueryGidEx(&context, 1, 2, nullptr, 0),
            flagcxInvalidArgument);
}

TEST(IbvCompatGid, ExtendedQueryReturnsMetadata) {
  ibv_context context = {};
  flagcxIbGidEntry entry = {};
  ScopedQueryGidExSymbol symbol(fakeQueryGidEx);
  queryGidExResult = 0;
  queryGidExErrno = 0;

  EXPECT_EQ(flagcxWrapIbvQueryGidEx(&context, 3, 4, &entry, 0), flagcxSuccess);
  EXPECT_EQ(entry.portNum, 3U);
  EXPECT_EQ(entry.gidIndex, 4U);
  EXPECT_EQ(entry.gidType, FLAGCX_IB_GID_TYPE_ROCE_V2);
  EXPECT_EQ(entry.ndevIfindex, 7U);
}

TEST(IbvCompatGid, ExtendedQueryClassifiesUnsupportedResults) {
  ibv_context context = {};
  flagcxIbGidEntry entry = {};
  ScopedQueryGidExSymbol symbol(fakeQueryGidEx);

  queryGidExErrno = 0;
  const int unsupportedResults[] = {ENOTSUP, EOPNOTSUPP, ENOSYS, -EOPNOTSUPP};
  for (int result : unsupportedResults) {
    queryGidExResult = result;
    EXPECT_EQ(flagcxWrapIbvQueryGidEx(&context, 1, 2, &entry, 0),
              flagcxNotSupported);
  }

  queryGidExResult = -1;
  queryGidExErrno = EOPNOTSUPP;
  EXPECT_EQ(flagcxWrapIbvQueryGidEx(&context, 1, 2, &entry, 0),
            flagcxNotSupported);
}

TEST(IbvCompatGid, ExtendedQueryReportsPermanentFailure) {
  ibv_context context = {};
  flagcxIbGidEntry entry = {};
  ScopedQueryGidExSymbol symbol(fakeQueryGidEx);
  queryGidExResult = EIO;
  queryGidExErrno = EIO;

  EXPECT_EQ(flagcxWrapIbvQueryGidEx(&context, 1, 2, &entry, 0),
            flagcxSystemError);
}
