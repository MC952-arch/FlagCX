#include "ib_common.h"
#include "ib_retrans.h"
#include "ibvwrap.h"

#include <cerrno>
#include <gtest/gtest.h>

namespace {

int postResult = IBV_SUCCESS;
int postCalls = 0;
ibv_send_wr *firstRejected = nullptr;

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
  constexpr uint32_t lid = 91972;
  ASSERT_EQ(flagcxIbSetAhDlid(&ahAttr, lid), flagcxSuccess);
  EXPECT_EQ(u17_to_32(ahAttr.dlid), lid);
#else
  constexpr uint32_t lid = 1234;
  ASSERT_EQ(flagcxIbSetAhDlid(&ahAttr, lid), flagcxSuccess);
  EXPECT_EQ(ahAttr.dlid, lid);
  EXPECT_EQ(flagcxIbSetAhDlid(&ahAttr, UINT16_MAX + 1U), flagcxInvalidArgument);
#endif
}

TEST(IbvCompatRetrans, ReportsUdControlChannelCapability) {
#if defined(USE_SHCA) && !defined(USE_IBUC)
  EXPECT_FALSE(flagcxIbRetransUdSupported());
#else
  EXPECT_TRUE(flagcxIbRetransUdSupported());
#endif
}
