/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include <errno.h>

#include "group.h"

namespace {

struct flagcxHeteroComm *invalidPreconnectNext() {
  return reinterpret_cast<struct flagcxHeteroComm *>(0x1);
}

TEST(GroupPreconnectOwnership, TakeAllTransfersAndClearsSource) {
  flagcxHeteroComm first = {};
  flagcxHeteroComm *source = &first;

  flagcxHeteroComm *owned = flagcxGroupCommPreconnectTakeAll(&source);

  EXPECT_EQ(owned, &first);
  EXPECT_EQ(source, nullptr);
}

TEST(GroupPreconnectOwnership, PopConsumesEachNodeExactlyOnce) {
  flagcxHeteroComm first = {};
  flagcxHeteroComm second = {};
  first.preconnectNext = &second;
  second.preconnectNext = nullptr;
  flagcxHeteroComm *owned = &first;

  EXPECT_EQ(flagcxGroupCommPreconnectPop(&owned), &first);
  EXPECT_EQ(owned, &second);
  EXPECT_EQ(first.preconnectNext, invalidPreconnectNext());

  EXPECT_EQ(flagcxGroupCommPreconnectPop(&owned), &second);
  EXPECT_EQ(owned, nullptr);
  EXPECT_EQ(second.preconnectNext, invalidPreconnectNext());

  EXPECT_EQ(flagcxGroupCommPreconnectPop(&owned), nullptr);
}

TEST(GroupPreconnectOwnership, PopDoesNotDereferenceSentinelHead) {
  flagcxHeteroComm *owned = invalidPreconnectNext();

  EXPECT_EQ(flagcxGroupCommPreconnectPop(&owned), nullptr);
  EXPECT_EQ(owned, invalidPreconnectNext());
}

int createCalls = 0;
int joinCalls = 0;
int failCreateAt = -1;
int failJoinAt = -1;

int fakeCreate(pthread_t *thread, const pthread_attr_t *, void *(*)(void *),
               void *arg) {
  int call = createCalls++;
  if (call == failCreateAt)
    return EAGAIN;
  *thread = pthread_t{};
  auto *job = static_cast<flagcxAsyncJob *>(arg);
  __atomic_store_n(&job->state, flagcxGroupJobDone, __ATOMIC_RELEASE);
  return 0;
}

int fakeJoin(pthread_t, void **) {
  int call = joinCalls++;
  return call == failJoinAt ? EINVAL : 0;
}

class GroupAsyncJobsTest : public ::testing::Test {
protected:
  void SetUp() override {
    createCalls = 0;
    joinCalls = 0;
    failCreateAt = -1;
    failJoinAt = -1;
    flagcxIntruQueueConstruct(&jobs);
    for (auto &job : storage) {
      job = {};
      job.state = flagcxGroupJobRunning;
      flagcxIntruQueueEnqueue(&jobs, &job);
    }
  }

  flagcxIntruQueue<flagcxAsyncJob, &flagcxAsyncJob::next> jobs;
  flagcxAsyncJob storage[3];
};

TEST_F(GroupAsyncJobsTest, RejectsMissingThreadOperations) {
  EXPECT_EQ(flagcxGroupLaunchAsyncJobs(nullptr, fakeCreate, fakeJoin),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxGroupLaunchAsyncJobs(&jobs, nullptr, fakeJoin),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxGroupLaunchAsyncJobs(&jobs, fakeCreate, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(createCalls, 0);
  EXPECT_EQ(joinCalls, 0);
}

TEST_F(GroupAsyncJobsTest, CreateFailureJoinsOnlyStartedPrefix) {
  failCreateAt = 1;

  EXPECT_EQ(flagcxGroupLaunchAsyncJobs(&jobs, fakeCreate, fakeJoin),
            flagcxSystemError);
  EXPECT_EQ(createCalls, 2);
  EXPECT_EQ(joinCalls, 1);
  EXPECT_EQ(storage[0].state, flagcxGroupJobJoined);
  EXPECT_EQ(storage[1].state, flagcxGroupJobRunning);
  EXPECT_EQ(storage[2].state, flagcxGroupJobRunning);
}

TEST_F(GroupAsyncJobsTest, JobFailureIsReturnedAfterAllThreadsJoin) {
  storage[1].result = flagcxRemoteError;

  EXPECT_EQ(flagcxGroupLaunchAsyncJobs(&jobs, fakeCreate, fakeJoin),
            flagcxRemoteError);
  EXPECT_EQ(createCalls, 3);
  EXPECT_EQ(joinCalls, 3);
  for (const auto &job : storage)
    EXPECT_EQ(job.state, flagcxGroupJobJoined);
}

TEST_F(GroupAsyncJobsTest, JoinFailureWaitsForCompletedJobBeforeReturning) {
  failJoinAt = 1;

  EXPECT_EQ(flagcxGroupLaunchAsyncJobs(&jobs, fakeCreate, fakeJoin),
            flagcxSystemError);
  EXPECT_EQ(createCalls, 3);
  EXPECT_EQ(joinCalls, 3);
  for (const auto &job : storage)
    EXPECT_EQ(job.state, flagcxGroupJobJoined);
}

} // namespace
