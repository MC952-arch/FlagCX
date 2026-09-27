/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "runner_result.h"
#include <gtest/gtest.h>

TEST(RunnerResult, AcceptedResultIncludesInProgress) {
  EXPECT_TRUE(flagcxRunnerResultAccepted(flagcxSuccess));
  EXPECT_TRUE(flagcxRunnerResultAccepted(flagcxInProgress));
  EXPECT_FALSE(flagcxRunnerResultAccepted(flagcxSystemError));
}

TEST(RunnerResult, EventQueryClassifiesCompletionAndErrors) {
  bool completed = false;

  EXPECT_EQ(flagcxRunnerClassifyEventQuery(flagcxSuccess, &completed),
            flagcxSuccess);
  EXPECT_TRUE(completed);

  completed = true;
  EXPECT_EQ(flagcxRunnerClassifyEventQuery(flagcxInProgress, &completed),
            flagcxSuccess);
  EXPECT_FALSE(completed);

  EXPECT_EQ(flagcxRunnerClassifyEventQuery(flagcxSystemError, &completed),
            flagcxSystemError);
  EXPECT_FALSE(completed);
  EXPECT_EQ(flagcxRunnerClassifyEventQuery(flagcxSuccess, nullptr),
            flagcxInvalidArgument);
}

TEST(RunnerResult, CleanupPreservesFirstError) {
  flagcxResult_t result = flagcxSuccess;
  flagcxRunnerRecordFirstError(flagcxInProgress, &result);
  EXPECT_EQ(result, flagcxSuccess);
  flagcxRunnerRecordFirstError(flagcxSystemError, &result);
  flagcxRunnerRecordFirstError(flagcxInvalidArgument, &result);
  EXPECT_EQ(result, flagcxSystemError);
}

TEST(RunnerResult, GroupAlwaysEndsAfterSuccessfulStart) {
  int starts = 0;
  int bodies = 0;
  int ends = 0;
  flagcxResult_t result = flagcxRunnerRunGroup(
      [&]() {
        starts++;
        return flagcxSuccess;
      },
      [&]() {
        bodies++;
        return flagcxSystemError;
      },
      [&]() {
        ends++;
        return flagcxInvalidArgument;
      });

  EXPECT_EQ(result, flagcxSystemError);
  EXPECT_EQ(starts, 1);
  EXPECT_EQ(bodies, 1);
  EXPECT_EQ(ends, 1);
}

TEST(RunnerResult, FailedGroupStartDoesNotRunBodyOrEnd) {
  int bodies = 0;
  int ends = 0;
  flagcxResult_t result =
      flagcxRunnerRunGroup([]() { return flagcxSystemError; },
                           [&]() {
                             bodies++;
                             return flagcxSuccess;
                           },
                           [&]() {
                             ends++;
                             return flagcxSuccess;
                           });

  EXPECT_EQ(result, flagcxSystemError);
  EXPECT_EQ(bodies, 0);
  EXPECT_EQ(ends, 0);
}

TEST(RunnerResult, GroupNormalizesAcceptedAsyncResults) {
  int posted = 0;
  int ends = 0;
  flagcxResult_t results[] = {flagcxSuccess, flagcxInProgress, flagcxSuccess};
  flagcxResult_t result =
      flagcxRunnerRunGroup([]() { return flagcxInProgress; },
                           [&]() {
                             for (flagcxResult_t opResult : results) {
                               if (!flagcxRunnerResultAccepted(opResult))
                                 return opResult;
                               posted++;
                             }
                             return flagcxInProgress;
                           },
                           [&]() {
                             ends++;
                             return flagcxInProgress;
                           });

  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_EQ(posted, 3);
  EXPECT_EQ(ends, 1);
}
