/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_RUNNER_RESULT_H_
#define FLAGCX_RUNNER_RESULT_H_

#include "flagcx.h"

static inline bool flagcxRunnerResultAccepted(flagcxResult_t result) {
  return result == flagcxSuccess || result == flagcxInProgress;
}

static inline void flagcxRunnerRecordFirstError(flagcxResult_t candidate,
                                                flagcxResult_t *firstError) {
  if (firstError != nullptr &&
      (*firstError == flagcxSuccess || *firstError == flagcxInProgress) &&
      candidate != flagcxSuccess && candidate != flagcxInProgress) {
    *firstError = candidate;
  }
}

template <typename Start, typename Body, typename End>
static inline flagcxResult_t flagcxRunnerRunGroup(Start start, Body body,
                                                  End end) {
  flagcxResult_t startResult = start();
  if (!flagcxRunnerResultAccepted(startResult))
    return startResult;

  flagcxResult_t firstError = flagcxSuccess;
  flagcxRunnerRecordFirstError(body(), &firstError);
  flagcxRunnerRecordFirstError(end(), &firstError);
  return firstError;
}

static inline flagcxResult_t
flagcxRunnerClassifyEventQuery(flagcxResult_t queryResult, bool *completed) {
  if (completed == nullptr)
    return flagcxInvalidArgument;
  *completed = false;
  if (queryResult == flagcxSuccess) {
    *completed = true;
    return flagcxSuccess;
  }
  if (queryResult == flagcxInProgress)
    return flagcxSuccess;
  return queryResult;
}

#endif // FLAGCX_RUNNER_RESULT_H_
