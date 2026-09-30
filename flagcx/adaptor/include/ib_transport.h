/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_IB_TRANSPORT_H_
#define FLAGCX_IB_TRANSPORT_H_

#include "ib_common.h"
#include "net_transport.h"

struct flagcxIbLane {
  struct flagcxNetLane base;
  struct flagcxIbQp *ibQp;
};

struct flagcxIbDataLanePolicy {
  enum flagcxNetLaneMode mode;
  uint64_t orderingKey;
  uint64_t *laneMask;
};

flagcxResult_t flagcxIbValidateDataLaneGeometry(int localNqps, int localSplit,
                                                int remoteNqps,
                                                int remoteSplit);

// Two-sided collective traffic uses a deterministic base lane when the core
// marks an ordering domain independent. Compatibility traffic retains the
// legacy round-robin cursor. stripeIndex walks the QPs inside that lane group.
void flagcxIbGetDataLanePolicy(struct flagcxIbDataLanePolicy *policy);
flagcxResult_t
flagcxIbSelectDataLane(struct flagcxIbNetCommBase *base,
                       const struct flagcxIbDataLanePolicy *policy,
                       uint32_t stripeIndex, struct flagcxIbLane *lane);
flagcxResult_t
flagcxIbPreviewDataLane(struct flagcxIbNetCommBase *base,
                        const struct flagcxIbDataLanePolicy *policy,
                        uint32_t stripeIndex, struct flagcxIbLane *lane);
flagcxResult_t
flagcxIbCommitDataLane(struct flagcxIbNetCommBase *base,
                       const struct flagcxIbDataLanePolicy *policy,
                       const struct flagcxIbLane *lane);

flagcxResult_t flagcxIbSelectLane(struct flagcxIbNetCommBase *base,
                                  enum flagcxNetLaneMode mode,
                                  uint64_t orderingKey,
                                  struct flagcxIbLane *lane);
flagcxResult_t flagcxIbCommitLane(struct flagcxIbNetCommBase *base,
                                  enum flagcxNetLaneMode mode,
                                  const struct flagcxIbLane *lane);

flagcxResult_t flagcxIbPostSendList(const struct flagcxIbLane *lane,
                                    struct ibv_send_wr *head, int count,
                                    bool retryable,
                                    struct flagcxNetPostResult *post);

flagcxResult_t flagcxIbResolveOneSideRange(
    const struct flagcxOneSideHandleInfo *info, int rank, uint64_t offset,
    size_t size, const struct flagcxIbLane *lane, bool local,
    struct flagcxNetResolvedRange *range, uint32_t *key);

#endif // FLAGCX_IB_TRANSPORT_H_
