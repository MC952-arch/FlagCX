/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_P2P_TRANSPORT_H_
#define FLAGCX_P2P_TRANSPORT_H_

#include "flagcx.h"

enum flagcxP2pTransferPlan {
  flagcxP2pPlanPending = 0,
  flagcxP2pPlanFifo = 1,
  flagcxP2pPlanRead = 2,
  flagcxP2pPlanWrite = 3,
};

// Resolve the common data path after both peers have published their local
// registration result. Any IPC export/import failure forces both peers onto
// FIFO so they cannot enter different protocols for the same operation.
flagcxResult_t flagcxP2pResolveTransferPlan(int senderRegistered,
                                            int receiverRegistered,
                                            int senderForceFifo,
                                            int receiverForceFifo,
                                            enum flagcxP2pTransferPlan *plan);

#endif // FLAGCX_P2P_TRANSPORT_H_
