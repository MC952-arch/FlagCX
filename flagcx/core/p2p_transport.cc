/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "p2p_transport.h"

flagcxResult_t flagcxP2pResolveTransferPlan(int senderRegistered,
                                            int receiverRegistered,
                                            int senderForceFifo,
                                            int receiverForceFifo,
                                            enum flagcxP2pTransferPlan *plan) {
  if (plan == NULL)
    return flagcxInvalidArgument;
  if ((senderRegistered != 0 && senderRegistered != 1) ||
      (receiverRegistered != 0 && receiverRegistered != 1) ||
      (senderForceFifo != 0 && senderForceFifo != 1) ||
      (receiverForceFifo != 0 && receiverForceFifo != 1))
    return flagcxInvalidArgument;

  if (senderForceFifo || receiverForceFifo)
    *plan = flagcxP2pPlanFifo;
  else if (receiverRegistered)
    *plan = flagcxP2pPlanWrite;
  else if (senderRegistered)
    *plan = flagcxP2pPlanRead;
  else
    *plan = flagcxP2pPlanFifo;
  return flagcxSuccess;
}
