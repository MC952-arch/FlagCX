/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Select the verbs ABI used by every FlagCX RDMA translation unit. SHCA
 * installations provide a vendor verbs.h whose public structure layout is
 * not compatible with FlagCX's embedded rdma-core declarations.
 ************************************************************************/

#ifndef FLAGCX_IBV_COMPAT_H_
#define FLAGCX_IBV_COMPAT_H_

#if defined(USE_SHCA) || defined(FLAGCX_BUILD_RDMA_CORE)
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif

#ifdef USE_SHCA
#include <infiniband/shca_17b_types.h>
#endif

#endif // FLAGCX_IBV_COMPAT_H_
