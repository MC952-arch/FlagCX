/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Select the verbs ABI used by every FlagCX RDMA translation unit. SHCA
 * installations provide a vendor verbs.h whose public structure layout is
 * not compatible with FlagCX's embedded rdma-core declarations.
 ************************************************************************/

#ifndef FLAGCX_IBV_COMPAT_H_
#define FLAGCX_IBV_COMPAT_H_

#include <stddef.h>

#if defined(USE_SHCA) || defined(FLAGCX_BUILD_RDMA_CORE)
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif

#ifdef USE_SHCA
#include <infiniband/shca_17b_types.h>
#endif

// ABI-compatible subset of struct ibv_gid_entry used by the optional
// _ibv_query_gid_ex symbol. Keeping a FlagCX-owned type lets the default
// dlopen build use the extended query without depending on the complete
// system verbs structure layout.
struct flagcxIbGidEntry {
  union ibv_gid gid;
  uint32_t gidIndex;
  uint32_t portNum;
  uint32_t gidType;
  uint32_t ndevIfindex;
};

static_assert(sizeof(struct flagcxIbGidEntry) == 32,
              "FlagCX GID entry must match the libibverbs ABI");
static_assert(offsetof(struct flagcxIbGidEntry, gid) == 0,
              "FlagCX GID value must match the libibverbs ABI");
static_assert(offsetof(struct flagcxIbGidEntry, gidIndex) == 16,
              "FlagCX GID index must match the libibverbs ABI");
static_assert(offsetof(struct flagcxIbGidEntry, portNum) == 20,
              "FlagCX GID port must match the libibverbs ABI");
static_assert(offsetof(struct flagcxIbGidEntry, gidType) == 24,
              "FlagCX GID type must match the libibverbs ABI");
static_assert(offsetof(struct flagcxIbGidEntry, ndevIfindex) == 28,
              "FlagCX GID network-device index must match the libibverbs ABI");

#endif // FLAGCX_IBV_COMPAT_H_
