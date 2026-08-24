/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Internal definition of flagcxShmemCommInternal for xccl XSHMEM.
 * Shared between xshmem_adaptor.cc and the xshmem device-API backend.
 ************************************************************************/

#ifndef FLAGCX_XSHMEM_ADAPTOR_H_
#define FLAGCX_XSHMEM_ADAPTOR_H_

#include <xshmem/xshmem.h>
#include <stdint.h>


struct flagcxShmemCommInternal {
  int rank, nRanks;
  int intraRank, intraSize;
  xshmem_team_t intraTeam;
  xshmem_team_t interTeam;
  xshmem_team_t worldTeam;

  uint64_t *signalBuffer;
  int signalCount;
  uint64_t *counterBuffer;
  int counterCount;
  uint64_t *shadowBuffer;

  uint64_t
      *gridSyncState;
};

#endif
