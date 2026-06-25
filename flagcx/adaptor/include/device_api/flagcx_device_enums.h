/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API Enums — Scalar enum types for IR/Triton integration.
 *
 * These enums encode cooperative group kind and team kind as plain
 * integers, enabling LLVM IR callers (e.g. Triton) to express these
 * concepts without instantiating C++ structs.
 *
 * Safe to include from: CUDA device code, host code, LLVM bitcode builds.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_ENUMS_H_
#define FLAGCX_DEVICE_ENUMS_H_

#include <stdint.h>

/* ================================================================
 * Cooperative Group Kind
 *
 * Identifies the cooperation scope for collective device operations.
 * Used by the scalar IR API in place of struct-based cooperative groups.
 * ================================================================ */
typedef enum {
  FLAGCX_COOP_THREAD = 0,    /* Single thread (no cooperation) */
  FLAGCX_COOP_WARP = 1,      /* Full warp (FLAGCX_SIMT_WIDTH threads) */
  FLAGCX_COOP_BLOCK = 2,     /* Entire CTA */
  FLAGCX_COOP_TILE_SPAN = 3, /* Consecutive tile span (needs t0, nTiles, id) */
  FLAGCX_COOP_LANES = 4,     /* Arbitrary lane bitmask */
} flagcxCoopKind_t;

/* ================================================================
 * Team Kind
 *
 * Identifies the team scope within a communicator.
 * Used by the scalar IR API in place of struct-based teams.
 * ================================================================ */
typedef enum {
  FLAGCX_TEAM_INTRA = 0, /* Intra-node ranks */
  FLAGCX_TEAM_INTER = 1, /* Inter-node representatives */
  FLAGCX_TEAM_WORLD = 2, /* All ranks */
} flagcxTeamKind_t;

#endif /* FLAGCX_DEVICE_ENUMS_H_ */
