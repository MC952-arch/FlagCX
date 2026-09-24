/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * GID candidate ranking shared by the verbs adaptors and unit tests.
 ************************************************************************/

#ifndef FLAGCX_IB_GID_H_
#define FLAGCX_IB_GID_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Values defined by enum ibv_gid_type. Keep this header independent of a
// particular verbs ABI so it can also be tested with FlagCX's embedded verbs
// declarations.
#define FLAGCX_IB_GID_TYPE_IB 0U
#define FLAGCX_IB_GID_TYPE_ROCE_V2 2U

enum flagcxIbAutoGidCandidateClass {
  flagcxIbGidNetworkRoutable = 0,
  flagcxIbGidNoNetworkRoutable = 1,
  flagcxIbGidNetworkPrivateV4 = 2,
  flagcxIbGidNetworkDegraded = 3,
  flagcxIbGidNoNetworkPrivateV4 = 4,
  flagcxIbGidNoNetworkDegraded = 5,
  flagcxIbGidFallbackNonzero = 6,
};

enum flagcxIbIpv4Scope {
  flagcxIbIpv4Routable = 0,
  flagcxIbIpv4Private = 1,
  flagcxIbIpv4LinkLocal = 2,
};

struct flagcxIbAutoGidCandidate {
  int gidIndex;
  uint32_t gidType;
  bool hasNetworkDevice;
  bool isIpv4Mapped;
  bool isLinkLocalIpv4;
  bool isLinkLocalIpv6;
  bool isOverlayNetwork;
  bool isPrivateIpv4;
  bool isNullGid;
  bool querySucceeded;
};

// Preserve the existing filtered GID selection whenever the user explicitly
// configures one of its selectors. The unconstrained extended-query ranking is
// only the default when none of those overrides is present.
static inline bool
flagcxIbShouldUseAutoGidSelection(bool addressFamilyConfigured,
                                  bool addressRangeConfigured,
                                  bool roceVersionConfigured) {
  return !addressFamilyConfigured && !addressRangeConfigured &&
         !roceVersionConfigured;
}

// ipv4 is in host byte order.
static inline enum flagcxIbIpv4Scope
flagcxIbClassifyIpv4Address(uint32_t ipv4) {
  const uint8_t octet1 = (ipv4 >> 24) & 0xff;
  const uint8_t octet2 = (ipv4 >> 16) & 0xff;
  if (octet1 == 169 && octet2 == 254)
    return flagcxIbIpv4LinkLocal;
  if (octet1 == 10 || (octet1 == 172 && octet2 >= 16 && octet2 <= 31) ||
      (octet1 == 192 && octet2 == 168) ||
      (octet1 == 100 && octet2 >= 64 && octet2 <= 127))
    return flagcxIbIpv4Private;
  return flagcxIbIpv4Routable;
}

struct flagcxIbAutoGidSelection {
  int gidIndex;
  enum flagcxIbAutoGidCandidateClass candidateClass;
};

static inline const char *flagcxIbAutoGidCandidateClassName(
    enum flagcxIbAutoGidCandidateClass candidateClass) {
  switch (candidateClass) {
    case flagcxIbGidNetworkRoutable:
      return "network-routable";
    case flagcxIbGidNoNetworkRoutable:
      return "no-network-routable";
    case flagcxIbGidNetworkPrivateV4:
      return "network-private-v4";
    case flagcxIbGidNetworkDegraded:
      return "network-degraded";
    case flagcxIbGidNoNetworkPrivateV4:
      return "no-network-private-v4";
    case flagcxIbGidNoNetworkDegraded:
      return "no-network-degraded";
    case flagcxIbGidFallbackNonzero:
      return "fallback-nonzero";
  }
  return "unknown";
}

static inline bool flagcxIbClassifyAutoGidCandidate(
    const struct flagcxIbAutoGidCandidate *candidate,
    enum flagcxIbAutoGidCandidateClass *candidateClass) {
  if (candidate == NULL || candidateClass == NULL ||
      !candidate->querySucceeded || candidate->gidIndex < 0 ||
      candidate->isNullGid)
    return false;

  if (candidate->gidType != FLAGCX_IB_GID_TYPE_ROCE_V2 &&
      candidate->gidType != FLAGCX_IB_GID_TYPE_IB)
    return false;

  const bool isRoceV2 = candidate->gidType == FLAGCX_IB_GID_TYPE_ROCE_V2;
  const bool isOverlay = isRoceV2 && candidate->isOverlayNetwork;
  const bool isPrivateV4 = isRoceV2 && !isOverlay && candidate->isIpv4Mapped &&
                           candidate->isPrivateIpv4;
  const bool isLinkLocal =
      isRoceV2 && ((candidate->isIpv4Mapped && candidate->isLinkLocalIpv4) ||
                   (!candidate->isIpv4Mapped && candidate->isLinkLocalIpv6));
  const bool isDegraded = isOverlay || isLinkLocal;

  if (candidate->hasNetworkDevice) {
    *candidateClass = isDegraded    ? flagcxIbGidNetworkDegraded
                      : isPrivateV4 ? flagcxIbGidNetworkPrivateV4
                                    : flagcxIbGidNetworkRoutable;
  } else {
    *candidateClass = isDegraded    ? flagcxIbGidNoNetworkDegraded
                      : isPrivateV4 ? flagcxIbGidNoNetworkPrivateV4
                                    : flagcxIbGidNoNetworkRoutable;
  }
  return true;
}

static inline bool flagcxIbSelectBestAutoGidCandidate(
    const struct flagcxIbAutoGidCandidate *candidates, size_t count,
    struct flagcxIbAutoGidSelection *selection) {
  if (candidates == NULL || selection == NULL)
    return false;

  bool foundPreferred = false;
  bool foundFallback = false;
  struct flagcxIbAutoGidSelection preferred = {-1, flagcxIbGidFallbackNonzero};
  int fallbackIndex = -1;

  for (size_t i = 0; i < count; ++i) {
    const struct flagcxIbAutoGidCandidate *candidate = candidates + i;
    enum flagcxIbAutoGidCandidateClass candidateClass;
    if (flagcxIbClassifyAutoGidCandidate(candidate, &candidateClass)) {
      if (!foundPreferred || candidateClass < preferred.candidateClass ||
          (candidateClass == preferred.candidateClass &&
           candidate->gidIndex < preferred.gidIndex)) {
        preferred.gidIndex = candidate->gidIndex;
        preferred.candidateClass = candidateClass;
        foundPreferred = true;
      }
    } else if (!foundFallback && candidate->querySucceeded &&
               !candidate->isNullGid && candidate->gidIndex >= 0) {
      fallbackIndex = candidate->gidIndex;
      foundFallback = true;
    }
  }

  if (foundPreferred) {
    *selection = preferred;
    return true;
  }
  if (foundFallback) {
    selection->gidIndex = fallbackIndex;
    selection->candidateClass = flagcxIbGidFallbackNonzero;
    return true;
  }
  return false;
}

#endif // FLAGCX_IB_GID_H_
