/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Unified MR Registry implementation.
 * Sorted flat array + binary search + pthread_rwlock.
 ************************************************************************/

#include "flagcx_mr_registry.h"

#include <stdlib.h>
#include <string.h>

#include "debug.h" // WARN, INFO

#define MR_REGISTRY_INITIAL_CAPACITY 16

/* ───── Global instance ───── */

struct flagcxMrRegistry *flagcxGlobalMrRegistry = NULL;

/* ───── Internal helpers ───── */

static inline int ownerBitToIdx(uint32_t ownerBit) {
  switch (ownerBit) {
    case FLAGCX_MR_OWNER_P2P:
      return FLAGCX_MR_OWNER_IDX_P2P;
    case FLAGCX_MR_OWNER_COLL:
      return FLAGCX_MR_OWNER_IDX_COLL;
    case FLAGCX_MR_OWNER_RMA:
      return FLAGCX_MR_OWNER_IDX_RMA;
    default:
      return -1;
  }
}

/*
 * Binary search: find the largest index i where entries[i].baseAddr <= addr.
 * Returns -1 if all entries have baseAddr > addr.
 */
static int bsearchContaining(const struct flagcxMrEntry *entries, int count,
                             uintptr_t addr) {
  int lo = 0, hi = count - 1, result = -1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (entries[mid].baseAddr <= addr) {
      result = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return result;
}

/*
 * Binary search: find exact match by baseAddr.
 * Returns index or -1 if not found.
 */
static int bsearchExact(const struct flagcxMrEntry *entries, int count,
                        uintptr_t addr) {
  int lo = 0, hi = count - 1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (entries[mid].baseAddr == addr)
      return mid;
    else if (entries[mid].baseAddr < addr)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return -1;
}

/*
 * Find insertion point: returns the index where a new entry with baseAddr=addr
 * should be inserted to maintain sorted order.
 */
static int findInsertionPoint(const struct flagcxMrEntry *entries, int count,
                              uintptr_t addr) {
  int lo = 0, hi = count;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (entries[mid].baseAddr < addr)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

static flagcxResult_t ensureCapacity(struct flagcxMrRegistry *reg) {
  if (reg->count < reg->capacity)
    return flagcxSuccess;

  int newCap =
      reg->capacity == 0 ? MR_REGISTRY_INITIAL_CAPACITY : reg->capacity * 2;
  struct flagcxMrEntry *newEntries = (struct flagcxMrEntry *)realloc(
      reg->entries, (size_t)newCap * sizeof(struct flagcxMrEntry));
  if (newEntries == NULL) {
    WARN("flagcxMrRegistry: realloc failed for capacity %d", newCap);
    return flagcxSystemError;
  }
  reg->entries = newEntries;
  reg->capacity = newCap;
  return flagcxSuccess;
}

static void freeEntryExtensions(struct flagcxMrEntry *entry) {
  if (entry->p2p) {
    free(entry->p2p);
    entry->p2p = NULL;
  }
  if (entry->coll) {
    free(entry->coll);
    entry->coll = NULL;
  }
  if (entry->rma) {
    free(entry->rma);
    entry->rma = NULL;
  }
}

/* ───── Lifecycle ───── */

flagcxResult_t flagcxMrRegistryCreate(struct flagcxMrRegistry **reg) {
  struct flagcxMrRegistry *r =
      (struct flagcxMrRegistry *)calloc(1, sizeof(struct flagcxMrRegistry));
  if (r == NULL)
    return flagcxSystemError;

  r->entries = NULL;
  r->count = 0;
  r->capacity = 0;
  r->nextId = 1;

  if (pthread_rwlock_init(&r->rwlock, NULL) != 0) {
    free(r);
    return flagcxSystemError;
  }

  *reg = r;
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryDestroy(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxSuccess;

  /* Free all extension structs */
  for (int i = 0; i < reg->count; i++) {
    freeEntryExtensions(&reg->entries[i]);
  }

  free(reg->entries);
  pthread_rwlock_destroy(&reg->rwlock);
  free(reg);
  return flagcxSuccess;
}

/* ───── Registration ───── */

flagcxResult_t flagcxMrRegistryRegister(struct flagcxMrRegistry *reg,
                                        uintptr_t addr, size_t size, int ibDevN,
                                        int ptrType, uint32_t ownerBit,
                                        void *mhandle, void *ext,
                                        uint64_t *outId) {
  if (reg == NULL || size == 0)
    return flagcxInternalError;

  int ownerIdx = ownerBitToIdx(ownerBit);
  if (ownerIdx < 0)
    return flagcxInternalError;

  pthread_rwlock_wrlock(&reg->rwlock);

  /* Check for exact baseAddr match */
  int exactIdx = bsearchExact(reg->entries, reg->count, addr);
  if (exactIdx >= 0) {
    struct flagcxMrEntry *existing = &reg->entries[exactIdx];

    /* Size mismatch from another owner is an error */
    if (existing->size != size) {
      WARN(
          "flagcxMrRegistry: addr 0x%lx size mismatch: existing %zu vs new %zu",
          (unsigned long)addr, existing->size, size);
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxInternalError;
    }

    /* Already owned by this subsystem */
    if (existing->ownerMask & ownerBit) {
      /* If mhandle changed (e.g., reconnect with new PD), update in place */
      if (existing->mhandles[ownerIdx] != mhandle && mhandle != NULL) {
        INFO(FLAGCX_REG,
             "MrRegistry: updating mhandle for addr 0x%lx owner 0x%x",
             (unsigned long)addr, ownerBit);
        existing->mhandles[ownerIdx] = mhandle;
        /* Only replace extension if caller provided a new one */
        if (ext != NULL) {
          switch (ownerBit) {
            case FLAGCX_MR_OWNER_P2P:
              free(existing->p2p);
              existing->p2p = (struct flagcxMrP2pExt *)ext;
              break;
            case FLAGCX_MR_OWNER_COLL:
              free(existing->coll);
              existing->coll = (struct flagcxMrCollExt *)ext;
              break;
            case FLAGCX_MR_OWNER_RMA:
              free(existing->rma);
              existing->rma = (struct flagcxMrRmaExt *)ext;
              break;
          }
        }
      }
      if (outId != NULL && ownerBit == FLAGCX_MR_OWNER_P2P && existing->p2p)
        *outId = existing->p2p->mrId;
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxSuccess;
    }

    /* Add new owner */
    existing->ownerMask |= ownerBit;
    existing->mhandles[ownerIdx] = mhandle;

    switch (ownerBit) {
      case FLAGCX_MR_OWNER_P2P:
        existing->p2p = (struct flagcxMrP2pExt *)ext;
        if (existing->p2p && outId)
          *outId = existing->p2p->mrId;
        break;
      case FLAGCX_MR_OWNER_COLL:
        existing->coll = (struct flagcxMrCollExt *)ext;
        break;
      case FLAGCX_MR_OWNER_RMA:
        existing->rma = (struct flagcxMrRmaExt *)ext;
        break;
    }

    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxSuccess;
  }

  /* No exact match — check for overlap at insertion point */
  int pos = findInsertionPoint(reg->entries, reg->count, addr);

  /* Check left neighbor overlap: entries[pos-1].baseAddr + size > addr */
  if (pos > 0) {
    struct flagcxMrEntry *left = &reg->entries[pos - 1];
    if (left->baseAddr + left->size > addr) {
      WARN("flagcxMrRegistry: overlap with left neighbor [0x%lx, +%zu) vs new "
           "[0x%lx, +%zu)",
           (unsigned long)left->baseAddr, left->size, (unsigned long)addr,
           size);
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxInternalError;
    }
  }

  /* Check right neighbor overlap: addr + size > entries[pos].baseAddr */
  if (pos < reg->count) {
    struct flagcxMrEntry *right = &reg->entries[pos];
    if (addr + size > right->baseAddr) {
      WARN("flagcxMrRegistry: overlap with right neighbor [0x%lx, +%zu) vs new "
           "[0x%lx, +%zu)",
           (unsigned long)right->baseAddr, right->size, (unsigned long)addr,
           size);
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxInternalError;
    }
  }

  /* Grow array if needed */
  flagcxResult_t res = ensureCapacity(reg);
  if (res != flagcxSuccess) {
    pthread_rwlock_unlock(&reg->rwlock);
    return res;
  }

  /* Shift entries right to make room */
  if (pos < reg->count) {
    memmove(&reg->entries[pos + 1], &reg->entries[pos],
            (size_t)(reg->count - pos) * sizeof(struct flagcxMrEntry));
  }

  /* Initialize new entry */
  struct flagcxMrEntry *entry = &reg->entries[pos];
  memset(entry, 0, sizeof(struct flagcxMrEntry));
  entry->baseAddr = addr;
  entry->size = size;
  entry->ibDevN = ibDevN;
  entry->ptrType = ptrType;
  entry->ownerMask = ownerBit;
  entry->mhandles[ownerIdx] = mhandle;

  switch (ownerBit) {
    case FLAGCX_MR_OWNER_P2P:
      entry->p2p = (struct flagcxMrP2pExt *)ext;
      if (entry->p2p) {
        /* Assign mrId from registry's monotonic counter if not pre-set */
        if (entry->p2p->mrId == 0)
          entry->p2p->mrId = reg->nextId++;
        if (outId)
          *outId = entry->p2p->mrId;
      }
      break;
    case FLAGCX_MR_OWNER_COLL:
      entry->coll = (struct flagcxMrCollExt *)ext;
      break;
    case FLAGCX_MR_OWNER_RMA:
      entry->rma = (struct flagcxMrRmaExt *)ext;
      break;
  }

  reg->count++;
  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryDeregister(struct flagcxMrRegistry *reg,
                                          uintptr_t addr, uint32_t ownerBit,
                                          struct flagcxMrEntry *outEntry,
                                          void **outExt) {
  if (reg == NULL)
    return flagcxInternalError;

  int ownerIdx = ownerBitToIdx(ownerBit);
  if (ownerIdx < 0)
    return flagcxInternalError;

  pthread_rwlock_wrlock(&reg->rwlock);

  int idx = bsearchExact(reg->entries, reg->count, addr);
  if (idx < 0) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  struct flagcxMrEntry *entry = &reg->entries[idx];

  if (!(entry->ownerMask & ownerBit)) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  /* Copy out before modification */
  if (outEntry)
    *outEntry = *entry;

  /* Extract subsystem extension */
  void *ext = NULL;
  switch (ownerBit) {
    case FLAGCX_MR_OWNER_P2P:
      ext = entry->p2p;
      entry->p2p = NULL;
      break;
    case FLAGCX_MR_OWNER_COLL:
      ext = entry->coll;
      entry->coll = NULL;
      break;
    case FLAGCX_MR_OWNER_RMA:
      ext = entry->rma;
      entry->rma = NULL;
      break;
  }
  if (outExt)
    *outExt = ext;

  entry->ownerMask &= ~ownerBit;
  entry->mhandles[ownerIdx] = NULL;

  /* If no owners remain, remove entry from array */
  if (entry->ownerMask == 0) {
    freeEntryExtensions(entry);
    if (idx < reg->count - 1) {
      memmove(&reg->entries[idx], &reg->entries[idx + 1],
              (size_t)(reg->count - 1 - idx) * sizeof(struct flagcxMrEntry));
    }
    reg->count--;
  }

  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxSuccess;
}

/* ───── Lookup ───── */

flagcxResult_t flagcxMrRegistryLookup(struct flagcxMrRegistry *reg,
                                      uintptr_t addr,
                                      struct flagcxMrEntry *outEntry) {
  if (reg == NULL || outEntry == NULL)
    return flagcxInternalError;

  pthread_rwlock_rdlock(&reg->rwlock);

  if (reg->count == 0) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  int idx = bsearchContaining(reg->entries, reg->count, addr);
  if (idx < 0) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  struct flagcxMrEntry *entry = &reg->entries[idx];
  if (addr >= entry->baseAddr && addr < entry->baseAddr + entry->size) {
    *outEntry = *entry;
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxSuccess;
  }

  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxInternalError;
}

flagcxResult_t flagcxMrRegistryFindExact(struct flagcxMrRegistry *reg,
                                         uintptr_t addr,
                                         struct flagcxMrEntry *outEntry) {
  if (reg == NULL || outEntry == NULL)
    return flagcxInternalError;

  pthread_rwlock_rdlock(&reg->rwlock);

  int idx = bsearchExact(reg->entries, reg->count, addr);
  if (idx < 0) {
    pthread_rwlock_unlock(&reg->rwlock);
    return flagcxInternalError;
  }

  *outEntry = reg->entries[idx];
  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryLookupById(struct flagcxMrRegistry *reg,
                                          uint64_t mrId,
                                          struct flagcxMrEntry *outEntry) {
  if (reg == NULL || outEntry == NULL)
    return flagcxInternalError;

  pthread_rwlock_rdlock(&reg->rwlock);

  for (int i = 0; i < reg->count; i++) {
    if (reg->entries[i].p2p && reg->entries[i].p2p->mrId == mrId) {
      *outEntry = reg->entries[i];
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxSuccess;
    }
  }

  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxInternalError;
}

flagcxResult_t flagcxMrRegistryFindByHandle(struct flagcxMrRegistry *reg,
                                            int ownerIdx, void *mhandle,
                                            struct flagcxMrEntry *outEntry) {
  if (reg == NULL || outEntry == NULL || mhandle == NULL)
    return flagcxInternalError;
  if (ownerIdx < 0 || ownerIdx >= FLAGCX_MR_OWNER_COUNT)
    return flagcxInternalError;

  pthread_rwlock_rdlock(&reg->rwlock);

  for (int i = 0; i < reg->count; i++) {
    if (reg->entries[i].mhandles[ownerIdx] == mhandle) {
      *outEntry = reg->entries[i];
      pthread_rwlock_unlock(&reg->rwlock);
      return flagcxSuccess;
    }
  }

  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxInternalError;
}

/* ───── Iteration support ───── */

flagcxResult_t flagcxMrRegistryRdLock(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxInternalError;
  pthread_rwlock_rdlock(&reg->rwlock);
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryRdUnlock(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxInternalError;
  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryWrLock(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxInternalError;
  pthread_rwlock_wrlock(&reg->rwlock);
  return flagcxSuccess;
}

flagcxResult_t flagcxMrRegistryWrUnlock(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return flagcxInternalError;
  pthread_rwlock_unlock(&reg->rwlock);
  return flagcxSuccess;
}

int flagcxMrRegistryCount(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return 0;
  return reg->count;
}

struct flagcxMrEntry *flagcxMrRegistryEntries(struct flagcxMrRegistry *reg) {
  if (reg == NULL)
    return NULL;
  return reg->entries;
}

/* ───── Global instance management ───── */

static pthread_once_t gRegistryOnce = PTHREAD_ONCE_INIT;
static flagcxResult_t gRegistryInitResult = flagcxSuccess;
static int gRegistryRefCount = 0;

static void flagcxMrRegistryGlobalInitOnce(void) {
  gRegistryInitResult = flagcxMrRegistryCreate(&flagcxGlobalMrRegistry);
}

flagcxResult_t flagcxMrRegistryGlobalInit(void) {
  pthread_once(&gRegistryOnce, flagcxMrRegistryGlobalInitOnce);
  __atomic_add_fetch(&gRegistryRefCount, 1, __ATOMIC_RELAXED);
  return gRegistryInitResult;
}

flagcxResult_t flagcxMrRegistryGlobalDestroy(void) {
  if (flagcxGlobalMrRegistry == NULL)
    return flagcxSuccess;
  flagcxResult_t res = flagcxMrRegistryDestroy(flagcxGlobalMrRegistry);
  flagcxGlobalMrRegistry = NULL;
  return res;
}

flagcxResult_t flagcxMrRegistryGlobalRelease(void) {
  int prev = __atomic_load_n(&gRegistryRefCount, __ATOMIC_ACQUIRE);
  if (prev <= 0)
    return flagcxInternalError;
  if (__atomic_sub_fetch(&gRegistryRefCount, 1, __ATOMIC_ACQ_REL) == 0) {
    return flagcxMrRegistryGlobalDestroy();
  }
  return flagcxSuccess;
}
