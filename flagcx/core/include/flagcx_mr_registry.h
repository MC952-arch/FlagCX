/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Unified MR Registry — shared address-range registry for all subsystems
 * (P2P Engine, Collective Proxy, RMA One-sided).
 *
 * Provides O(log n) containment lookup via sorted flat array + binary search,
 * with pthread_rwlock for concurrent readers on the data path.
 ************************************************************************/

#ifndef FLAGCX_MR_REGISTRY_H_
#define FLAGCX_MR_REGISTRY_H_

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "flagcx.h" // flagcxResult_t

#ifdef __cplusplus
extern "C" {
#endif

/* ───── Owner bitmask constants ───── */

#define FLAGCX_MR_OWNER_P2P 0x01
#define FLAGCX_MR_OWNER_COLL 0x02
#define FLAGCX_MR_OWNER_RMA 0x04

#define FLAGCX_MR_OWNER_COUNT 3
#define FLAGCX_MR_OWNER_IDX_P2P 0
#define FLAGCX_MR_OWNER_IDX_COLL 1
#define FLAGCX_MR_OWNER_IDX_RMA 2

/* ───── Subsystem extension structs ───── */

#define FLAGCX_MR_IPC_HANDLE_BYTES 64
#define FLAGCX_MR_DESC_SIZE 64

struct flagcxMrP2pExt {
  uint64_t mrId;
  bool hasIpc;
  uint32_t ipcHandleSize;
  char ipcHandle[FLAGCX_MR_IPC_HANDLE_BYTES];
  char descBuf[FLAGCX_MR_DESC_SIZE];
};

struct flagcxMrCollExt {
  void *proxyConn;
  int channelId;
};

struct flagcxMrRmaExt {
  int oneSideHandleIdx;
};

/* ───── Core entry ───── */

struct flagcxMrEntry {
  uintptr_t baseAddr;
  size_t size;
  int ibDevN;
  int ptrType;
  uint32_t ownerMask;
  void *mhandles[FLAGCX_MR_OWNER_COUNT]; /* per-subsystem adaptor mhandle */

  struct flagcxMrP2pExt *p2p;   /* NULL if !(ownerMask & P2P) */
  struct flagcxMrCollExt *coll; /* NULL if !(ownerMask & COLL) */
  struct flagcxMrRmaExt *rma;   /* NULL if !(ownerMask & RMA) */
};

/* ───── Registry container ───── */

struct flagcxMrRegistry {
  struct flagcxMrEntry *entries; /* sorted by baseAddr, contiguous */
  int count;
  int capacity;
  uint64_t nextId; /* monotonic ID generator for all subsystems */
  pthread_rwlock_t rwlock;
};

/* ───── Lifecycle ───── */

flagcxResult_t flagcxMrRegistryCreate(struct flagcxMrRegistry **reg);
flagcxResult_t flagcxMrRegistryDestroy(struct flagcxMrRegistry *reg);

/* ───── Registration (write-locks internally) ───── */

/*
 * Register a memory region or add ownership to an existing entry.
 *
 * - If exact (addr, size) match exists: adds ownerBit, stores mhandle/ext.
 * - If exact addr match but different size: returns flagcxInternalError.
 * - If overlap with a different entry: returns flagcxInternalError.
 * - If new: inserts into sorted array.
 *
 * ownerBit: one of FLAGCX_MR_OWNER_{P2P,COLL,RMA}
 * mhandle:  adaptor handle (stored in mhandles[ownerIdx])
 * ext:      subsystem extension struct pointer (ownership transferred to entry)
 * outId:    if non-NULL, returns a monotonic ID. For P2P, this is persisted
 *           in p2p->mrId and stable across repeated calls. For COLL/RMA,
 *           this is a one-shot assignment (not stored on the entry).
 */
flagcxResult_t flagcxMrRegistryRegister(struct flagcxMrRegistry *reg,
                                        uintptr_t addr, size_t size, int ibDevN,
                                        int ptrType, uint32_t ownerBit,
                                        void *mhandle, void *ext,
                                        uint64_t *outId);

/*
 * Remove one owner from entry identified by baseAddr.
 * If ownerMask becomes 0, removes entry from array.
 *
 * outEntry: if non-NULL, populated with common fields before removal.
 * outExt:   if non-NULL, returns the subsystem extension pointer (caller
 * frees).
 */
flagcxResult_t flagcxMrRegistryDeregister(struct flagcxMrRegistry *reg,
                                          uintptr_t addr, uint32_t ownerBit,
                                          struct flagcxMrEntry *outEntry,
                                          void **outExt);

/* ───── Lookup (read-locks internally) ───── */

/*
 * O(log n) containment lookup: find entry where baseAddr <= addr <
 * baseAddr+size. Returns flagcxSuccess if found, flagcxInternalError if not.
 */
flagcxResult_t flagcxMrRegistryLookup(struct flagcxMrRegistry *reg,
                                      uintptr_t addr,
                                      struct flagcxMrEntry *outEntry);

/*
 * O(log n) exact-match lookup by baseAddr.
 * Returns flagcxSuccess if found, flagcxInternalError if not.
 */
flagcxResult_t flagcxMrRegistryFindExact(struct flagcxMrRegistry *reg,
                                         uintptr_t addr,
                                         struct flagcxMrEntry *outEntry);

/*
 * O(n) lookup by P2P mrId. Only used in non-hot paths (PrepareDesc, MrDestroy).
 * Returns flagcxSuccess if found, flagcxInternalError if not.
 */
flagcxResult_t flagcxMrRegistryLookupById(struct flagcxMrRegistry *reg,
                                          uint64_t mrId,
                                          struct flagcxMrEntry *outEntry);

/*
 * O(n) lookup by mhandle pointer for a given owner index.
 * Useful for deregistration when only the handle is known.
 * Returns flagcxSuccess if found, flagcxInternalError if not.
 */
flagcxResult_t flagcxMrRegistryFindByHandle(struct flagcxMrRegistry *reg,
                                            int ownerIdx, void *mhandle,
                                            struct flagcxMrEntry *outEntry);

/* ───── Iteration (external locking) ───── */

flagcxResult_t flagcxMrRegistryRdLock(struct flagcxMrRegistry *reg);
flagcxResult_t flagcxMrRegistryRdUnlock(struct flagcxMrRegistry *reg);
flagcxResult_t flagcxMrRegistryWrLock(struct flagcxMrRegistry *reg);
flagcxResult_t flagcxMrRegistryWrUnlock(struct flagcxMrRegistry *reg);

/* Access entries directly (only valid while holding lock) */
int flagcxMrRegistryCount(struct flagcxMrRegistry *reg);
struct flagcxMrEntry *flagcxMrRegistryEntries(struct flagcxMrRegistry *reg);

/* ───── Global instance ───── */

extern struct flagcxMrRegistry *flagcxGlobalMrRegistry;

flagcxResult_t flagcxMrRegistryGlobalInit(void);
flagcxResult_t flagcxMrRegistryGlobalRelease(void);

#ifdef __cplusplus
}
#endif

#endif /* FLAGCX_MR_REGISTRY_H_ */
