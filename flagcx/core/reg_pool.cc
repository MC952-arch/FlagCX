#include "reg_pool.h"
#include "p2p.h"
#include "proxy.h"
#include <cstdio>
#include <cstdlib>

flagcxRegPool::flagcxRegPool() { pageSize = sysconf(_SC_PAGESIZE); }

flagcxRegPool::~flagcxRegPool() {
  regMap.clear();
  regPool.clear();
}

inline void flagcxRegPool::getPagedAddr(void *data, size_t length,
                                        uintptr_t *beginAddr,
                                        uintptr_t *endAddr) {
  *beginAddr = reinterpret_cast<uintptr_t>(data) & -pageSize;
  *endAddr =
      (reinterpret_cast<uintptr_t>(data) + length + pageSize - 1) & -pageSize;
}

flagcxResult_t
flagcxRegPool::addNetHandle(void *comm, flagcxRegItem *reg, void *handle,
                            struct flagcxProxyConnector *proxyConn) {
  if (reg == nullptr) {
    return flagcxSuccess;
  }
  for (auto &handlePair : reg->handles) {
    if (handlePair.first.proxyConn == proxyConn) {
      handlePair.first.handle = handle;
      return flagcxSuccess;
    }
  }
  flagcxRegNetHandle netHandle{handle, proxyConn, comm};
  flagcxRegP2pHandle p2pHandle{nullptr, nullptr, nullptr};
  reg->handles.push_back(std::make_pair(netHandle, p2pHandle));

  return flagcxSuccess;
}

flagcxResult_t
flagcxRegPool::addP2pHandle(void *comm, flagcxRegItem *reg, void *handle,
                            struct flagcxProxyConnector *proxyConn) {
  if (reg == nullptr) {
    return flagcxSuccess;
  }
  for (auto &handlePair : reg->handles) {
    if (handlePair.second.proxyConn == proxyConn) {
      handlePair.second.handle = handle;
      return flagcxSuccess;
    }
  }
  flagcxRegNetHandle netHandle{nullptr, nullptr, nullptr};
  flagcxRegP2pHandle p2pHandle{handle, proxyConn, comm};
  reg->handles.push_back(std::make_pair(netHandle, p2pHandle));

  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::removeRegItemNetHandles(void *comm,
                                                      flagcxRegItem *reg) {
  if (comm == nullptr || reg == nullptr) {
    return flagcxSuccess;
  }

  for (size_t i = 0; i < reg->handles.size();) {
    auto &entry = reg->handles[i];
    if (entry.first.handle) {
      FLAGCXCHECK(flagcxNetDeregisterBuffer(
          entry.first.ownerComm, entry.first.proxyConn, entry.first.handle));
      entry.first.handle = nullptr;
      entry.first.proxyConn = nullptr;
      entry.first.ownerComm = nullptr;
    }
    if (entry.first.handle == nullptr && entry.second.handle == nullptr) {
      reg->handles[i] = reg->handles.back();
      reg->handles.pop_back();
    } else {
      ++i;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::removeRegItemP2pHandles(void *comm,
                                                      flagcxRegItem *reg) {
  if (comm == nullptr || reg == nullptr) {
    return flagcxSuccess;
  }

  for (size_t i = 0; i < reg->handles.size();) {
    auto &entry = reg->handles[i];
    if (entry.second.handle) {
      flagcxIpcRegInfo *ipcInfo = (flagcxIpcRegInfo *)entry.second.handle;
      FLAGCXCHECK(flagcxP2pDeregisterBuffer(
          reinterpret_cast<flagcxHeteroComm *>(entry.second.ownerComm),
          ipcInfo));
      entry.second.handle = nullptr;
      entry.second.proxyConn = nullptr;
      entry.second.ownerComm = nullptr;
    }
    if (entry.first.handle == nullptr && entry.second.handle == nullptr) {
      reg->handles[i] = reg->handles.back();
      reg->handles.pop_back();
    } else {
      ++i;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::removeAllP2pHandles(void *comm) {
  if (comm == nullptr) {
    return flagcxSuccess;
  }
  // Iterate over all items in the global pool and remove p2p handles
  // associated with this comm
  auto &globalPool = regPool[GLOBAL_POOL_KEY];
  for (auto &pair : globalPool) {
    FLAGCXCHECK(removeRegItemP2pHandles(comm, pair.second.get()));
  }
  return flagcxSuccess;
}

void flagcxRegPool::mapRegItemPages(uintptr_t commKey, flagcxRegItem *reg) {
  if (reg == nullptr) {
    return;
  }
  auto &regCommMap = regMap[commKey];
  for (uintptr_t addr = reg->beginAddr; addr < reg->endAddr; addr += pageSize) {
    regCommMap[addr] = reg;
  }
}

flagcxResult_t flagcxRegPool::registerBuffer(void *comm, void *data,
                                             size_t length) {
  if (data == nullptr || length == 0)
    return flagcxSuccess;

  uintptr_t commKey =
      comm ? reinterpret_cast<uintptr_t>(comm) : GLOBAL_POOL_KEY;
  uintptr_t beginAddr, endAddr;
  getPagedAddr(data, length, &beginAddr, &endAddr);

  // Always check/insert into the global pool (single source of truth)
  auto &globalPool = regPool[GLOBAL_POOL_KEY];
  auto it = globalPool.find(beginAddr);
  if (it != globalPool.end()) {
    // Already registered: bump refCount
    it->second->refCount++;
    // If comm is non-null, ensure it's mapped in the comm-specific regMap
    if (comm != nullptr) {
      mapRegItemPages(commKey, it->second.get());
    }
    return flagcxSuccess;
  }

  // Not found: create new item in global pool
  auto reg = std::make_unique<flagcxRegItem>();
  reg->beginAddr = beginAddr;
  reg->endAddr = endAddr;
  reg->refCount = 1;
  auto [it2, didInsert] = globalPool.emplace(beginAddr, std::move(reg));
  flagcxRegItem *regPtr = it2->second.get();

  // Map pages in global regMap
  mapRegItemPages(GLOBAL_POOL_KEY, regPtr);

  // If comm is non-null, also map pages in comm-specific regMap
  if (comm != nullptr) {
    mapRegItemPages(commKey, regPtr);
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::deregisterBuffer(void *comm, void *handle) {
  if (handle == nullptr) {
    return flagcxSuccess;
  }

  uintptr_t commKey =
      comm ? reinterpret_cast<uintptr_t>(comm) : GLOBAL_POOL_KEY;
  flagcxRegItem *reg = (flagcxRegItem *)handle;

  // Find the item in the global pool
  auto &globalPool = regPool[GLOBAL_POOL_KEY];
  auto poolIt = globalPool.find(reg->beginAddr);
  if (poolIt == globalPool.end() || poolIt->second.get() != reg) {
    WARN("Could not find the given handle in regPool");
    return flagcxInvalidUsage;
  }

  reg->refCount--;

  // Remove comm-specific page mappings
  if (comm != nullptr && commKey != GLOBAL_POOL_KEY) {
    auto mapIt = regMap.find(commKey);
    if (mapIt != regMap.end()) {
      auto &commMap = mapIt->second;
      for (uintptr_t addr = reg->beginAddr; addr < reg->endAddr;
           addr += pageSize) {
        commMap.erase(addr);
      }
      if (commMap.empty()) {
        regMap.erase(mapIt);
      }
    }
  }

  if (reg->refCount > 0) {
    return flagcxSuccess;
  }

  // refCount == 0: full cleanup
  FLAGCXCHECK(removeRegItemNetHandles(comm, reg));
  FLAGCXCHECK(removeRegItemP2pHandles(comm, reg));

  // Remove from global regMap
  auto globalMapIt = regMap.find(GLOBAL_POOL_KEY);
  if (globalMapIt != regMap.end()) {
    auto &globalMap = globalMapIt->second;
    for (uintptr_t addr = reg->beginAddr; addr < reg->endAddr;
         addr += pageSize) {
      globalMap.erase(addr);
    }
    if (globalMap.empty()) {
      regMap.erase(globalMapIt);
    }
  }

  // Remove from global pool (this destroys the flagcxRegItem)
  globalPool.erase(poolIt);
  return flagcxSuccess;
}

std::unordered_map<uintptr_t, std::unordered_map<uintptr_t, flagcxRegItem *>> &
flagcxRegPool::getGlobalMap() {
  return regMap;
}

flagcxRegItem *flagcxRegPool::getItem(const void *comm, void *data) {
  uintptr_t beginAddr, endAddr;
  getPagedAddr(data, 0, &beginAddr, &endAddr);

  // If comm is non-null, check comm-specific regMap first
  if (comm != nullptr) {
    uintptr_t commKey = reinterpret_cast<uintptr_t>(comm);
    auto mapIt = regMap.find(commKey);
    if (mapIt != regMap.end()) {
      auto it = mapIt->second.find(beginAddr);
      if (it != mapIt->second.end()) {
        return it->second;
      }
    }
  }

  // Fall through to global pool
  auto globalMapIt = regMap.find(GLOBAL_POOL_KEY);
  if (globalMapIt != regMap.end()) {
    auto it = globalMapIt->second.find(beginAddr);
    if (it != globalMapIt->second.end()) {
      return it->second;
    }
  }

  return nullptr;
}

void flagcxRegPool::dump() {
  printf("========================\n");
  printf("RegPool(pageSize=%lu\n", pageSize);
  for (auto &c : regMap) {
    printf("==comm(%lu)==\n", c.first);
    for (auto &p : c.second) {
      printf("beginAddr(%lu) -> regItem[%lu,%lu,%d]\n", p.first,
             p.second->beginAddr, p.second->endAddr, p.second->refCount);
      for (auto &h : p.second->handles) {
        printf("handlePtr(%p) -> netHandle[%p,%p] p2pHandle[%p,%p]\n", &h,
               h.first.handle, h.first.proxyConn, h.second.handle,
               h.second.proxyConn);
      }
    }
    printf("==comm(%lu)==\n", c.first);
  }
  printf("========================\n");
}
