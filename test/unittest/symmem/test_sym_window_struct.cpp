// Unit tests for symmetric memory structs and parameter validation.
// No MPI or GPU required — runs locally.
// Links against libflagcx.

#include <gtest/gtest.h>

#include "adaptor.h"
#include "flagcx.h"
#include "flagcx_net_adaptor.h"
#include "mem_alloc_registry.h"
#include "onesided.h"
#include "sym_heap.h"
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>

namespace {

int unmapCalls = 0;
int vaFreeCalls = 0;
int physFreeCalls = 0;
int multicastTeardownCalls = 0;
int multicastFreeCalls = 0;
int cleanupStep = 0;
int multicastFreeStep = 0;
int flatUnmapStep = 0;
int physFreeStep = 0;
int vaRegCalls = 0;
int dmaBufRegCalls = 0;
int deregMrCalls = 0;
int dmaSupportCalls = 0;
int allocationCapsCalls = 0;
uint32_t mockAllocationCaps = FLAGCX_VMM_MR_CAP_NONE;
int dmaBufRegPtrType = -1;
uint64_t dmaBufRegOffset = UINT64_MAX;
void *dmaBufExportBase = nullptr;
size_t dmaBufExportSize = 0;
void *mockRangeBase = nullptr;
size_t mockRangeSize = 0;
int multicastProviderRefs = 0;
int multicastObjectDestroyCalls = 0;

flagcxResult_t mockGetAllocationVmmMrCaps(const void *, uint32_t *caps) {
  if (caps == nullptr)
    return flagcxInvalidArgument;
  allocationCapsCalls++;
  *caps = mockAllocationCaps;
  return flagcxSuccess;
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *old = std::getenv(name);
    if (old != nullptr) {
      hadOldValue_ = true;
      oldValue_ = old;
    }
    if (value != nullptr)
      setenv(name, value, 1);
    else
      unsetenv(name);
  }

  ~ScopedEnvVar() {
    if (hadOldValue_)
      setenv(name_, oldValue_.c_str(), 1);
    else
      unsetenv(name_);
  }

private:
  const char *name_;
  bool hadOldValue_ = false;
  std::string oldValue_;
};

flagcxResult_t failUnmapOnce(void *, size_t, int) {
  unmapCalls++;
  return unmapCalls == 1 ? flagcxRemoteError : flagcxSuccess;
}

flagcxResult_t failVaFreeOnce(void *, size_t, int) {
  vaFreeCalls++;
  return vaFreeCalls == 1 ? flagcxRemoteError : flagcxSuccess;
}

flagcxResult_t countFlatVaFree(void *, size_t, int) {
  vaFreeCalls++;
  return flagcxSuccess;
}

flagcxResult_t countMulticastVaFree(void *, size_t) {
  vaFreeCalls++;
  return flagcxSuccess;
}

flagcxResult_t countPhysFree(void *) {
  physFreeCalls++;
  physFreeStep = ++cleanupStep;
  return flagcxSuccess;
}

flagcxResult_t failMulticastTeardownOnce(void *, size_t) {
  multicastTeardownCalls++;
  if (multicastTeardownCalls == 1)
    return flagcxRemoteError;
  ++cleanupStep;
  return flagcxSuccess;
}

flagcxResult_t countMulticastFree(void *) {
  multicastFreeCalls++;
  multicastFreeStep = ++cleanupStep;
  return flagcxSuccess;
}

flagcxResult_t retainMockMulticastHandle(int, void **handle) {
  if (handle == nullptr)
    return flagcxInvalidArgument;
  *handle = malloc(1);
  if (*handle == nullptr)
    return flagcxSystemError;
  multicastProviderRefs++;
  return flagcxSuccess;
}

flagcxResult_t releaseMockMulticastHandle(void *handle) {
  if (handle == nullptr)
    return flagcxSuccess;
  free(handle);
  multicastProviderRefs--;
  if (multicastProviderRefs == 0)
    multicastObjectDestroyCalls++;
  return flagcxSuccess;
}

flagcxResult_t teardownMockMulticastMapping(void *, size_t) {
  multicastTeardownCalls++;
  return flagcxSuccess;
}

flagcxResult_t countFlatUnmap(void *, size_t, int) {
  unmapCalls++;
  flatUnmapStep = ++cleanupStep;
  return flagcxSuccess;
}

flagcxResult_t mockNetProperties(int, void *properties) {
  auto *props = static_cast<flagcxNetProperties_t *>(properties);
  memset(props, 0, sizeof(*props));
  props->ptrSupport = FLAGCX_PTR_CUDA | FLAGCX_PTR_DMABUF;
  return flagcxSuccess;
}

flagcxResult_t mockNetPropertiesUnsupported(int, void *) {
  return flagcxNotSupported;
}

flagcxResult_t mockNetPropertiesSystemError(int, void *) {
  return flagcxSystemError;
}

flagcxResult_t mockNetPropertiesDeviceError(int, void *) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t mockDmaSupport(bool *supported) {
  dmaSupportCalls++;
  if (supported == nullptr)
    return flagcxInvalidArgument;
  *supported = true;
  return flagcxSuccess;
}

flagcxResult_t mockDmaSupportFailure(bool *) {
  dmaSupportCalls++;
  return flagcxUnhandledDeviceError;
}

flagcxResult_t mockDmaBufExportUnsupported(void *, void *, size_t,
                                           unsigned long long) {
  return flagcxNotSupported;
}

flagcxResult_t mockDmaBufExportFailure(void *, void *, size_t,
                                       unsigned long long) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t mockDmaBufExportSuccess(void *handleOut, void *buffer,
                                       size_t size, unsigned long long) {
  if (handleOut == nullptr)
    return flagcxInvalidArgument;
  int fd = open("/dev/null", O_RDONLY);
  if (fd < 0)
    return flagcxSystemError;
  *static_cast<int *>(handleOut) = fd;
  dmaBufExportBase = buffer;
  dmaBufExportSize = size;
  return flagcxSuccess;
}

flagcxResult_t mockAddressRange(const void *ptr, void **base, size_t *size) {
  if (ptr == nullptr || base == nullptr || size == nullptr)
    return flagcxInvalidArgument;
  if (mockRangeBase != nullptr) {
    *base = mockRangeBase;
    *size = mockRangeSize;
    return flagcxSuccess;
  }
  long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0)
    return flagcxSystemError;
  uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  *base = reinterpret_cast<void *>(address - address % pageSize);
  *size = static_cast<size_t>(pageSize) * 2;
  return flagcxSuccess;
}

flagcxResult_t mockVaRegMr(void *, void *, size_t, int, int, void **handle) {
  vaRegCalls++;
  if (handle != nullptr)
    *handle = reinterpret_cast<void *>(0x1234);
  return flagcxSuccess;
}

flagcxResult_t mockVaRegMrFailure(void *, void *, size_t, int, int,
                                  void **handle) {
  vaRegCalls++;
  if (handle != nullptr)
    *handle = nullptr;
  return flagcxSystemError;
}

flagcxResult_t mockVaRegMrSuccessWithoutHandle(void *, void *, size_t, int, int,
                                               void **handle) {
  vaRegCalls++;
  if (handle != nullptr)
    *handle = nullptr;
  return flagcxSuccess;
}

flagcxResult_t mockDmaBufRegMr(void *, void *, size_t, int ptrType,
                               uint64_t offset, int, int, void **handle) {
  dmaBufRegCalls++;
  dmaBufRegPtrType = ptrType;
  dmaBufRegOffset = offset;
  if (handle != nullptr)
    *handle = reinterpret_cast<void *>(0x5678);
  return flagcxSuccess;
}

flagcxResult_t mockDmaBufRegMrUnsupported(void *, void *, size_t, int, uint64_t,
                                          int, int, void **handle) {
  dmaBufRegCalls++;
  if (handle != nullptr)
    *handle = nullptr;
  return flagcxNotSupported;
}

flagcxResult_t mockDmaBufRegMrPartialUnsupported(void *, void *, size_t, int,
                                                 uint64_t, int, int,
                                                 void **handle) {
  dmaBufRegCalls++;
  if (handle != nullptr)
    *handle = reinterpret_cast<void *>(0x5678);
  return flagcxNotSupported;
}

flagcxResult_t mockDeregMr(void *, void *handle) {
  deregMrCalls++;
  return handle == reinterpret_cast<void *>(0x5678) ? flagcxSuccess
                                                    : flagcxInvalidArgument;
}

flagcxResult_t mockDeregMrFailure(void *, void *) {
  deregMrCalls++;
  return flagcxRemoteError;
}

flagcxResult_t mockDmaBufRegMrFailure(void *, void *, size_t, int, uint64_t,
                                      int, int, void **handle) {
  dmaBufRegCalls++;
  if (handle != nullptr)
    *handle = nullptr;
  return flagcxSystemError;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Struct layout tests — verify fields exist and have expected types
// ---------------------------------------------------------------------------

TEST(SymWindowStruct, WindowStructLayout) {
  struct flagcxWindow win;
  memset(&win, 0, sizeof(win));

  // Verify fields exist and are accessible
  win.vendorBase = nullptr;
  win.defaultBase = nullptr;
  win.isSymmetricDefault = 0;

  EXPECT_EQ(win.vendorBase, nullptr);
  EXPECT_EQ(win.defaultBase, nullptr);
  EXPECT_EQ(win.isSymmetricDefault, 0);
}

TEST(SymWindowStruct, SymWindowStructLayout) {
  struct flagcxSymWindow sw;
  memset(&sw, 0, sizeof(sw));

  // Verify all fields exist
  sw.flatBase = nullptr;
  sw.mcBase = nullptr;
  sw.mrIndex = -1;
  sw.mrBase = 0;
  sw.heapSize = 1024;
  sw.allocSize = 2048;
  sw.localRanks = 2;
  sw.physHandle = nullptr;
  sw.mcHandle = nullptr;
  sw.allocationIsVmm = true;
  sw.isVMM = false;

  EXPECT_EQ(sw.mrIndex, -1);
  EXPECT_EQ(sw.heapSize, 1024u);
  EXPECT_EQ(sw.allocSize, 2048u);
  EXPECT_EQ(sw.localRanks, 2);
  EXPECT_TRUE(sw.allocationIsVmm);
  EXPECT_FALSE(sw.isVMM);
}

// ---------------------------------------------------------------------------
// 2. Flag constants
// ---------------------------------------------------------------------------

TEST(SymWindowStruct, WindowFlagConstants) {
  EXPECT_EQ(FLAGCX_WIN_DEFAULT, 0x00);
  EXPECT_EQ(FLAGCX_WIN_COLL_SYMMETRIC, 0x01);
}

TEST(SymWindowMrRoute, PrefersDmaBufWhenEveryLayerSupportsIt) {
  EXPECT_EQ(flagcxOneSideSelectVmmMrRoute(FLAGCX_VMM_MR_CAP_DMABUF |
                                              FLAGCX_VMM_MR_CAP_VA,
                                          FLAGCX_PTR_DMABUF | FLAGCX_PTR_CUDA,
                                          /*dmaBufExportSupported=*/true,
                                          /*hasDmaBufRegistration=*/true,
                                          /*hasVaRegistration=*/true),
            FLAGCX_VMM_MR_ROUTE_DMABUF);
}

TEST(SymWindowMrRoute, FallsBackToVaWhenDmaBufExportIsUnavailable) {
  EXPECT_EQ(flagcxOneSideSelectVmmMrRoute(FLAGCX_VMM_MR_CAP_DMABUF |
                                              FLAGCX_VMM_MR_CAP_VA,
                                          FLAGCX_PTR_DMABUF | FLAGCX_PTR_CUDA,
                                          /*dmaBufExportSupported=*/false,
                                          /*hasDmaBufRegistration=*/true,
                                          /*hasVaRegistration=*/true),
            FLAGCX_VMM_MR_ROUTE_VA);
}

TEST(SymWindowMrRoute, FallsBackToVaWhenProviderHasNoDmaBufEntryPoint) {
  EXPECT_EQ(flagcxOneSideSelectVmmMrRoute(FLAGCX_VMM_MR_CAP_DMABUF |
                                              FLAGCX_VMM_MR_CAP_VA,
                                          FLAGCX_PTR_DMABUF | FLAGCX_PTR_CUDA,
                                          /*dmaBufExportSupported=*/true,
                                          /*hasDmaBufRegistration=*/false,
                                          /*hasVaRegistration=*/true),
            FLAGCX_VMM_MR_ROUTE_VA);
}

TEST(SymWindowMrRoute, RejectsVmmWhenNeitherRouteIsUsable) {
  EXPECT_EQ(flagcxOneSideSelectVmmMrRoute(FLAGCX_VMM_MR_CAP_DMABUF,
                                          FLAGCX_PTR_DMABUF,
                                          /*dmaBufExportSupported=*/false,
                                          /*hasDmaBufRegistration=*/true,
                                          /*hasVaRegistration=*/true),
            FLAGCX_VMM_MR_ROUTE_NONE);
}

TEST(SymWindowMrRoute, ParsesStrictRouteModes) {
  flagcxVmmMrMode_t mode = flagcxVmmMrModeDmaBuf;
  EXPECT_EQ(flagcxOneSideParseVmmMrMode(nullptr, &mode), flagcxSuccess);
  EXPECT_EQ(mode, flagcxVmmMrModeAuto);
  EXPECT_EQ(flagcxOneSideParseVmmMrMode("AUTO", &mode), flagcxSuccess);
  EXPECT_EQ(mode, flagcxVmmMrModeAuto);
  EXPECT_EQ(flagcxOneSideParseVmmMrMode("dmabuf", &mode), flagcxSuccess);
  EXPECT_EQ(mode, flagcxVmmMrModeDmaBuf);
  EXPECT_EQ(flagcxOneSideParseVmmMrMode("VA", &mode), flagcxSuccess);
  EXPECT_EQ(mode, flagcxVmmMrModeVa);
  EXPECT_EQ(flagcxOneSideParseVmmMrMode("fallback", &mode),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxOneSideParseVmmMrMode("auto", nullptr),
            flagcxInvalidArgument);
}

TEST(SymWindowMrRoute, PreservesNetPropertyQueryResults) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "auto");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  deviceAdaptor = &testDevice;

  using GetPropertiesFn = flagcxResult_t (*)(int, void *);
  struct TestCase {
    const char *name;
    GetPropertiesFn getProperties;
    flagcxResult_t expected;
  };
  const TestCase cases[] = {
      {"missing", nullptr, flagcxNotSupported},
      {"unsupported", mockNetPropertiesUnsupported, flagcxNotSupported},
      {"system-error", mockNetPropertiesSystemError, flagcxSystemError},
      {"device-error", mockNetPropertiesDeviceError,
       flagcxUnhandledDeviceError},
  };

  char buffer[64] = {};
  for (const auto &testCase : cases) {
    SCOPED_TRACE(testCase.name);
    struct flagcxNetAdaptor testNet = {};
    struct flagcxHeteroComm comm = {};
    testNet.getProperties = testCase.getProperties;
    testNet.regMr = mockVaRegMr;
    testNet.regMrDmaBuf = mockDmaBufRegMr;
    comm.netAdaptor = &testNet;
    comm.netDev = 0;
    void *mrHandle = reinterpret_cast<void *>(0x1);
    flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_VA;
    vaRegCalls = 0;
    dmaBufRegCalls = 0;
    dmaSupportCalls = 0;

    EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                      buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                      /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                      &mrHandle, &route),
              testCase.expected);
    EXPECT_EQ(mrHandle, nullptr);
    EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
    EXPECT_EQ(vaRegCalls, 0);
    EXPECT_EQ(dmaBufRegCalls, 0);
    EXPECT_EQ(dmaSupportCalls, 0);
  }

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, StrictVaSkipsDmaBufProbeAndRegistration) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "va");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, AllocationCapabilityCanRejectUnsafeVaRoute) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "va");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = reinterpret_cast<void *>(0x1);
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_DMABUF;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.getAllocationVmmMrCaps = mockGetAllocationVmmMrCaps;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  mockAllocationCaps = FLAGCX_VMM_MR_CAP_DMABUF;
  allocationCapsCalls = 0;
  vaRegCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxNotSupported);
  EXPECT_EQ(allocationCapsCalls, 1);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, StrictDmaBufNeverFallsBackToVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "dmabuf");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrUnsupported;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxNotSupported);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, DmaBufProviderErrorDoesNotFallbackToVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "auto");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrFailure;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSystemError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, PartialDmaBufMrIsReleasedBeforeVaFallback) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "auto");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrPartialUnsupported;
  testNet.deregMr = mockDeregMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  deregMrCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(deregMrCalls, 1);
  EXPECT_EQ(vaRegCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, PartialDmaBufMrCleanupFailureSuppressesVaFallback) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "auto");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrPartialUnsupported;
  testNet.deregMr = mockDeregMrFailure;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  deregMrCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxRemoteError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(deregMrCalls, 1);
  EXPECT_EQ(vaRegCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, DmaBufSubrangeExportsAllocationAndUsesPageOffset) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "dmabuf");
  long pageSizeResult = sysconf(_SC_PAGESIZE);
  ASSERT_GT(pageSizeResult, 0);
  const size_t pageSize = static_cast<size_t>(pageSizeResult);
  void *allocation = nullptr;
  ASSERT_EQ(posix_memalign(&allocation, pageSize, pageSize * 3), 0);

  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  auto *subrange = static_cast<char *>(allocation) + pageSize + 123;
  constexpr size_t subrangeSize = 257;
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  dmaBufRegOffset = UINT64_MAX;
  dmaBufExportBase = nullptr;
  dmaBufExportSize = 0;
  mockRangeBase = allocation;
  mockRangeSize = pageSize * 3;
  flagcxMemAllocationInfo tracked = {allocation, pageSize * 2 + 333,
                                     flagcxMemCCL, flagcxMemAllocBackendNative,
                                     true};
  ASSERT_EQ(globalMemAllocRegistry.insert(tracked), flagcxSuccess);
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    subrange, subrangeSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(dmaBufExportBase, allocation);
  EXPECT_EQ(dmaBufExportSize, pageSize * 3);
  EXPECT_EQ(dmaBufRegOffset, pageSize);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);

  deviceAdaptor = savedDevice;
  EXPECT_EQ(globalMemAllocRegistry.erase(allocation), flagcxSuccess);
  mockRangeBase = nullptr;
  mockRangeSize = 0;
  free(allocation);
}

TEST(SymWindowMrRoute, VaProviderErrorIsPreserved) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "va");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMrFailure;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSystemError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, ProviderSuccessWithoutHandleIsInternalError) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "va");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = reinterpret_cast<void *>(0x1);
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_VA;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMrSuccessWithoutHandle;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxInternalError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, StrictDmaBufUsesDmaBufRoute) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "dmabuf");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, InvalidStrictModeFailsBeforeRegistration) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "invalid");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxInvalidArgument);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, ExportNotSupportedFallsBackToValidatedVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportUnsupported;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  dmaBufRegPtrType = -1;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 1);
  EXPECT_EQ(dmaBufRegPtrType, -1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, LegacyV1ExportSuccessUsesDmaBuf) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "invalid");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_NONE;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  dmaBufRegPtrType = -1;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 0);
  EXPECT_EQ(dmaBufRegPtrType, FLAGCX_PTR_CUDA);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, LegacyV1ExportFailureFallsBackToVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_NONE;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, LatestExportDeviceErrorDoesNotFallback) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = {};
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.getAllocationVmmMrCaps = nullptr;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrSlots, SelectsOnlySlotFreeOnEveryRank) {
  const uint8_t occupancy[] = {
      1, 0, 0, // rank 0 released slot 1
      1, 1, 0, // rank 1 retained slot 1 after deregMr failed
  };
  int slot = -1;
  EXPECT_EQ(flagcxOneSideSelectCommonPublishSlot(occupancy, 2, 3, &slot),
            flagcxSuccess);
  EXPECT_EQ(slot, 2);
}

TEST(SymWindowMrSlots, AppendsWhenNoCommonHoleExists) {
  const uint8_t occupancy[] = {
      1, 0, 1, // rank 0
      1, 1, 0, // rank 1
  };
  int slot = -1;
  EXPECT_EQ(flagcxOneSideSelectCommonPublishSlot(occupancy, 2, 3, &slot),
            flagcxSuccess);
  EXPECT_EQ(slot, 3);
}

TEST(SymWindowMrSlots, RejectsDivergentConnectionOwner) {
  const uint8_t occupancy[] = {
      0, 0, // rank 0
      1, 0, // rank 1
  };
  int slot = -1;
  EXPECT_EQ(flagcxOneSideSelectCommonPublishSlot(occupancy, 2, 2, &slot),
            flagcxInvalidUsage);
}

TEST(SymWindowRoutes, RequiresAUsablePathForEveryPeer) {
  char allocation[256] = {};
  uintptr_t baseVas[] = {reinterpret_cast<uintptr_t>(allocation), 0, 0, 0};
  size_t regionSizes[] = {sizeof(allocation), 0, 0, 0};
  struct flagcxOneSideHandleInfo handle = {};
  handle.baseVas = baseVas;
  handle.regionSizes = regionSizes;
  struct flagcxOneSideHandleInfo *handles[] = {&handle};

  struct flagcxHeteroComm comm = {};
  comm.rank = 0;
  comm.nRanks = 4;
  comm.localRanks = 2;
  comm.oneSideHandles = handles;
  comm.oneSideHandleCount = 1;

  struct flagcxSymWindow window = {};
  window.localBase = allocation;
  window.heapSize = sizeof(allocation);
  window.mrIndex = -1;
  window.ipcSlot = -1;

  // Flat mappings are node-local and cannot satisfy remote peers.
  window.isVMM = true;
  window.flatBase = reinterpret_cast<void *>(0x1000);
  EXPECT_EQ(flagcxSymWindowValidateDataRoutes(&comm, &window),
            flagcxNotSupported);

  window.hasNetworkMrRef = true;
  window.mrIndex = 0;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutes(&comm, &window), flagcxSuccess);

  // A local-only communicator can use flat or IPC mappings without an MR.
  comm.nRanks = 2;
  comm.localRanks = 2;
  window.hasNetworkMrRef = false;
  window.mrIndex = -1;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, true),
            flagcxSuccess);
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, false),
            flagcxNotSupported);
  window.isVMM = false;
  window.flatBase = nullptr;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, true),
            flagcxNotSupported);
  window.ipcSlot = 3;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, true),
            flagcxSuccess);
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, false),
            flagcxNotSupported);

  // A valid NET MR remains usable when local P2P transports are disabled.
  window.hasNetworkMrRef = true;
  window.mrIndex = 0;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, false),
            flagcxSuccess);

  // With no peer, the local allocation itself is sufficient.
  comm.nRanks = 1;
  comm.localRanks = 1;
  window.ipcSlot = -1;
  window.hasNetworkMrRef = false;
  window.mrIndex = -1;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, false),
            flagcxSuccess);
}

TEST(SymWindowMrRoute, UsesRecordedVmmProvenanceForPublicRegistration) {
  char allocation[256] = {};
  flagcxMemAllocationInfo info = {allocation, sizeof(allocation), flagcxMemCCL,
                                  flagcxMemAllocBackendNative, true};
  ASSERT_EQ(globalMemAllocRegistry.insert(info), flagcxSuccess);
  EXPECT_TRUE(flagcxOneSideMemoryIsVmm(allocation + 32, 64));
  EXPECT_FALSE(flagcxOneSideMemoryIsVmm(allocation + 240, 32));
  EXPECT_EQ(globalMemAllocRegistry.erase(allocation), flagcxSuccess);
  EXPECT_FALSE(flagcxOneSideMemoryIsVmm(allocation, sizeof(allocation)));
}

// ---------------------------------------------------------------------------
// 3. Parameter validation — NULL / zero args
// ---------------------------------------------------------------------------

TEST(SymWindowValidation, RegisterNullComm) {
  flagcxWindow_t win = nullptr;
  char dummy[64] = {};
  flagcxResult_t res =
      flagcxSymWindowRegister(nullptr, dummy, sizeof(dummy), &win, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, RegisterNullBuff) {
  // Pass a non-null comm placeholder so the test exercises the buff-null check
  // specifically, not the comm-null check. The function checks all args in a
  // single compound condition before dereferencing comm.
  flagcxHeteroComm_t fakeComm = reinterpret_cast<flagcxHeteroComm_t>(0x1);
  flagcxWindow_t win = nullptr;
  flagcxResult_t res =
      flagcxSymWindowRegister(fakeComm, nullptr, 1024, &win, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, RegisterZeroSize) {
  flagcxHeteroComm_t fakeComm = reinterpret_cast<flagcxHeteroComm_t>(0x1);
  flagcxWindow_t win = nullptr;
  char dummy[64] = {};
  flagcxResult_t res = flagcxSymWindowRegister(fakeComm, dummy, 0, &win, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, RegisterNullWinPtr) {
  flagcxHeteroComm_t fakeComm = reinterpret_cast<flagcxHeteroComm_t>(0x1);
  char dummy[64] = {};
  flagcxResult_t res =
      flagcxSymWindowRegister(fakeComm, dummy, sizeof(dummy), nullptr, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, DeregisterNull) {
  // Deregistering a NULL window should be a safe no-op
  flagcxResult_t res = flagcxSymWindowDeregister(nullptr, nullptr);
  EXPECT_EQ(res, flagcxSuccess);
}

TEST(SymWindowValidation, CommWindowDeregisterNull) {
  // Public API: deregistering NULL window should succeed
  flagcxResult_t res = flagcxCommWindowDeregister(nullptr, nullptr);
  EXPECT_EQ(res, flagcxSuccess);
}

TEST(SymWindowOwnership, TeardownFailureRetainsWindowForRetry) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symFlatMappingUnmap = failUnmapOnce;
  testAdaptor.symFlatVaFree = countFlatVaFree;
  testAdaptor.symPhysFree = countPhysFree;
  unmapCalls = 0;
  physFreeCalls = 0;

  auto *comm = static_cast<flagcxHeteroComm *>(
      calloc(1, sizeof(struct flagcxHeteroComm)));
  auto *win =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *sym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(sym, nullptr);
  deviceAdaptor = &testAdaptor;
  win->defaultBase = sym;
  win->isSymmetricDefault = 1;
  sym->owner = win;
  sym->isVMM = true;
  sym->flatBase = reinterpret_cast<void *>(0x1000);
  sym->flatMappingOwned = true;
  sym->flatVaOwned = true;
  sym->physHandle = reinterpret_cast<void *>(0x2000);
  sym->allocSize = 4096;
  sym->localRanks = 2;
  sym->published = true;
  comm->symWindows = sym;

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxRemoteError);
  EXPECT_EQ(comm->symWindows, sym);
  EXPECT_EQ(win->defaultBase, sym);
  EXPECT_EQ(sym->flatBase, reinterpret_cast<void *>(0x1000));
  EXPECT_EQ(sym->physHandle, reinterpret_cast<void *>(0x2000));
  EXPECT_EQ(physFreeCalls, 0);

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(comm->symWindows, nullptr);
  EXPECT_EQ(unmapCalls, 2);
  EXPECT_EQ(physFreeCalls, 1);

  deviceAdaptor = savedAdaptor;
  free(comm);
}

TEST(SymWindowOwnership, FlatUnmapIsNotRepeatedWhenVaFreeRetryIsRequired) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symFlatMappingUnmap = countFlatUnmap;
  testAdaptor.symFlatVaFree = failVaFreeOnce;
  testAdaptor.symPhysFree = countPhysFree;
  unmapCalls = 0;
  vaFreeCalls = 0;
  physFreeCalls = 0;

  auto *comm = static_cast<flagcxHeteroComm *>(
      calloc(1, sizeof(struct flagcxHeteroComm)));
  auto *win =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *sym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(sym, nullptr);
  deviceAdaptor = &testAdaptor;
  win->defaultBase = sym;
  win->isSymmetricDefault = 1;
  sym->owner = win;
  sym->isVMM = true;
  sym->flatBase = reinterpret_cast<void *>(0x1000);
  sym->flatMappingOwned = true;
  sym->flatVaOwned = true;
  sym->physHandle = reinterpret_cast<void *>(0x2000);
  sym->allocSize = 4096;
  sym->localRanks = 2;
  sym->published = true;
  comm->symWindows = sym;

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxRemoteError);
  EXPECT_FALSE(sym->flatMappingOwned);
  EXPECT_TRUE(sym->flatVaOwned);
  EXPECT_EQ(sym->flatBase, reinterpret_cast<void *>(0x1000));
  EXPECT_EQ(unmapCalls, 1);
  EXPECT_EQ(vaFreeCalls, 1);
  EXPECT_EQ(physFreeCalls, 0);

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(comm->symWindows, nullptr);
  EXPECT_EQ(unmapCalls, 1);
  EXPECT_EQ(vaFreeCalls, 2);
  EXPECT_EQ(physFreeCalls, 1);

  deviceAdaptor = savedAdaptor;
  free(comm);
}

TEST(SymWindowOwnership, LocalTeardownSkipsBootstrapRendezvous) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symFlatMappingUnmap = countFlatUnmap;
  testAdaptor.symFlatVaFree = countFlatVaFree;
  testAdaptor.symPhysFree = countPhysFree;
  unmapCalls = 0;
  physFreeCalls = 0;
  cleanupStep = 0;

  auto *comm = static_cast<flagcxHeteroComm *>(
      calloc(1, sizeof(struct flagcxHeteroComm)));
  auto *win =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *sym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(sym, nullptr);
  deviceAdaptor = &testAdaptor;

  // A non-null sentinel proves local teardown does not attempt to use the
  // bootstrap object even though this window describes multiple local ranks.
  comm->bootstrap = reinterpret_cast<bootstrapState *>(0x1);
  comm->symWindows = sym;
  win->defaultBase = sym;
  win->isSymmetricDefault = 1;
  sym->owner = win;
  sym->isVMM = true;
  sym->flatBase = reinterpret_cast<void *>(0x1000);
  sym->flatMappingOwned = true;
  sym->flatVaOwned = true;
  sym->physHandle = reinterpret_cast<void *>(0x2000);
  sym->allocSize = 4096;
  sym->localRanks = 2;
  sym->published = true;

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(comm->symWindows, nullptr);
  EXPECT_EQ(unmapCalls, 1);
  EXPECT_EQ(physFreeCalls, 1);

  deviceAdaptor = savedAdaptor;
  free(comm);
}

TEST(SymWindowOwnership, MulticastTeardownCompletesBeforeObjectRelease) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symMulticastMappingUnmap = failMulticastTeardownOnce;
  testAdaptor.symMulticastVaFree = countMulticastVaFree;
  testAdaptor.symMulticastFree = countMulticastFree;
  testAdaptor.symFlatMappingUnmap = countFlatUnmap;
  testAdaptor.symFlatVaFree = countFlatVaFree;
  testAdaptor.symPhysFree = countPhysFree;
  multicastTeardownCalls = 0;
  multicastFreeCalls = 0;
  unmapCalls = 0;
  physFreeCalls = 0;
  cleanupStep = 0;
  multicastFreeStep = 0;
  flatUnmapStep = 0;
  physFreeStep = 0;

  auto *comm = static_cast<flagcxHeteroComm *>(
      calloc(1, sizeof(struct flagcxHeteroComm)));
  auto *win =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *sym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(sym, nullptr);
  deviceAdaptor = &testAdaptor;
  win->defaultBase = sym;
  win->isSymmetricDefault = 1;
  sym->owner = win;
  sym->isVMM = true;
  sym->mcBase = reinterpret_cast<void *>(0x1000);
  sym->mcMapSize = 4096;
  sym->multicastMappingOwned = true;
  sym->multicastVaOwned = true;
  sym->mcHandle = reinterpret_cast<void *>(0x2000);
  sym->flatBase = reinterpret_cast<void *>(0x3000);
  sym->flatMappingOwned = true;
  sym->flatVaOwned = true;
  sym->physHandle = reinterpret_cast<void *>(0x4000);
  sym->allocSize = 4096;
  sym->localRanks = 2;
  sym->published = true;
  comm->symWindows = sym;

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxRemoteError);
  EXPECT_EQ(comm->symWindows, sym);
  EXPECT_EQ(multicastTeardownCalls, 1);
  EXPECT_EQ(multicastFreeCalls, 0);
  EXPECT_EQ(unmapCalls, 0);
  EXPECT_EQ(physFreeCalls, 0);

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(comm->symWindows, nullptr);
  EXPECT_EQ(multicastTeardownCalls, 2);
  EXPECT_EQ(multicastFreeCalls, 1);
  EXPECT_EQ(unmapCalls, 1);
  EXPECT_EQ(physFreeCalls, 1);
  EXPECT_LT(multicastFreeStep, flatUnmapStep);
  EXPECT_LT(flatUnmapStep, physFreeStep);

  deviceAdaptor = savedAdaptor;
  free(comm);
}

TEST(SymWindowOwnership,
     PerRankMulticastReferencesPermitCreatorFirstLocalDestroy) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symMulticastImport = retainMockMulticastHandle;
  testAdaptor.symMulticastMappingUnmap = teardownMockMulticastMapping;
  testAdaptor.symMulticastVaFree = countMulticastVaFree;
  testAdaptor.symMulticastFree = releaseMockMulticastHandle;
  multicastProviderRefs = 0;
  multicastObjectDestroyCalls = 0;
  multicastTeardownCalls = 0;

  struct flagcxHeteroComm creatorComm = {};
  struct flagcxHeteroComm peerComm = {};
  auto *creatorWin =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *peerWin =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *creatorSym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  auto *peerSym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(creatorWin, nullptr);
  ASSERT_NE(peerWin, nullptr);
  ASSERT_NE(creatorSym, nullptr);
  ASSERT_NE(peerSym, nullptr);
  deviceAdaptor = &testAdaptor;

  ASSERT_EQ(testAdaptor.symMulticastImport(7, &creatorSym->mcHandle),
            flagcxSuccess);
  ASSERT_EQ(testAdaptor.symMulticastImport(7, &peerSym->mcHandle),
            flagcxSuccess);
  ASSERT_EQ(multicastProviderRefs, 2);

  creatorWin->defaultBase = creatorSym;
  creatorWin->isSymmetricDefault = 1;
  creatorSym->owner = creatorWin;
  creatorSym->isVMM = true;
  creatorSym->mcBase = reinterpret_cast<void *>(0x1000);
  creatorSym->mcMapSize = 4096;
  creatorSym->multicastMappingOwned = true;
  creatorSym->multicastVaOwned = true;
  creatorSym->published = true;
  creatorComm.symWindows = creatorSym;

  peerWin->defaultBase = peerSym;
  peerWin->isSymmetricDefault = 1;
  peerSym->owner = peerWin;
  peerSym->isVMM = true;
  peerSym->mcBase = reinterpret_cast<void *>(0x2000);
  peerSym->mcMapSize = 4096;
  peerSym->multicastMappingOwned = true;
  peerSym->multicastVaOwned = true;
  peerSym->published = true;
  peerComm.symWindows = peerSym;

  EXPECT_EQ(flagcxSymWindowDeregister(&creatorComm, creatorWin,
                                      flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(multicastTeardownCalls, 1);
  EXPECT_EQ(multicastProviderRefs, 1);
  EXPECT_EQ(multicastObjectDestroyCalls, 0);
  EXPECT_NE(peerSym->mcBase, nullptr);
  EXPECT_NE(peerSym->mcHandle, nullptr);

  EXPECT_EQ(
      flagcxSymWindowDeregister(&peerComm, peerWin, flagcxSymCleanupLocal),
      flagcxSuccess);
  EXPECT_EQ(multicastTeardownCalls, 2);
  EXPECT_EQ(multicastProviderRefs, 0);
  EXPECT_EQ(multicastObjectDestroyCalls, 1);

  deviceAdaptor = savedAdaptor;
}

TEST(SymWindowLookup, PublishThenFindPreservesNonZeroOffsetAndBounds) {
  char allocation[64] = {};
  struct flagcxHeteroComm comm = {};
  comm.rank = 0;
  comm.nRanks = 1;
  comm.localRanks = 1;
  struct flagcxWindow win = {};
  struct flagcxSymWindow sym = {};
  win.defaultBase = &sym;
  sym.owner = &win;
  sym.localBase = allocation;
  sym.heapSize = sizeof(allocation);

  EXPECT_EQ(flagcxSymWindowFind(&comm, allocation + 7, 8, nullptr), nullptr);
  ASSERT_EQ(flagcxSymWindowPublish(&comm, &win), flagcxSuccess);
  EXPECT_TRUE(sym.published);
  EXPECT_EQ(comm.symWindows, &sym);

  size_t offset = 0;
  EXPECT_EQ(flagcxSymWindowFind(&comm, allocation + 7, 8, &offset), &sym);
  EXPECT_EQ(offset, 7u);
  EXPECT_EQ(flagcxSymWindowFind(&comm, allocation + 60, 4, &offset), &sym);
  EXPECT_EQ(offset, 60u);
  EXPECT_EQ(flagcxSymWindowFind(&comm, allocation + 60, 5, &offset), nullptr);

  // Publishing is idempotent and must not create a self-referential list.
  EXPECT_EQ(flagcxSymWindowPublish(&comm, &win), flagcxSuccess);
  EXPECT_EQ(sym.next, nullptr);
}

TEST(SymWindowLookup, ResolveSelfPeerValidatesRangeBeforePointerArithmetic) {
  char allocation[64] = {};
  int rankToNode[] = {0};
  int rankToLocalRank[] = {0};
  struct flagcxHeteroComm comm = {};
  comm.rank = 0;
  comm.nRanks = 1;
  comm.node = 0;
  comm.rankToNode = rankToNode;
  comm.rankToLocalRank = rankToLocalRank;

  struct flagcxSymWindow sym = {};
  sym.localBase = allocation;
  sym.heapSize = sizeof(allocation);
  sym.localRanks = 1;

  void *peer = nullptr;
  EXPECT_EQ(flagcxSymWindowResolveIpcPeerPtr(&comm, &sym, 0, 9, 8, &peer),
            flagcxSuccess);
  EXPECT_EQ(peer, static_cast<void *>(allocation + 9));
  peer = reinterpret_cast<void *>(0x1);
  EXPECT_EQ(flagcxSymWindowResolveIpcPeerPtr(&comm, &sym, 0, 60, 5, &peer),
            flagcxInvalidArgument);
  EXPECT_EQ(peer, nullptr);
}
