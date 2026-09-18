// Reuse the platform-neutral CUDA-compatible native Device API tests. CoreX
// defines __IVCORE_ARCH__ only for the device pass.
#if defined(__IVCORE_ARCH__)
#define FLAGCX_ILUVATAR_DEVICE_COMPILE 1
#endif

#include "../nvidia/device_api.cu"

// Compile every operation in the DefaultBackend atomic contract, including
// the less frequently used bitwise/exchange paths. Runtime acceptance still
// requires the single- and dual-device tests on real CoreX hardware.
__global__ void flagcxIluvatarAtomicContractKernel(uint64_t *value) {
  if (FLAGCX_THREAD_IDX_X != 0)
    return;
  uint64_t expected =
      DeviceAPI::Atomic::load(value, flagcxDeviceMemoryOrderAcquire);
  DeviceAPI::Atomic::store(value, expected, flagcxDeviceMemoryOrderRelease);
  DeviceAPI::Atomic::fetchAdd(value, uint64_t{1},
                              flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::fetchSub(value, uint64_t{1},
                              flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::fetchOr(value, uint64_t{1}, flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::fetchAnd(value, ~uint64_t{0},
                              flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::exchange(value, expected, flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::compareExchange(value, expected, expected,
                                     flagcxDeviceMemoryOrderAcqRel);
}

// This kernel is intentionally not part of the normal success path. A CoreX
// runtime test launches it separately and expects a kernel error, proving that
// an unsupported partial mask fails rather than widening to a full-wave
// barrier and hanging.
__global__ void flagcxIluvatarUnsupportedCoopKernel() {
  flagcxCoopLanes partial(flagcxLaneMask_t{3});
  partial.sync();
}
