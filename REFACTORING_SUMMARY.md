# FlagCX Device API Refactoring Summary

## Changes Applied

### 1. Split `flagcx_device.h` into three files:

**`flagcx_device_core.h`** (NEW, 960 lines)
- Device-side types and inline functions only
- Safe for LLVM bitcode compilation (`--cuda-device-only`)
- Includes: `device_utils.h`, `flagcx.h`, `comm_traits.h`
- Does NOT include: `flagcx_kernel.h`, `shmutils.h`, `adaptor.h`
- Contains: Sections 3-12 (flagcxDevComm, flagcxDevMem, flagcxTeam, barriers, etc.)
- Constructors from `flagcxDevCommInternal`/`flagcxDevMemInternal` guarded with `#ifndef __clang_llvm_bitcode_lib__`

**`flagcx_device_internal.h`** (NEW, 140 lines)
- Host-side internal structs only
- NOT safe for bitcode compilation
- Includes: `flagcx_kernel.h`, `shmutils.h`
- Contains: Sections 1-2 (flagcxDevCommInternal, flagcxDevMemInternal)

**`flagcx_device.h`** (MODIFIED, now 26 lines)
- Umbrella header that includes both core + internal
- For bitcode: only includes `flagcx_device_core.h`
- For normal builds: includes both headers
- Zero impact on existing code

### 2. Updated bitcode wrapper

**`bindings/ir/flagcx_device_wrapper.h`**
- Changed: `#include "flagcx_device.h"` → `#include "flagcx_device_core.h"`
- Now only pulls device-safe headers

### 3. Guard host headers in `flagcx_kernel.h`

**`flagcx/include/flagcx_kernel.h`** line 4:
```c
#ifndef __clang_llvm_bitcode_lib__
#include "adaptor.h"
#endif
```

### 4. Guard CUDA headers in `device_utils.h`

**`flagcx/adaptor/include/device_utils.h`** lines 33-35:
```c
#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
#ifndef __clang_llvm_bitcode_lib__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
```

## Benefits

1. **Clean separation**: Device vs host concerns clearly separated
2. **Bitcode-safe**: No host infrastructure headers in bitcode path
3. **Zero migration cost**: Existing `#include "flagcx_device.h"` unchanged
4. **Follows NCCL pattern**: Similar to `nccl_device/core.h` + `impl/*__types.h`
5. **No scattered guards**: All guards in logical places (file boundaries)

## Files Modified

- `flagcx/adaptor/include/device_api/flagcx_device.h` (1054 → 26 lines)
- `flagcx/adaptor/include/device_api/flagcx_device_core.h` (NEW, 960 lines)
- `flagcx/adaptor/include/device_api/flagcx_device_internal.h` (NEW, 140 lines)
- `flagcx/include/flagcx_kernel.h` (guard adaptor.h include)
- `flagcx/adaptor/include/device_utils.h` (guard CUDA headers)
- `bindings/ir/flagcx_device_wrapper.h` (include path change)

## Next Steps

1. Test bitcode compilation on remote machine (workspace, not /share/project)
2. Verify normal builds still work
3. Run full test suite
