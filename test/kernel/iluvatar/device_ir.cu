// Reuse the complete IR test surface with CoreX's explicit device-pass marker.
#if defined(__IVCORE_ARCH__)
#define FLAGCX_ILUVATAR_DEVICE_COMPILE 1
#endif

#include "../nvidia/device_ir.cu"
