# test/kernel/klx/Makefile
# Kunlunxin (P800) test kernel compilation.
#
# device_api.xpu talks to native xshmem device API (no comm_traits). It is
# compiled by the XTDK clang in a single relocatable object that device-links
# libxshmem_device.a inline (-xpu-l xshmem_device), so the resulting
# device_api.o is host-linkable by g++ into the perf/unittest binaries.

include ../../make.inc

# Toolchain / SDK roots (symlinks under /usr/local point at the SDKs).
XTDK_HOME ?= /workspace/my_flagcx/xtdk-llvm15-ubuntu2004_x86_64
XCCL_HOME ?= /usr/local/xccl
XRE_HOME  ?= /usr/local/xpu
XPU_CLANG := $(XTDK_HOME)/bin/clang++

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/bindings/ir \
    -I$(PROJECT_ROOT)/test/kernel/include \
    -I$(XCCL_HOME)/include \
    -I$(XRE_HOME)/include

# xpu3 = P800. c++17 for the FlagCX headers. Device-link the xshmem static lib
# so the __global__ kernel's xshmem device symbols resolve at compile time.
XPU_CFLAGS := --xpu-arch=xpu3 -std=c++17 -O2 -fno-builtin \
    --target=x86_64-linux-gnu -DUSE_KUNLUNXIN_ADAPTOR -DCOMPILE_KERNEL
XPU_DEVLINK := -xpu-L $(XCCL_HOME)/lib -xpu-l xshmem_device

.PHONY: all clean device_api.o device_ir.o dlink.o dlink_api.o dlink_ir.o

all: $(OBJDIR)/device_api.o

device_api.o: $(OBJDIR)/device_api.o

$(OBJDIR)/device_api.o: device_api.xpu
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.xpu (KLX / native xshmem)"
	@$(XPU_CLANG) $(XPU_CFLAGS) $(EXTRA_XPU_FLAGS) $(ALL_INCLUDES) $(XPU_DEVLINK) -c -o $@ $<

# The native xshmem path performs its own device-link inline (above), so the
# generic SHMEM/IR/device-link forward targets are no-ops here.
device_ir.o dlink.o dlink_api.o dlink_ir.o:
	@echo "KLX: no separate device-link step (xshmem device-linked in device_api.o)"

clean:
	@rm -rf $(BUILDDIR)
