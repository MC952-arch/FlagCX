# makefiles/platforms/kunlunxin.mk
# KunlunXin platform configuration.

DEVICE_HOME  ?= /usr/local/xpu
DEVICE_LIB   := $(DEVICE_HOME)/so
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lxpurt -lcudart -lxpucuda -lxpuml
# DEVICE_PLATFORM selects the test kernel subdir (test/kernel/<lowercase>): klx.
DEVICE_PLATFORM := KLX
# XPU (XTDK) clang toolchain — only used to compile test .xpu kernels
# (COMPILE_KERNEL / test/kernel/klx). The main libflagcx.so build uses g++.
XTDK_HOME ?= /workspace/my_flagcx/xtdk-llvm15-ubuntu2004_x86_64
DEVICE_COMPILER := $(XTDK_HOME)/bin/clang++
DEVICE_COMPILE_FLAG :=
DEVICE_LINK_FLAG :=
DEVICE_FILE_EXTENSION := xpu

CCL_HOME    ?= /usr/local/xccl
CCL_LIB     := $(CCL_HOME)/so
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lbkcl
ADAPTOR_FLAG := -DUSE_KUNLUNXIN_ADAPTOR

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/kunlunxin
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))

ifeq ($(USE_XSHMEM), 1)
  SHMEM_HOME := $(CCL_HOME)
  PLATFORM_EXTRA_SRCS := flagcx/adaptor/shmem/xshmem_adaptor.cc \
                         flagcx/adaptor/device_api/xshmem_dev_api_backend.cc
else
  PLATFORM_EXTRA_SRCS := flagcx/adaptor/device_api/default_dev_api_backend.cc
endif
