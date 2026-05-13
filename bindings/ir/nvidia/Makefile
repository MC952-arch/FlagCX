# Copyright (c) 2026 BAAI. All rights reserved.
#
# LLVM IR generation for FlagCX Device API — NVIDIA CUDA platform.
#
# Usage:
#   make EMIT_LLVM_IR=1 USE_NVIDIA=1 \
#        DEVICE_HOME=/usr/local/cuda CCL_HOME=/usr/local/nccl/build
#
# Produces:
#   $(BUILDDIR)/lib/libflagcx_device.bc    — final bitcode for Triton linking
#   $(BUILDDIR)/include/flagcx_device_wrapper.h — consumer header

EMIT_LLVM_IR ?= 1

ifneq ($(EMIT_LLVM_IR), 0)

# Source (one level up — shared wrapper)
LLVM_SRC := ../flagcx_device_wrapper_impl.h
BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/llvm_ir
LIBDIR := $(BUILDDIR)/lib
INCLUDEDIR := $(BUILDDIR)/include

# Output files
UNOPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.unoptimized
OPTIMIZED_BC   := $(OBJDIR)/libflagcx_device.bc.optimized
LLVM_IR_FILE   := $(OBJDIR)/libflagcx_device.ll
FINAL_BC       := $(LIBDIR)/libflagcx_device.bc
WRAPPER_HEADER := $(INCLUDEDIR)/flagcx_device_wrapper.h

# GPU architecture
DEVICE_HOME ?= /usr/local/cuda
CCL_HOME    ?= /usr/local/nccl/build

# Detect CUDA version for arch selection
CUDA_MAJOR ?= $(shell $(DEVICE_HOME)/bin/nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1)
ifeq ($(CUDA_MAJOR),)
  CUDA_MAJOR := 12
endif
CUDA_MINOR ?= 0

ifeq ($(shell test "$(CUDA_MAJOR)" -lt 12; echo $$?),0)
  BITCODE_LIB_ARCH ?= sm_70
else
  BITCODE_LIB_ARCH ?= sm_90
endif

BITCODE_CXX_STD ?= c++17

# Tools
CLANG    ?= clang
OPT      ?= opt
LLVM_DIS ?= llvm-dis
LLVM_AS  ?= llvm-as

# FlagCX source root (three levels up from bindings/ir/nvidia/)
FLAGCX_ROOT := $(abspath ../../..)

# Include paths
#   -I..  : shared wrapper headers (flagcx_device_wrapper.h, flagcx_device_wrapper_impl.h)
#   -I.   : platform-specific headers (flagcx_device_types.h)
FLAGCX_INCLUDES := \
	-I.. \
	-I. \
	-I$(FLAGCX_ROOT)/flagcx/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api \
	-I$(FLAGCX_ROOT)/flagcx/core/include \
	-I$(FLAGCX_ROOT)/flagcx/runner/include

CUDA_INCLUDES := \
	-I$(DEVICE_HOME)/include \
	-I$(DEVICE_HOME)/include/cccl

CCL_INCLUDES := \
	-I$(CCL_HOME)/include

# Adaptor flag (determines which CommTraits specialization is active)
ADAPTOR_FLAG ?= -DUSE_NVIDIA_ADAPTOR

# Clang flags for LLVM IR generation
COMMON_CLANG_FLAGS := \
	-std=$(BITCODE_CXX_STD) -x cuda \
	--cuda-path=$(DEVICE_HOME) --cuda-device-only \
	--cuda-gpu-arch=$(BITCODE_LIB_ARCH) \
	$(FLAGCX_INCLUDES) $(CUDA_INCLUDES) $(CCL_INCLUDES) \
	-D__clang_llvm_bitcode_lib__ \
	-DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR) \
	$(ADAPTOR_FLAG)

CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS)

# Optimization passes
OPT_PASSES := --passes='internalize,inline,globaldce' \
	-internalize-public-api-list='*flagcx*'

# ---- Build rules ----

$(WRAPPER_HEADER): ../flagcx_device_wrapper.h
	@mkdir -p $(INCLUDEDIR)
	cp -f $< $@

$(UNOPTIMIZED_BC): $(LLVM_SRC)
	@mkdir -p $(OBJDIR)
	@echo "Generating unoptimized LLVM bitcode..."
	$(CLANG) $(CLANG_FLAGS) $(LLVM_SRC) -o $@

$(OPTIMIZED_BC): $(UNOPTIMIZED_BC)
	@echo "Optimizing LLVM bitcode..."
	$(OPT) $(OPT_PASSES) $< -o $@

$(LLVM_IR_FILE): $(OPTIMIZED_BC)
	@echo "Generating LLVM IR..."
	$(LLVM_DIS) $< -o $@.tmp
	@echo "Cleaning LLVM IR (removing nvvm-reflect-ftz)..."
	@myVar="$$(cat $@.tmp | grep -E '!([0-9]+) = !\{[^"]*"nvvm-reflect-ftz"' | cut -d ' ' -f 1)"; \
	awk '!/nvvm-reflect-ftz/' $@.tmp | sed "/^!llvm\.module\.flags = /s/$$myVar, //" > $@
	@rm -f $@.tmp

$(FINAL_BC): $(LLVM_IR_FILE)
	@mkdir -p $(LIBDIR)
	@echo "Generating final bitcode from cleaned LLVM IR..."
	$(LLVM_AS) $< -o $@

llvm_ir: $(FINAL_BC) $(OPTIMIZED_BC) $(WRAPPER_HEADER)
	@echo "LLVM IR and bitcode generated successfully:"
	@echo "  C++ Standard:      $(BITCODE_CXX_STD)"
	@echo "  GPU Architecture:  $(BITCODE_LIB_ARCH)"
	@echo "  Adaptor:           $(ADAPTOR_FLAG)"
	@echo "  Unoptimized:       $(UNOPTIMIZED_BC)"
	@echo "  Optimized:         $(OPTIMIZED_BC)"
	@echo "  LLVM IR:           $(LLVM_IR_FILE)"
	@echo "  Final BC:          $(FINAL_BC)"
	@echo "  Wrapper Header:    $(WRAPPER_HEADER)"

else
llvm_ir:
	@echo "LLVM IR generation disabled (EMIT_LLVM_IR=0)"
endif

build: llvm_ir

clean:
	rm -rf $(OBJDIR) $(FINAL_BC) $(WRAPPER_HEADER)

.PHONY: llvm_ir build clean
