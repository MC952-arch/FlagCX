#*************************************************************************
# Copyright (c) 2026 BAAI. All rights reserved.
#
# test/kernel/ — standalone MPI+GPU integration tests for FlagCX Device API.
# Each test_*.cpp / test_*.cu produces a separate binary in build/bin/.
#*************************************************************************

include ../../make.inc

# ---------------------------------------------------------------------------
# CUDA / Device compiler
# ---------------------------------------------------------------------------
DEVICE_HOME ?=
DEVICE_INCLUDE =
DEVICE_LIB =
DEVICE_LINK =
DEVICE_COMPILER =
PLATFORM_INCLUDES =

ifeq ($(USE_NVIDIA), 1)
DEVICE_HOME      ?= /usr/local/cuda
DEVICE_INCLUDE    = $(DEVICE_HOME)/include
DEVICE_LIB        = $(DEVICE_HOME)/lib64
DEVICE_LINK       = -L$(DEVICE_LIB) -L$(DEVICE_HOME)/lib64/stubs -lcudart -lcuda
DEVICE_COMPILER   = $(DEVICE_HOME)/bin/nvcc
PLATFORM_INCLUDES = -I$(PROJECT_ROOT)/bindings/ir/nvidia
else ifeq ($(USE_DU), 1)
DEVICE_HOME      ?= $(CUDA_PATH)
DEVICE_INCLUDE    = $(DEVICE_HOME)/include
DEVICE_LIB        = $(DEVICE_HOME)/lib64
DEVICE_LINK       = -L$(DEVICE_LIB) -L$(DEVICE_HOME)/lib64/stubs -lcudart -lcuda
DEVICE_COMPILER   = $(DEVICE_HOME)/bin/nvcc
PLATFORM_INCLUDES = -I$(PROJECT_ROOT)/bindings/ir/du
endif

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
# IR wrapper headers
IR_INCLUDE := -I$(PROJECT_ROOT)/bindings/ir

# Device API internal headers (device_api.h lives in this directory)
LOCAL_INCLUDE := -I.

# Perf test infrastructure (tools.h)
PERF_INCLUDE := -I$(PROJECT_ROOT)/test/include

# Aggregate all includes for .cpp targets
ALL_INCLUDES := $(FLAGCX_INCLUDES) $(IR_INCLUDE) $(PLATFORM_INCLUDES) \
                $(LOCAL_INCLUDE) $(PERF_INCLUDE) $(MPI_INCLUDE) \
                -I$(DEVICE_INCLUDE)

# Aggregate all includes for .cu targets (adds adaptor internals for nvcc)
CU_INCLUDES := $(ALL_INCLUDES) \
               -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api

# ---------------------------------------------------------------------------
# Link flags — FLAGCX_LINK from make.inc provides -L, -rpath (absolute), -lflagcx
# ---------------------------------------------------------------------------
ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK)

# ---------------------------------------------------------------------------
# Shared objects: device_api.o + tools.o (compiled once, linked by all targets)
# ---------------------------------------------------------------------------
DEVICE_API_OBJ := $(OBJDIR)/device_api.o
TOOLS_OBJ      := $(OBJDIR)/tools.o

# ---------------------------------------------------------------------------
# Targets — one binary per test source
# ---------------------------------------------------------------------------
CPP_TESTS := test_intranode test_internode_twosided test_internode_onesided \
             test_device_api test_multi_fifo

CU_TESTS := test_device_ir

ALL_TARGETS := $(CPP_TESTS:%=$(BINDIR)/%) $(CU_TESTS:%=$(BINDIR)/%)

.PHONY: all clean device_api.o

all: $(ALL_TARGETS)

# Alias for test/unittest/kernel/Makefile: `make -C test/kernel device_api.o`
device_api.o: $(DEVICE_API_OBJ)

# ---------------------------------------------------------------------------
# Compile rules
# ---------------------------------------------------------------------------

# CUDA kernel object (device_api.cu)
$(DEVICE_API_OBJ): device_api.cu device_api.h
	@mkdir -p $(OBJDIR)
	@echo "Compiling $< (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 -arch=sm_80 -c -o $@ $< \
		$(CU_INCLUDES) -DCOMPILE_KERNEL -DUSE_NVIDIA_ADAPTOR

# tools.cc (shared perf infrastructure)
$(TOOLS_OBJ): $(PROJECT_ROOT)/test/tools.cc
	@mkdir -p $(OBJDIR)
	@echo "Compiling tools.cc"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(ALL_INCLUDES)

# C++ test binaries
$(BINDIR)/%: %.cpp $(DEVICE_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_API_OBJ) $(TOOLS_OBJ) \
		$(ALL_INCLUDES) $(ALL_LINK) -DCOMPILE_KERNEL

# CUDA test binary (test_device_ir)
$(BINDIR)/test_device_ir: test_device_ir.cu
	@mkdir -p $(BINDIR)
	@echo "Linking   $@ (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 -arch=sm_80 -o $@ $< $(TOOLS_OBJ) \
		$(CU_INCLUDES) $(ALL_LINK) -DCOMPILE_KERNEL -DUSE_NVIDIA_ADAPTOR

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	@rm -rf $(BUILDDIR)
