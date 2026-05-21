#*************************************************************************
# Copyright (c) 2026 BAAI. All rights reserved.
#
# test/kernel/ — standalone MPI+GPU integration tests for FlagCX Device API.
# Each test_*.cpp / test_*.cu produces a separate binary in build/bin/.
#*************************************************************************

include ../make.inc

# ---------------------------------------------------------------------------
# CUDA / Device compiler (from make.inc device platform resolution)
# ---------------------------------------------------------------------------
PLATFORM_INCLUDES = -I$(PLATFORM_IR_DIR)

# Auto-detect GPU architectures (same as main Makefile)
ifeq ($(USE_NVIDIA), 1)
include $(PROJECT_ROOT)/makefiles/nvidia_gencode.mk
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
ALL_INCLUDES := $(CCL_INCLUDE) $(FLAGCX_INCLUDES) $(IR_INCLUDE) $(PLATFORM_INCLUDES) \
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
# Shared objects: device_api.o + device_ir.o + tools.o (compiled once)
# ---------------------------------------------------------------------------
DEVICE_API_OBJ := $(OBJDIR)/device_api.o
DEVICE_IR_OBJ  := $(OBJDIR)/device_ir.o
TOOLS_OBJ      := $(OBJDIR)/tools.o

# ---------------------------------------------------------------------------
# Targets — one binary per test source
# ---------------------------------------------------------------------------
CPP_TESTS := test_intranode test_internode_twosided test_internode_onesided \
             test_device_api test_multi_fifo test_device_ir

ALL_TARGETS := $(CPP_TESTS:%=$(BINDIR)/%)

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
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) -c -o $@ $< \
		$(CU_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

# CUDA kernel object (device_ir.cu — IR wrapper tests)
$(DEVICE_IR_OBJ): device_ir.cu device_ir.h
	@mkdir -p $(OBJDIR)
	@echo "Compiling $< (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) -c -o $@ $< \
		$(CU_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

# tools.cc (shared perf infrastructure)
$(TOOLS_OBJ): $(PROJECT_ROOT)/test/tools.cc
	@mkdir -p $(OBJDIR)
	@echo "Compiling tools.cc"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(ALL_INCLUDES)

# C++ test binaries (default: link with device_api.o)
$(BINDIR)/%: %.cpp $(DEVICE_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_API_OBJ) $(TOOLS_OBJ) \
		$(ALL_INCLUDES) $(ALL_LINK) -DCOMPILE_KERNEL

# test_device_ir links with device_ir.o instead of device_api.o
$(BINDIR)/test_device_ir: test_device_ir.cpp $(DEVICE_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(TOOLS_OBJ) \
		$(ALL_INCLUDES) $(ALL_LINK) -DCOMPILE_KERNEL

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	@rm -rf $(BUILDDIR)
