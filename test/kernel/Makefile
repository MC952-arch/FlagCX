#*************************************************************************
# Copyright (c) 2026 BAAI. All rights reserved.
#*************************************************************************

MPI_HOME ?= /usr/local/mpi
MPI_INCLUDE = $(MPI_HOME)/include
MPI_LIB = $(MPI_HOME)/lib
MPI_LINK = -lmpi
COMPILER = g++
EXTRA_COMPILER_FLAG = -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wl,-rpath,../../build/lib -std=c++11 -g

BUILDDIR := build

INCLUDEDIR := $(abspath ../include)
LIBSRCFILES:= ../tools.cc

DEVICE_HOME ?=
DEVICE_LIB =
DEVICE_INCLUDE =
DEVICE_LINK =
DEVICE_COMPILER =
PLATFORM_INCLUDES =
ifeq ($(USE_NVIDIA), 1)
	DEVICE_HOME = /usr/local/cuda
	DEVICE_LIB = $(DEVICE_HOME)/lib64 -L$(DEVICE_HOME)/lib64/stubs
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcudart -lcuda
	DEVICE_COMPILER = $(DEVICE_HOME)/bin/nvcc
	PLATFORM_INCLUDES = -I../../bindings/ir/nvidia
else ifeq ($(USE_DU), 1)
        DEVICE_HOME = $(CUDA_PATH)
        DEVICE_LIB = $(DEVICE_HOME)/lib64 -L$(DEVICE_HOME)/lib64/stubs
        DEVICE_INCLUDE = $(DEVICE_HOME)/include
        DEVICE_LINK = -lcudart -lcuda
        DEVICE_COMPILER = $(DEVICE_HOME)/bin/nvcc
        PLATFORM_INCLUDES = -I../../bindings/ir/du
endif

COMMON_INCLUDES = -I../../flagcx/include -I../../flagcx/adaptor/include -I../../flagcx/adaptor/include/device_api -I../../flagcx/service/include -I../../flagcx/core/include -I../../bindings/ir -I../../third-party/json/single_include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -I$(DEVICE_INCLUDE)
COMMON_LIBS = -L../../build/lib -L$(MPI_LIB) -L$(DEVICE_LIB) -lflagcx $(MPI_LINK) $(DEVICE_LINK)

DEVICE_API_OBJ := $(BUILDDIR)/device_api.o

all: test-intranode test-internode-twosided test-internode-onesided test-device-api test-multi-fifo test-device-ir

# Compile device_api.cu (test kernels) into object file
$(DEVICE_API_OBJ): device_api.cu
	@mkdir -p $(BUILDDIR)
	@echo "Compiling $@ (CUDA)"
	@$(DEVICE_COMPILER) -std=c++11 -arch=sm_80 -c -o $@ $< $(COMMON_INCLUDES) $(PLATFORM_INCLUDES) -DCOMPILE_KERNEL -DUSE_NVIDIA_ADAPTOR

# Alias for external Makefiles that invoke: make -C test/kernel device_api.o
device_api.o: $(DEVICE_API_OBJ)

test-intranode: test_intranode.cpp $(DEVICE_API_OBJ)
	@mkdir -p $(BUILDDIR)
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o $(BUILDDIR)/test_intranode test_intranode.cpp $(DEVICE_API_OBJ) $(LIBSRCFILES) $(COMMON_INCLUDES) -I. $(COMMON_LIBS) -DCOMPILE_KERNEL

test-internode-twosided: test_internode_twosided.cpp $(DEVICE_API_OBJ)
	@mkdir -p $(BUILDDIR)
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o $(BUILDDIR)/test_internode_twosided test_internode_twosided.cpp $(DEVICE_API_OBJ) $(LIBSRCFILES) $(COMMON_INCLUDES) -I. $(COMMON_LIBS) -DCOMPILE_KERNEL

test-internode-onesided: test_internode_onesided.cpp $(DEVICE_API_OBJ)
	@mkdir -p $(BUILDDIR)
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o $(BUILDDIR)/test_internode_onesided test_internode_onesided.cpp $(DEVICE_API_OBJ) $(LIBSRCFILES) $(COMMON_INCLUDES) -I. $(COMMON_LIBS) -DCOMPILE_KERNEL

test-device-api: test_device_api.cpp $(DEVICE_API_OBJ)
	@mkdir -p $(BUILDDIR)
	@echo "Compiling $@ (CUDA)"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o $(BUILDDIR)/test_device_api test_device_api.cpp $(DEVICE_API_OBJ) $(LIBSRCFILES) $(COMMON_INCLUDES) -I. $(COMMON_LIBS) -DCOMPILE_KERNEL

test-multi-fifo: test_multi_fifo.cpp $(DEVICE_API_OBJ)
	@mkdir -p $(BUILDDIR)
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o $(BUILDDIR)/test_multi_fifo test_multi_fifo.cpp $(DEVICE_API_OBJ) $(LIBSRCFILES) $(COMMON_INCLUDES) -I. $(COMMON_LIBS) -DCOMPILE_KERNEL

test-device-ir: test_device_ir.cu
	@mkdir -p $(BUILDDIR)
	@echo "Compiling $@ (CUDA)"
	@$(DEVICE_COMPILER) -std=c++11 -arch=sm_80 -Xlinker -rpath -Xlinker ../../build/lib -o $(BUILDDIR)/test_device_ir test_device_ir.cu $(LIBSRCFILES) $(COMMON_INCLUDES) $(PLATFORM_INCLUDES) $(COMMON_LIBS) -DCOMPILE_KERNEL -DUSE_NVIDIA_ADAPTOR

clean:
	@rm -rf $(BUILDDIR)

print_var:
	@echo "INCLUDEDIR: $(INCLUDEDIR)"
	@echo "USE_NVIDIA: $(USE_NVIDIA)"
	@echo "MPI_INCLUDE: $(MPI_INCLUDE)"
	@echo "MPI_LIB: $(MPI_LIB)"
	@echo "COMPILER: $(COMPILER)"
	@echo "EXTRA_COMPILER_FLAG: $(EXTRA_COMPILER_FLAG)"
