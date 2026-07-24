include ../../make.inc

# Auto-detect GPU architectures
ifeq ($(USE_NVIDIA), 1)
include $(PROJECT_ROOT)/makefiles/nvidia_gencode.mk
endif

# Each test_*.cpp becomes its own binary: $(BINDIR)/perf_<name>
SRCS := $(wildcard test_*.cpp)
BINS := $(SRCS:test_%.cpp=$(BINDIR)/perf_%)

# Shared source files
TOOLS_SRC   := $(PROJECT_ROOT)/test/tools.cc
COMMON_SRCS := $(TOOLS_SRC)

# Shared device_api object (built in test/device_api/)
DEVICE_API_DIR := $(PROJECT_ROOT)/test/device_api
DEVICE_API_OBJ := $(DEVICE_API_DIR)/build/obj/device_api.o

# Device-link object (only needed for NVSHMEM)
# Use dlink_api.o — perf tests only link device_api.o, not device_ir.o
ifeq ($(USE_SHMEM), 1)
DLINK_OBJ := $(DEVICE_API_DIR)/build/obj/dlink_api.o
SHMEM_SYNC_OBJ := $(DEVICE_API_DIR)/build/obj/nvshmem_state_sync.o
else
DLINK_OBJ :=
SHMEM_SYNC_OBJ :=
endif

# Includes: flagcx, perf headers, MPI, device SDK, device_api headers
ALL_INCLUDES := $(FLAGCX_INCLUDES) $(PERF_INCLUDE) $(MPI_INCLUDE) \
    -I$(PROJECT_ROOT)/flagcx/core -I$(PROJECT_ROOT)/flagcx/service \
    -I$(DEVICE_API_DIR) $(DEVICE_INCLUDE)

ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK)

# Kernel tests need -DCOMPILE_KERNEL
CXXFLAGS += -DCOMPILE_KERNEL

.PHONY: all clean

all: $(BINS)

# Build device_api.o (and dlink_api.o if USE_SHMEM) via shared Makefile
DEVICE_API_TARGETS := device_api.o
ifeq ($(USE_SHMEM), 1)
DEVICE_API_TARGETS += nvshmem_state_sync.o dlink_api.o
endif

$(DEVICE_API_OBJ) $(DLINK_OBJ) $(SHMEM_SYNC_OBJ):
	@$(MAKE) -C $(DEVICE_API_DIR) $(DEVICE_API_TARGETS) USE_NVIDIA=$(USE_NVIDIA) USE_DU=$(USE_DU) \
		USE_SHMEM=$(USE_SHMEM) SHMEM_HOME=$(SHMEM_HOME)

$(BINDIR)/perf_%: test_%.cpp $(COMMON_SRCS) $(DEVICE_API_OBJ) $(DLINK_OBJ) $(SHMEM_SYNC_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Compiling $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_SRCS) $(DEVICE_API_OBJ) $(SHMEM_SYNC_OBJ) $(DLINK_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)
