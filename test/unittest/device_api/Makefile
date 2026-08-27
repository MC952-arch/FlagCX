include ../../make.inc

# Shared device kernel objects (built in test/kernel/[platform]/)
DEVICE_API_OBJ := $(KERNEL_OBJ_DIR)/device_api.o
DEVICE_IR_OBJ  := $(KERNEL_OBJ_DIR)/device_ir.o

# Device-link objects (only needed for NVSHMEM). KLX selects xccl xshmem with the
# same USE_SHMEM switch, but its device code is linked into device_api.o.
ifeq ($(USE_SHMEM)$(KERNEL_PLATFORM), 1klx)
DLINK_API_OBJ :=
DLINK_IR_OBJ  :=
else ifeq ($(USE_SHMEM), 1)
DLINK_API_OBJ := $(KERNEL_OBJ_DIR)/dlink_api.o
DLINK_IR_OBJ  := $(KERNEL_OBJ_DIR)/dlink_ir.o
else
DLINK_API_OBJ :=
DLINK_IR_OBJ  :=
endif

# Shared tools object
TOOLS_SRC := $(PROJECT_ROOT)/test/tools.cc
TOOLS_OBJ := $(OBJDIR)/tools.o

# IR wrapper headers
IR_INCLUDE := -I$(PROJECT_ROOT)/bindings/ir
PLATFORM_INCLUDES := -I$(PLATFORM_IR_DIR)

# All includes
ALL_INCLUDES := $(FLAGCX_INCLUDES) $(IR_INCLUDE) $(PLATFORM_INCLUDES) \
    $(KERNEL_INCLUDE) $(PERF_INCLUDE) $(MPI_INCLUDE) $(DEVICE_INCLUDE)

ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK)

# Kunlunxin: device_api.o (native xshmem kernel) references xshmem host symbols
# that live in libbkcl, and the XRE runtime libs. libflagcx.so imports (not
# exports) these, so the executable must link them explicitly. Add rpaths so the
# binary finds libbkcl / libxpurt* at runtime.
ifeq ($(USE_KUNLUNXIN), 1)
ALL_LINK += -L$(CCL_LIB) -Wl,-rpath,$(CCL_LIB) $(CCL_LINK) \
            -Wl,-rpath,$(DEVICE_LIB)
endif

# Kernel flag
CXXFLAGS += -DCOMPILE_KERNEL

# Test targets. The Device API tests link test/kernel/<platform>/device_api.o,
# which provides every launcher in test/kernel/include/device_api.h (KLX
# implements two natively and stubs the rest as flagcxNotSupported), so they
# always link. The IR tests instead need a device_ir.<ext> kernel, which KLX has
# no counterpart for -- skip them EXPLICITLY there rather than letting the build
# fail on a missing object / undefined references.
IR_TARGETS  := $(BINDIR)/test_device_ir_intra \
               $(BINDIR)/test_device_ir_inter
API_TARGETS := $(BINDIR)/test_device_api_intra \
               $(BINDIR)/test_device_api_inter

SKIPPED_TARGETS :=
ifeq ($(KERNEL_PLATFORM), klx)
  SKIPPED_TARGETS := $(notdir $(IR_TARGETS))
  IR_TARGETS :=
endif

MPI_TARGETS := $(IR_TARGETS) $(API_TARGETS)

.PHONY: all clean run-unit run-mpi run-mpi-intra run-mpi-inter run

all: $(MPI_TARGETS)
ifneq ($(SKIPPED_TARGETS),)
	@echo "Skipped on $(KERNEL_PLATFORM) (no device_ir.$(DEVICE_FILE_EXTENSION) kernel in test/kernel/$(KERNEL_PLATFORM)): $(SKIPPED_TARGETS)"
endif

# Build device kernel objects via dispatcher
KERNEL_API_TARGETS := device_api.o
KERNEL_IR_TARGETS  := device_ir.o
ifneq ($(DLINK_API_OBJ),)
KERNEL_API_TARGETS += dlink_api.o
KERNEL_IR_TARGETS  += dlink_ir.o
endif

$(DEVICE_API_OBJ) $(DLINK_API_OBJ):
	@$(MAKE) -C $(KERNEL_DIR) $(KERNEL_API_TARGETS)

$(DEVICE_IR_OBJ) $(DLINK_IR_OBJ):
	@$(MAKE) -C $(KERNEL_DIR) $(KERNEL_IR_TARGETS)

$(TOOLS_OBJ): $(TOOLS_SRC)
	@mkdir -p $(OBJDIR)
	@echo "Compiling tools.cc"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(ALL_INCLUDES)

$(BINDIR)/test_device_ir_intra: test_device_ir_intra.cpp $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_ir_inter: test_device_ir_inter.cpp $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_api_intra: test_device_api_intra.cpp $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_api_inter: test_device_api_inter.cpp $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)

# No pure unit tests here — these all require MPI + GPU
run-unit:
	@echo "device_api: no unit tests (all require MPI + GPU)"

# Env vars for inter-node testing on single node
HETERO_ENV := -x FLAGCX_USE_HETERO_COMM=1 \
              -x FLAGCX_MEM_ENABLE=1 \
              -x FLAGCX_VMM_ENABLE=0 \
              -x FLAGCX_P2P_DISABLE=1 \
              -x FLAGCX_DEBUG=Version \
              -x FLAGCX_DEBUG_SUBSYS= \
              -x LD_LIBRARY_PATH

# Intra-node tests: safe to run on a single node. The IR binaries are skipped on
# platforms without a device_ir kernel (see IR_TARGETS above).
run-mpi-intra: $(filter %_intra,$(MPI_TARGETS))
	@echo "Running intra-node tests..."
	@for t in $(filter %_intra,$(MPI_TARGETS)); do \
		mpirun -np 8 --allow-run-as-root $$t || exit 1; \
	done

# Inter-node tests: require FLAGCX_USE_HETERO_COMM=1 or actual multi-node
run-mpi-inter: $(filter %_inter,$(MPI_TARGETS))
	@if [ -x $(BINDIR)/test_device_ir_inter ]; then \
		echo "Running inter-node Device IR tests..."; \
		mpirun -np 8 --allow-run-as-root $(HETERO_ENV) $(BINDIR)/test_device_ir_inter || exit 1; \
	fi
	@echo "Running inter-node Device API tests..."
	mpirun -np 8 --allow-run-as-root $(HETERO_ENV) $(BINDIR)/test_device_api_inter -b 1M -e 4M -f 2 -R 2

run-mpi: run-mpi-intra run-mpi-inter

run: run-mpi
