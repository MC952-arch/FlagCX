include ../../make.inc

# Auto-detect GPU architectures
ifeq ($(USE_NVIDIA), 1)
include $(PROJECT_ROOT)/makefiles/nvidia_gencode.mk
endif

# Shared device objects (built in test/device_api/)
DEVICE_API_DIR := $(PROJECT_ROOT)/test/device_api
DEVICE_API_OBJ := $(DEVICE_API_DIR)/build/obj/device_api.o
DEVICE_IR_OBJ  := $(DEVICE_API_DIR)/build/obj/device_ir.o

# Shared tools object
TOOLS_SRC := $(PROJECT_ROOT)/test/tools.cc
TOOLS_OBJ := $(OBJDIR)/tools.o

# IR wrapper headers
IR_INCLUDE := -I$(PROJECT_ROOT)/bindings/ir
PLATFORM_INCLUDES := -I$(PLATFORM_IR_DIR)

# All includes
ALL_INCLUDES := $(FLAGCX_INCLUDES) $(IR_INCLUDE) $(PLATFORM_INCLUDES) \
    -I$(DEVICE_API_DIR) $(PERF_INCLUDE) $(MPI_INCLUDE) $(DEVICE_INCLUDE)

ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK)

# Kernel flag
CXXFLAGS += -DCOMPILE_KERNEL

# Test targets
MPI_TARGETS := $(BINDIR)/test_device_api $(BINDIR)/test_device_ir

.PHONY: all clean run-unit run-mpi run

all: $(MPI_TARGETS)

# Build shared CUDA objects
$(DEVICE_API_OBJ):
	@$(MAKE) -C $(DEVICE_API_DIR) device_api.o USE_NVIDIA=$(USE_NVIDIA) USE_DU=$(USE_DU)

$(DEVICE_IR_OBJ):
	@$(MAKE) -C $(DEVICE_API_DIR) device_ir.o USE_NVIDIA=$(USE_NVIDIA) USE_DU=$(USE_DU)

$(TOOLS_OBJ): $(TOOLS_SRC)
	@mkdir -p $(OBJDIR)
	@echo "Compiling tools.cc"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(ALL_INCLUDES)

$(BINDIR)/test_device_api: test_device_api.cpp $(DEVICE_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_API_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_ir: test_device_ir.cpp $(DEVICE_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)

# No pure unit tests here — these all require MPI + GPU
run-unit:
	@echo "device_api: no unit tests (all require MPI + GPU)"

run-mpi: $(MPI_TARGETS)
	@echo "Running Device API correctness tests..."
	mpirun --allow-run-as-root -np 8 $(BINDIR)/test_device_api -b 1M -e 4M -f 2 -R 2
	mpirun --allow-run-as-root -np 8 $(BINDIR)/test_device_ir -b 1M -e 4M -f 2 -R 2

run: run-mpi
