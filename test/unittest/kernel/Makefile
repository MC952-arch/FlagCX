include ../../make.inc

# Local include directories for test fixtures
LOCAL_INCLUDE := -I$(abspath include)

# device_api.cu object (compiled in test/kernel/build/)
KERNEL_DIR     := $(PROJECT_ROOT)/test/kernel
DEVICE_API_OBJ := $(KERNEL_DIR)/build/device_api.o
DEVICE_API_INCLUDE := -I$(KERNEL_DIR)

# Platform-aware device library linking
DEVICE_HOME ?=
DEVICE_LINK =
ifeq ($(USE_NVIDIA), 1)
DEVICE_HOME = /usr/local/cuda
DEVICE_LINK = -L$(DEVICE_HOME)/lib64 -lcudart -lcuda
else ifeq ($(USE_DU), 1)
DEVICE_HOME = $(CUDA_PATH)
DEVICE_LINK = -L$(DEVICE_HOME)/lib64 -lcudart -lcuda
else
# Default: NVIDIA
DEVICE_HOME = /usr/local/cuda
DEVICE_LINK = -L$(DEVICE_HOME)/lib64 -lcudart -lcuda
endif

# Environment variables to force hetero mode on single node
HETERO_ENV := FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0 FLAGCX_P2P_DISABLE=1

# [unit] tests: test_*.cpp — no MPI, no GPU required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/kernel_unit_tests

# [mpi] tests: coll_*.cpp + fixture sources + main_mpi.cpp — requires MPI + GPUs
MPI_SRCS   := $(wildcard coll_*.cpp) flagcx_kernel_test.cpp deviceapi_test.cpp main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/kernel_mpi_tests

.PHONY: all clean run-unit run-mpi run device-api-obj

# Only build targets that have sources
ALL_TARGETS :=
ifneq ($(UNIT_SRCS),)
ALL_TARGETS += $(UNIT_TARGET)
endif
ifneq ($(MPI_SRCS),)
ALL_TARGETS += $(MPI_TARGET)
endif

all: $(ALL_TARGETS)

# Build device_api.o from test/kernel/ if not already built
device-api-obj:
	@$(MAKE) -C $(KERNEL_DIR) device_api.o USE_NVIDIA=$(USE_NVIDIA) USE_DU=$(USE_DU)

$(UNIT_TARGET): $(UNIT_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK)

$(MPI_TARGET): device-api-obj $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(MPI_OBJS) $(DEVICE_API_OBJ) -o $@ $(FLAGCX_LINK) $(GTEST_LINK) $(MPI_LINK) $(DEVICE_LINK)

$(OBJDIR)/test_%.o: test_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(LOCAL_INCLUDE)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(DEVICE_API_INCLUDE) $(MPI_INCLUDE)

clean:
	@rm -rf $(BUILDDIR)

run-unit: $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi: $(MPI_TARGET)
	@$(HETERO_ENV) mpirun --allow-run-as-root -np 4 $(MPI_TARGET)

run: run-unit run-mpi
