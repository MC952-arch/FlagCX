include ../../make.inc

# Local include directory for kernel test fixtures
LOCAL_INCLUDE := -I$(abspath include)

# device_api.cu object (compiled in test/kernel/)
KERNEL_DIR     := $(PROJECT_ROOT)/test/kernel
DEVICE_API_OBJ := $(KERNEL_DIR)/device_api.o
DEVICE_API_INCLUDE := -I$(KERNEL_DIR)

# [mpi] tests: coll_*.cpp + main_mpi.cpp + flagcx_kernel_test.cpp
MPI_SRCS   := $(wildcard coll_*.cpp) flagcx_kernel_test.cpp main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/kernel_mpi_tests

.PHONY: all clean run-unit run-mpi run device-api-obj

all: $(MPI_TARGET)

# Build device_api.o from test/kernel/ if not already built
device-api-obj:
	@$(MAKE) -C $(KERNEL_DIR) device_api.o USE_NVIDIA=$(USE_NVIDIA)

$(MPI_TARGET): device-api-obj $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(MPI_OBJS) $(DEVICE_API_OBJ) -o $@ $(FLAGCX_LINK) $(GTEST_LINK) $(MPI_LINK)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(DEVICE_API_INCLUDE) $(MPI_INCLUDE)

clean:
	@rm -rf $(BUILDDIR)

run-unit:
	@echo "No unit tests in kernel/"

run-mpi: $(MPI_TARGET)
	@mpirun --allow-run-as-root -np 8 $(MPI_TARGET)

run: run-mpi
