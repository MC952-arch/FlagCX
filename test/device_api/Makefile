# Shared device kernel objects (device_api.cu, device_ir.cu).
# Other test Makefiles invoke:
#   make -C $(PROJECT_ROOT)/test/device_api device_api.o
#   make -C $(PROJECT_ROOT)/test/device_api device_ir.o
#
# Objects are placed in build/obj/ and headers live alongside the .cu files.

include ../make.inc

# Auto-detect GPU architectures
ifeq ($(USE_NVIDIA), 1)
include $(PROJECT_ROOT)/makefiles/nvidia_gencode.mk
endif

# Propagate FORCE_DEFAULT_PATH to device compilation (must match library build)
ifeq ($(FORCE_DEFAULT_PATH), 1)
ADAPTOR_DEFINE += -DFLAGCX_FORCE_DEFAULT_PATH
endif

PLATFORM_INCLUDES := -I$(PLATFORM_IR_DIR)

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/bindings/ir \
    $(PLATFORM_INCLUDES) \
    $(CCL_INCLUDE) \
    $(DEVICE_INCLUDE)

# NVSHMEM requires relocatable device code (-dc) + device-link step
ifeq ($(USE_SHMEM), 1)
NVCC_RDC_FLAG := -dc
else
NVCC_RDC_FLAG := -c
endif

.PHONY: all clean device_api.o device_ir.o dlink.o dlink_api.o dlink_ir.o

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o

$(OBJDIR)/device_api.o: device_api.cu device_api.h
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.cu (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) $(NVCC_RDC_FLAG) -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.cu device_ir.h
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_ir.cu (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) $(NVCC_RDC_FLAG) -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

# Device-link step: required when USE_SHMEM=1 to resolve NVSHMEM device symbols.
# Split into separate targets so consumers link only what they need.
ifeq ($(USE_SHMEM), 1)
SHMEM_DEVICE_LIB := $(SHMEM_HOME)/lib/libnvshmem_device.a

# Individual dlinks for consumers that only need one .cu
$(OBJDIR)/dlink_api.o: $(OBJDIR)/device_api.o
	@echo "Device-linking device_api (NVSHMEM)"
	@$(DEVICE_COMPILER) $(DEVICE_COMPILER_GENCODE) -dlink -o $@ $^ \
		$(SHMEM_DEVICE_LIB) -lcudart -lcudadevrt

$(OBJDIR)/dlink_ir.o: $(OBJDIR)/device_ir.o
	@echo "Device-linking device_ir (NVSHMEM)"
	@$(DEVICE_COMPILER) $(DEVICE_COMPILER_GENCODE) -dlink -o $@ $^ \
		$(SHMEM_DEVICE_LIB) -lcudart -lcudadevrt

# Combined dlink for consumers that use both device_api.o and device_ir.o
$(OBJDIR)/dlink.o: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o
	@echo "Device-linking all (NVSHMEM)"
	@$(DEVICE_COMPILER) $(DEVICE_COMPILER_GENCODE) -dlink -o $@ $^ \
		$(SHMEM_DEVICE_LIB) -lcudart -lcudadevrt

dlink.o: $(OBJDIR)/dlink.o
dlink_api.o: $(OBJDIR)/dlink_api.o
dlink_ir.o: $(OBJDIR)/dlink_ir.o

all: $(OBJDIR)/dlink.o
endif

clean:
	@rm -rf $(BUILDDIR)
