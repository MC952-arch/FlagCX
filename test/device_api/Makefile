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

.PHONY: all clean device_api.o device_ir.o

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o

$(OBJDIR)/device_api.o: device_api.cu device_api.h
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.cu (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) -dc -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.cu device_ir.h
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_ir.cu (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) -dc -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

clean:
	@rm -rf $(BUILDDIR)
