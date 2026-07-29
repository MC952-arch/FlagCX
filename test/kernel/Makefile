# test/kernel/Makefile
# Dispatches kernel compilation to the active platform subdirectory.

include ../make.inc

KERNEL_PLATFORM_DIR := $(CURDIR)/$(KERNEL_PLATFORM)

.PHONY: all clean

all:
	@$(MAKE) -C $(KERNEL_PLATFORM_DIR) all \
		USE_SHMEM=$(USE_SHMEM) SHMEM_HOME=$(SHMEM_HOME) \
		FORCE_DEFAULT_PATH=$(FORCE_DEFAULT_PATH)

clean:
	@$(MAKE) -C $(KERNEL_PLATFORM_DIR) clean

# Pass through any specific target (device_api.o, dlink_ir.o, etc.)
%:
	@$(MAKE) -C $(KERNEL_PLATFORM_DIR) $@ \
		USE_SHMEM=$(USE_SHMEM) SHMEM_HOME=$(SHMEM_HOME) \
		FORCE_DEFAULT_PATH=$(FORCE_DEFAULT_PATH)
