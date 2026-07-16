# Build: make NVSHMEM_HOME=/path/to/nvshmem CUDA_HOME=/usr/local/cuda
# Run:   nvshmrun -np 2 ./test_nvshmem_adaptor

NVSHMEM_HOME ?= /usr/local/nvshmem
CUDA_HOME ?= /usr/local/cuda

NVCC := $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS := -std=c++17 -O2 -g \
	-DFLAGCX_COMM_TRAITS_NVSHMEM -DUSE_NVIDIA_ADAPTOR \
	-I../../../flagcx/include \
	-I../../../flagcx/adaptor/include \
	-I../../../flagcx/adaptor/shmem \
	-I$(NVSHMEM_HOME)/include \
	-I$(CUDA_HOME)/include \
	-rdc=true

LDFLAGS := -L$(NVSHMEM_HOME)/lib -lnvshmem_device -lnvshmem_host \
	-L$(CUDA_HOME)/lib64 -lcudart -lcuda

TARGET := test_nvshmem_adaptor
SRCS := test_nvshmem_adaptor.cu ../../../flagcx/adaptor/shmem/nvshmem_adaptor.cc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o *.d
