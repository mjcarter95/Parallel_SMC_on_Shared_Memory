# ===============================================================
# Makefile for Parallel SMC Sampler on GPU
# ===============================================================
# Usage:
#   make
#   make clean
# ===============================================================

CXX := clang++

CUDA_HOME ?= $(CUDA_PATH)
CUDA_HOME ?= $(CUDAHOME)
CUDA_HOME ?= $(shell dirname $$(dirname $$(which nvcc)) 2>/dev/null)

GPU_ARCH ?= sm_80

OMP_OFFLOAD_FLAGS := -fopenmp \
  -fopenmp-targets=nvptx64-nvidia-cuda \
  -Xopenmp-target=nvptx64-nvidia-cuda \
  --cuda-path=$(CUDA_HOME) \
  -march=$(GPU_ARCH)

CXXFLAGS := -O3 -std=c++17 -I . $(OMP_OFFLOAD_FLAGS)
LDFLAGS  := -fopenmp

SRCS := main.cpp \
        smc.cpp \
        importance_sampling.cpp \
        resampling.cpp \
        redistribution.cpp \
        prefix_reduction_operations.cpp \
        reduction_operations.cpp \
        model.cpp \
        random.cpp \
		binary_search.cpp

TARGET := smc_gpu

T      ?= 1
N      ?= 10
K      ?= 10
REDIST ?= 0
MC     ?= 10

all:
	@echo "Compiling $(TARGET) with clang OpenMP offload..."
	@echo "  CUDA_HOME=$(CUDA_HOME)"
	@echo "  GPU_ARCH=$(GPU_ARCH)"
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: ./$(TARGET)"

clean:
	@echo "Removing executable..."
	rm -f $(TARGET)
	@echo "Done."
