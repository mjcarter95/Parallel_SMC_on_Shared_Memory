# ===============================================================
# Makefile for Parallel SMC Sampler (CPU Version)
# ===============================================================
# Author: Alessandro Varsi
# Description:
#   Builds and runs the OpenMP (CPU) implementation of the
#   Parallel SMC Sampler for variable-size samples.
#
# Usage:
#   make                 # compile only
#   make run T=8 N=15 K=10 REDIST=1   # compile and run with args
#   make clean           # remove executable
#
# ===============================================================

# Compiler settings
CXX      := g++
CXXFLAGS := -O3 -fopenmp -I .
LDFLAGS  := -fopenmp

# Source files
SRCS := main.cpp \
        smc.cpp \
        importance_sampling.cpp \
        resampling.cpp \
        redistribution.cpp \
        prefix_reduction_operations.cpp \
        reduction_operations.cpp \
        model.cpp \
        random.cpp \
        prefix_sum_load_balancing.cpp

# Output executable
TARGET := smc_cpu

# Default runtime arguments (can be overridden)
T      ?= 4       # Number of OpenMP threads
N      ?= 10      # log2 of number of particles (N = 2^log2N)
K      ?= 5       # Number of SMC iterations
REDIST ?= 1       # Redistribution method (0 = naive, 1 = optimal)

# ===============================================================
# Build and run rules
# ===============================================================

all:
	@echo "Compiling $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: ./$(TARGET)"

run: all
	@echo "Running with parameters: T=$(T), N=$(N), K=$(K), REDIST=$(REDIST)"
	./$(TARGET) $(T) $(N) $(K) $(REDIST)

clean:
	@echo "Removing executable..."
	rm -f $(TARGET)
	@echo "Done."

# ===============================================================
# End of Makefile
# ===============================================================
