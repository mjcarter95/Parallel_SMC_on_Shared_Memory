#!/bin/bash -l
#SBATCH -J smc_h100
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -o %j.out
#SBATCH --export=ALL

module purge
module load llvm/19.1.3-gcc14.2.0
module load cuda/12.8.0-gcc14.2.0

cd /users/mcarter/fastscratch/Parallel_SMC_on_Shared_Memory/GPU_Code

LLVM_LIBDIR="/opt/apps/pkg/applications/spack_apps/v0231_apps/linux-rocky9-x86_64_v3/gcc-14.2.0/llvm-19.1.3-z37ezwxeua32zof7s6rsnpscg5rdejvu/bin/../lib/x86_64-unknown-linux-gnu"
export LD_LIBRARY_PATH="${LLVM_LIBDIR}:${LD_LIBRARY_PATH}"

export OMP_TARGET_OFFLOAD=MANDATORY

export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}"
export GPU_ARCH="${GPU_ARCH:-sm_90}"

echo "=============================================================="
echo " Job running on node(s): $SLURM_NODELIST"
echo " GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi -q -d PERFORMANCE,POWER,CLOCK,COMPUTE
echo "--------------------------------------------------------------"
echo " Compiler: $(clang++ --version | head -n 1)"
echo " CUDA_HOME: $CUDA_HOME"
echo " GPU_ARCH:  $GPU_ARCH"
echo " CUDA version: $(nvcc --version | grep release)"
echo " LLVM libdir: $LLVM_LIBDIR"
echo " LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo " Start time: $(date)"
echo "=============================================================="

make clean
make all GPU_ARCH="$GPU_ARCH" CUDA_HOME="$CUDA_HOME"

./smc_gpu 1 20 10 1 10

echo "=============================================================="
echo " End time: $(date)"
echo "=============================================================="
