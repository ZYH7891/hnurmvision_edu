#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SIM_ROOT="$REPO_ROOT/simulation"
source /opt/ros/humble/setup.bash
source "$SIM_ROOT/install_interfaces/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-89}

# onnxruntime 的 CUDA 后端依赖 CUDA12 运行时与 cuDNN9（由 pip 的 nvidia-*-cu12 提供）。
# 找不到就自动回退 CPU，不影响运行。
NV="$HOME/.local/lib/python3.10/site-packages/nvidia"
ORT="$HOME/.local/lib/python3.10/site-packages/onnxruntime/capi"
if [ -d "$NV" ]; then
  export LD_LIBRARY_PATH="$ORT:$NV/cudnn/lib:$NV/cublas/lib:$NV/cuda_runtime/lib:$NV/cufft/lib:$NV/curand/lib:${LD_LIBRARY_PATH}"
fi

cd "$REPO_ROOT"
exec ./build/training configs/simulation.yaml
