#!/bin/bash

## use under huju account with an a40 node

module load cuda/11.3.0
module load cudnn/v8.2.1.32
module load compiler/gcc-10.1

# 指定你下载的 ninja binary 文件的路径
NINJA_BINARY_PATH="/storage/user/huju/transferred2/ws_dditnach/DDIT_thirdparty/ninja"

# 将 ninja binary 文件所在的目录添加到 PATH 环境变量的前面
export PATH="$(dirname "$NINJA_BINARY_PATH"):$PATH"

echo "Using custom ninja from $NINJA_BINARY_PATH"

export LD_LIBRARY_PATH=/usr/stud/huju/miniconda3/envs/diffusionsdf/lib/python3.9/site-packages/torch/lib/:$LD_LIBRARY_PATH
