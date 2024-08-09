#!/bin/bash

# 指定不需要删除的环境名称
EXCLUDE_ENVS=("base" "open3d37" "hjp_ddit38" "hjp_ddit38" "hjp_diffusionsdf" "hjp_diffusionsdfnew" "hjp_diffusionsdfnew_dimr")

# 获取所有环境名称
ALL_ENVS=$(conda env list | awk '{print $1}' | tail -n +4)

# 删除所有不在 EXCLUDE_ENVS 列表中的环境
for env in $ALL_ENVS; do
    if [[ ! " ${EXCLUDE_ENVS[@]} " =~ " ${env} " ]]; then
        echo "Deleting environment: $env"
        conda remove --name "$env" --all -y
    else
        echo "Skipping environment: $env"
    fi
done

echo "Environment cleanup completed."
