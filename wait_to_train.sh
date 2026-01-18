#!/bin/bash

# 配置
TRAIN_SCRIPT="/home/dsl/learn/poem/train-new.py"
CONDA_ENV="poem-ft"
TARGET_GPU=1                # 监控 GPU 1
MAX_USED_MEM_MB=4000        # 已用显存阈值：4GB（即剩余 > 20GB）
CHECK_INTERVAL=60           # 检查间隔（秒）

echo "🔍 监控 GPU $TARGET_GPU，等待已用显存 < ${MAX_USED_MEM_MB}MB（剩余 > 20GB）..."

while true; do
    # 获取所有 GPU 的已用显存（单位 MB）
    gpu_used_mems=($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits))
    
    # 检查 GPU 1 是否存在
    if [ ${#gpu_used_mems[@]} -le $TARGET_GPU ]; then
        echo "❌ 系统中没有 GPU $TARGET_GPU！"
        exit 1
    fi

    current_used=${gpu_used_mems[$TARGET_GPU]}
    echo "$(date): GPU $TARGET_GPU 已用显存: ${current_used} MB"

    # 判断是否满足条件
    if [ "$current_used" -lt "$MAX_USED_MEM_MB" ]; then
        echo "✅ GPU $TARGET_GPU 剩余显存 > 20GB，开始训练！"
        
        # 激活 Conda 环境并运行训练脚本（指定使用 GPU 1）
        eval "$(conda shell.bash hook)"
        conda activate $CONDA_ENV
        CUDA_VISIBLE_DEVICES=$TARGET_GPU python $TRAIN_SCRIPT
        
        echo "🎉 训练完成！"
        break
    fi

    sleep $CHECK_INTERVAL
done