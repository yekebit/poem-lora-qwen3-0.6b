#!/bin/bash

set -e

# 初始化 Conda（仅一次）
eval "$(conda shell.bash hook)"

echo "🚀 激活 Conda 环境 poem-ft..."
conda activate poem-ft

echo "📦 安装 PyTorch 及相关 CUDA 组件（使用清华镜像）..."
pip install torch torchvision torchaudio \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/pytorch-cu121 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

echo "📦 安装 Hugging Face 及其他 Python 库（使用清华镜像）..."
pip install transformers accelerate datasets peft sentencepiece wandb einops \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

echo "✅ 所有依赖安装完成！"
