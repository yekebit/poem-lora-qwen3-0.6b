# 唐诗生成模型 (Qwen3-0.6B + LoRA + DPO)

## 项目简介

本项目旨在通过微调 Qwen3-0.6B 模型，实现高质量的唐诗生成。项目采用了三个阶段的训练策略，最终通过引入 DPO (Direct Preference Optimization) 技术，显著提升了模型生成唐诗的质量和准确性。

## 训练过程

### 第一阶段：基础 LoRA 微调
- **数据**：使用原始唐诗数据集
- **方法**：直接应用 LoRA 技术进行微调
- **结果**：生成质量较差，不符合唐诗的韵律和风格要求

### 第二阶段：数据优化
- **数据**：对原始数据进行清洗和优化
- **方法**：改进数据处理流程，重新进行 LoRA 微调
- **结果**：效果有所提升，但仍不理想

### 第三阶段：引入 DPO
- **数据**：构造 DPO 偏好数据集（包含优质回答和劣质回答）
- **方法**：使用 TRL 库的 DPOTrainer 进行训练
- **结果**：生成质量显著提升，能够更好地符合唐诗的韵律和风格

## 技术栈

- **基础模型**：Qwen3-0.6B
- **微调技术**：LoRA (Low-Rank Adaptation)
- **优化技术**：DPO (Direct Preference Optimization)
- **框架**：Transformers, PEFT, TRL
- **硬件**：GPU 加速

## 项目结构

```
poem-lora-qwen3-0.6b/
├── train.py              # 基础 LoRA 训练脚本
├── train-new.py          # 第二阶段采用新数据的训练脚本
├── train-dpo.py          # DPO 训练脚本(final版本)
├── inference_and_eval.py # 模型评估脚本
├── processed_data/       # 数据处理
│   ├── src/              # 数据处理脚本
│   ├── data_drop/        # 第一阶段数据
│   ├── data_new/         # 第二阶段数据
│   └── data_dpo/         # 第三阶段 DPO 数据
├── output/               # 训练结果
│   ├── qwen3-poem-lora/      # 第一阶段模型
│   ├── qwen3-poem-lora-new/  # 第二阶段模型
│   └── qwen3-poem-dpo/       # 第三阶段模型
└── chinese-poetry-master/ # 原始唐诗数据集
```

## 数据处理

1. **原始数据**：来自 chinese-poetry-master 仓库的唐诗数据集
2. **数据清洗**：
   - 去除噪声数据
   - 标准化格式
   - 过滤低质量样本
3. **DPO 数据构造**：
   - 为每个指令生成优质回答（正确的唐诗）
   - 为每个指令生成劣质回答（不符合要求的唐诗）
   - 构建偏好数据集

## 模型训练

### 基础 LoRA 训练

```python
# 核心参数
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    # 其他参数...
)
```

### DPO 训练

```python
# 核心参数
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,          # DPO 显存需求更高
    gradient_accumulation_steps=8,
    learning_rate=1e-5,                    # DPO 通常使用更小的学习率
    num_train_epochs=1,                    # DPO 1 epoch 通常足够
    beta=0.1,                              # DPO 核心超参：偏好强度
    # 其他参数...
)
```

## 模型评估

项目提供了 `inference_and_eval.py` 脚本，用于对比基础模型和微调后模型的生成效果。评估指标包括：

- 韵律一致性
- 风格符合度
- 主题相关性
- 整体质量

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据处理

```bash
# 处理原始唐诗数据
python processed_data/src/1处理不带tag的唐诗.py

# 初步清洗
python processed_data/src/2初步clean.py

# 再次清洗
python processed_data/src/3再次clean_again.py

# 划分训练集和验证集
python processed_data/src/4划分split.py

# 构造 DPO 偏好数据
python processed_data/src/5DPO数据构造.py
```

### 3. 模型训练

```bash
# 第一阶段：基础 LoRA 训练
python train.py

# 第二阶段：使用优化后的数据重新训练
# 修改 train.py 中的数据路径后运行
python train.py

# 第三阶段：DPO 训练
python train-dpo.py
```

### 4. 模型评估

```bash
# 评估模型效果
python inference_and_eval.py
```

## 生成示例

### 输入
```
写一首[五言绝句] 要求4句 主题为: 山水、自然、宁静
```

### 输出（第三阶段模型）
```
空山新雨后，天气晚来秋。
明月松间照，清泉石上流。
```

## 总结与反思

1. **数据质量至关重要**：第一阶段和第二阶段的差异表明，数据清洗和优化对模型性能有显著影响。

2. **DPO 技术的有效性**：第三阶段引入 DPO 后，模型生成质量得到显著提升，证明了偏好优化在诗生成任务中的有效性。

3. **参数调整的重要性**：不同阶段的训练参数需要根据任务特点进行调整，特别是 DPO 训练需要更小的学习率和批处理大小。

4. **未来改进方向**：
   - 进一步扩大和优化训练数据
   - 尝试不同的基础模型
   - 探索更高级的微调技术
   - 引入更多的评估指标

## 参考资料

- [Qwen3 模型](https://github.com/QwenLM/Qwen3)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [chinese-poetry 数据集](https://github.com/chinese-poetry/chinese-poetry)

## 许可证

本项目采用 MIT 许可证。
