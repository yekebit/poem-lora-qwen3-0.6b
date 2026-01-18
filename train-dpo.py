import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig  # 👈 新增 DPO 组件

# =========================
# 0. 路径设置
# =========================
MODEL_PATH = "/home/dsl/learn/poem/qwen3-0_6b"
TRAIN_FILE = "/home/dsl/learn/poem/processed_data/data_dpo/train.jsonl"  # 👈 DPO 偏好数据
OUTPUT_DIR = "/home/dsl/learn/poem/output/qwen3-poem-dpo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. 加载 tokenizer
# =========================
print("🔧 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# 2. 构建 prompt（仅用于 DPO 输入）
# =========================
def build_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

# =========================
# 3. 加载 DPO 偏好数据集
# =========================
print("📂 加载 DPO 偏好数据集...")
dataset = load_dataset("json", data_files={"train": TRAIN_FILE})

# 构造 DPO 所需的三列: prompt, chosen, rejected
def preprocess(example):
    return {
        "prompt": build_prompt(example["instruction"]),
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dpo_dataset = dataset["train"].map(
    preprocess,
    remove_columns=dataset["train"].column_names,  # 移除原始列
    desc="构建 DPO 格式"
)

print(f"✅ DPO 数据集大小: {len(dpo_dataset)}")

# =========================
# 4. 加载基础模型
# =========================
print("🧠 加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# =========================
# 5. LoRA 配置（与 SFT 相同）
# =========================
print("🧩 配置 LoRA 微调...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 6. DPO 训练参数（关键调整！）
# =========================
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,          # 👈 DPO 显存高，batch_size 减小
    gradient_accumulation_steps=8,          # 等效 batch_size=16
    learning_rate=1e-5,                    # 👈 DPO 通常用更小 lr
    num_train_epochs=1,                    # 👈 DPO 1 epoch 通常足够
    beta=0.1,                              # 👈 DPO 核心超参：偏好强度
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    bf16=True,
    report_to="none",
        max_length=256,                        # 总长度 (prompt + response)
    max_prompt_length=128
)

# =========================
# 7. DPO Trainer（替代原 Trainer）
# =========================
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,                        # 👈 自动使用当前 model 作为 reference
    args=training_args,
    train_dataset=dpo_dataset,
    processing_class=tokenizer                  # prompt 最大长度
)

# =========================
# 8. 开始 DPO 训练
# =========================
print("🚀 开始 DPO 训练...")
dpo_trainer.train()

# =========================
# 9. 保存模型
# =========================
dpo_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ DPO 模型已保存到 {OUTPUT_DIR}")