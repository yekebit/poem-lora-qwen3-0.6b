import os
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# =========================
# 0. 路径设置
# =========================
MODEL_PATH = "/home/dsl/learn/poem/qwen3-0_6b"
TRAIN_FILE = "/home/dsl/learn/poem/processed_data/data_new/train.jsonl"
VAL_FILE = "/home/dsl/learn/poem/processed_data/data_new/val.jsonl"
OUTPUT_DIR = "/home/dsl/learn/poem/output/qwen3-poem-lora-new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. 加载 tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# 2. Prompt 构造函数
# =========================
def build_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

# =========================
# 3. 数据集加载 & tokenize
# =========================
print("加载数据集...")
dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE}
)

def tokenize_fn(example):
    text = build_prompt(example)
    tokens = tokenizer(text, truncation=True, max_length=512, padding=False)
    # tokens["labels"] = tokens["input_ids"].copy()  # 训练目标就是生成 output
    # import pdb;pdb.set_trace()
    return tokens

print("Tokenizing 数据集...")
tokenized_dataset = dataset.map(
    tokenize_fn,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing",
    batched=False
)

print(f"训练集大小: {len(tokenized_dataset['train'])}, 验证集大小: {len(tokenized_dataset['validation'])}")

# =========================
# 4. 加载模型
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# =========================
# 5. LoRA 配置
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 6. 训练参数
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    eval_strategy="steps",  # 每隔 eval_steps 评估一次
    eval_steps=500,               # 每 500 steps 做一次验证
    save_strategy="steps",        # 保存策略与 evaluation 对齐
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    report_to="none",
    load_best_model_at_end=True
)

# =========================
# 7. 数据整理器
# =========================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# =========================
# 8. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator
)

# =========================
# 9. 训练并记录 loss
# =========================
train_history = trainer.train()
metrics = train_history.metrics
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# =========================
# 10. 可视化 loss (修复版)
# =========================

# 从 trainer 的日志历史中提取 loss
log_history = trainer.state.log_history

# 训练 loss（每50步记录一次）
train_logs = [log for log in log_history if "loss" in log and "eval_loss" not in log]
train_steps = [log["step"] for log in train_logs]
train_losses = [log["loss"] for log in train_logs]

# 验证 loss（每500步记录一次）
eval_logs = [log for log in log_history if "eval_loss" in log]
eval_steps = [log["step"] for log in eval_logs]
eval_losses = [log["eval_loss"] for log in eval_logs]

# 绘制
plt.figure(figsize=(10, 6))

if train_losses:
    plt.plot(train_steps, train_losses, label="Train Loss", marker='o', linestyle='-')

if eval_losses:
    plt.plot(eval_steps, eval_losses, label="Eval Loss", marker='s', linestyle='--')

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("LoRA Fine-tuning Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Loss 曲线已保存到 {OUTPUT_DIR}/loss_curve.png")
