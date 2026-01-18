import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ======================================================
# 1. 配置
# ======================================================
BASE_MODEL_PATH = "/home/dsl/learn/poem/qwen3-0_6b"
LORA_PATH = "/home/dsl/learn/poem/output/qwen3-poem-lora-new"
VAL_FILE = "/home/dsl/learn/poem/processed_data/data_dpo/val.jsonl"

NUM_SAMPLES = 5
MAX_NEW_TOKENS = 256
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 2. 固定随机性（科研必备）
# ======================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ======================================================
# 3. tokenizer（共享即可）
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# 4. ❗加载两个完全独立的模型实例
# ======================================================

print("🧠 加载 Base Model（纯原始）...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
base_model.eval()

print("🧩 加载 LoRA Model（Base + Adapter）...")
lora_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
lora_model = PeftModel.from_pretrained(lora_base, LORA_PATH)
lora_model.eval()

# ======================================================
# 5. Prompt（与训练严格一致）
# ======================================================
def build_prompt(instruction):
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )

# ======================================================
# 6. 读取验证集
# ======================================================
with open(VAL_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.seed(SEED)
samples = random.sample(data, min(NUM_SAMPLES, len(data)))

# ======================================================
# 7. 生成参数（统一）
# ======================================================
GEN_KWARGS = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True
)

def decode_new(outputs, input_len):
    text = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )
    return text.split("###")[0].strip()

# ======================================================
# 8. 对比生成
# ======================================================
print("\n" + "=" * 100)
print("🎨 Base vs LoRA 生成对比（修复版）")
print("=" * 100)

for i, sample in enumerate(samples, 1):
    instruction = sample["instruction"]
    gt = sample["output"]

    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]

    # Base
    set_seed(SEED)
    with torch.no_grad():
        out_base = base_model.generate(**inputs, **GEN_KWARGS)
    base_text = decode_new(out_base, input_len)

    # LoRA
    set_seed(SEED)
    with torch.no_grad():
        out_lora = lora_model.generate(**inputs, **GEN_KWARGS)
    lora_text = decode_new(out_lora, input_len)

    print(f"\n【样本 {i}】")
    print(f"📌 指令：{instruction}")
    print(f"✅ 真实：{gt}")
    print(f"🔵 Base：{base_text}")
    print(f"🟢 LoRA：{lora_text}")
    print("-" * 100)

print("\n✅ 对比完成（现在是**真实有效的对比**）")