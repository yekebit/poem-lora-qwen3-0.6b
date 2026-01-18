import json
import random

random.seed(42)

data = []
with open("/home/dsl/learn/poem/processed_data/data_dpo/dpo_preference_data_clean.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

random.shuffle(data)

split = int(len(data) * 0.95)
train_data = data[:split]
val_data = data[split:]

with open("processed_data/data_dpo/train.jsonl", "w", encoding="utf-8") as f:
    for x in train_data:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with open("processed_data/data_dpo/val.jsonl", "w", encoding="utf-8") as f:
    for x in val_data:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")