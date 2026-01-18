import json
from tqdm import tqdm
import re
import re
from collections import Counter

# 提取每行的汉字数
def get_line_char_counts(poem_text):
    # 按常见中文标点切分
    lines = re.split(r"[，。！？；：\n]", poem_text)
    counts = []

    for line in lines:
        # 只保留汉字
        chars = re.findall(r"[\u4e00-\u9fff]", line)
        if len(chars) >= 3:  # 太短的行忽略（防止噪声）
            counts.append(len(chars))

    return counts


# 判断诗是几言（返回 None 表示无法判断）
def infer_poem_style(poem_text):
    counts = get_line_char_counts(poem_text)
    if not counts:
        return None

    counter = Counter(counts)
    most_common, freq = counter.most_common(1)[0]

    # 要求至少一半行数一致，才认为可信
    if freq / len(counts) >= 0.6:
        return most_common
    return None

def infer_instruction_style(instruction_text):
    if "五言" in instruction_text:
        return 5
    if "七言" in instruction_text:
        return 7
    if "六言" in instruction_text:
        return 6
    if "四言" in instruction_text:
        return 4
    return None


def clean_text(text):
    return re.sub(r"[^\u4e00-\u9fff，。！？；：\s]", "", text)

def is_valid_text(text, threshold=0.9):
    clean_len = len(re.findall(r"[\u4e00-\u9fff，。！？；：]", text))
    return clean_len / max(len(text),1) >= threshold

input_file = "/home/dsl/learn/poem/new_train.jsonl"
output_file = "/home/dsl/learn/poem/new_train1.jsonl"

cleaned = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="清洗数据"):
        item = json.loads(line)
        output_clean = item["output"]
        instruction_clean = item["instruction"]
        
        if not is_valid_text(output_clean):
            continue  # 丢弃低质量输出
        #这里需要查看是否可以对应上几言和真正的内容
        # 推断几言
        poem_style = infer_poem_style(output_clean)
        instr_style = infer_instruction_style(instruction_clean)

        # 如果 instruction 明确要求几言，但诗无法判断 → 丢弃
        if instr_style is not None and poem_style is None:
            continue

        # 如果二者都存在但不匹配 → 丢弃或记录
        if instr_style is not None and poem_style != instr_style:
            # 可选：统计而不是直接丢弃
            # mismatch_count += 1
            continue
        
        cleaned.append({
            "instruction": instruction_clean,
            "output": output_clean
        })

with open(output_file, "w", encoding="utf-8") as f:
    for item in cleaned:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 清洗完成，总条目数: {len(cleaned)}")

