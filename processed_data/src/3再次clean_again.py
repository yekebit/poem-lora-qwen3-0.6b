# aggressive_clean.py
import json
import re

def aggressive_clean(text):
    """彻底清洗文献垃圾"""
    
    # 1. 删除所有括号及其内容（包括嵌套）
    text = re.sub(r'（[^）]*）', '', text)  # 中文括号
    text = re.sub(r'\([^)]*\)', '', text)  # 英文括号
    text = re.sub(r'\[[^\]]*\]', '', text) # 方括号
    
    # 2. 删除文献引用
    text = re.sub(r'《[^》]*》', '', text)  # 书名号
    text = re.sub(r'卷\d+', '', text)      # 卷号
    text = re.sub(r'第.{0,5}[页册章]', '', text)  # 页码
    
    # 3. 删除解释性文字
    text = re.sub(r'此[词诗首句段].*', '', text)  # "此诗/此词..."
    text = re.sub(r'上文.*', '', text)           # "上文..."
    text = re.sub(r'原为.*', '', text)           # "原为..."
    text = re.sub(r'均写.*', '', text)           # "均写..."
    text = re.sub(r'即.*', '', text)             # "即..."
    
    # 4. 删除重复标点
    text = re.sub(r'[，。！？；：]{2,}', '，', text)  # 多个标点→单个
    
    # 5. 删除非诗歌内容
    text = re.sub(r'\d+年\d+月\d+日', '', text)   # 日期
    text = re.sub(r'同前卷\d+', '', text)        # "同前卷二"
    
    # 6. 清理首尾空格和残留字符
    text = text.strip()
    text = re.sub(r'^[,，。！？；：]', '', text)   # 开头标点
    
    return text

# 清洗并检查
with open("/home/dsl/learn/poem/new_train1.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

clean_data = []
removed = []

for item in raw_data:
    original = item['output']
    cleaned = aggressive_clean(original)
    
    # 过滤：清洗后太短或太长的
    if 20 < len(cleaned) < 200:
        item['output'] = cleaned
        clean_data.append(item)
    else:
        removed.append((original, cleaned))

print(f"原始数据: {len(raw_data)} 条")
print(f"清洗后: {len(clean_data)} 条")
print(f"删除: {len(removed)} 条（过长/过短/污染）")

# 查看被删除的样本（确认清洗正确）
print("\n【被删除的样本示例】")
for i, (orig, clean) in enumerate(removed[:3]):
    print(f"\n--- 样本 {i+1} ---")
    print(f"原文: {orig[:100]}...")
    print(f"清洗后: {clean[:100]}...")

# 保存清洗数据
with open("/home/dsl/learn/poem/new_train2.jsonl", "w", encoding="utf-8") as f:
    for item in clean_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
