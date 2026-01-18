import os
import json
from opencc import OpenCC

cc = OpenCC("t2s")  # 繁体 -> 简体

base_dir = "/home/dsl/learn/poem/chinese-poetry-master/全唐诗"

for fname in os.listdir(base_dir):
    if not fname.startswith("poet.tang.") or not fname.endswith(".json"):
        continue
    # if not fname.startswith("唐诗三百首"):
    #     continue
    file_path = os.path.join(base_dir, fname)
    print(f"处理文件: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)   # data 是一个 list

    for item in data:
        # author
        if "author" in item and isinstance(item["author"], str):
            item["author"] = cc.convert(item["author"])

        # title
        if "title" in item and isinstance(item["title"], str):
            item["title"] = cc.convert(item["title"])

        # paragraphs
        if "paragraphs" in item and isinstance(item["paragraphs"], list):
            item["paragraphs"] = [
                cc.convert(line) if isinstance(line, str) else line
                for line in item["paragraphs"]
            ]

    # 原地覆盖写回
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)