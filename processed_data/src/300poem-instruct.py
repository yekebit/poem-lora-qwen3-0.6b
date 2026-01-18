import json
import re

# 主题关键词映射（用于 fallback）
THEME_KEYWORDS = {
    "月亮": ["月", "玉盘", "桂魄", "婵娟", "明月"],
    "思乡": ["乡", "归", "家", "故园", "客", "思乡"],
    "春天": ["春", "花", "柳", "燕", "莺", "芳草"],
    "梅花": ["梅", "寒梅", "雪中梅"],
    "山水": ["山", "水", "江", "河", "湖", "云", "峰"],
    "离别": ["别", "送", "离", "分", "泪", "相送"],
    "饮酒": ["酒", "醉", "杯", "饮", "樽", "酌"],
    "边塞": ["塞", "关", "胡", "征人", "烽火", "边城"],
    "怀古": ["古", "昔", "往事", "怀古", "旧时"],
    "爱情": ["情", "相思", "爱", "心", "泪", "梦"],
    "宫怨": ["宫", "深宫", "寂寞", "空房", "君王"],
}

# 从 tags 中识别的主题白名单（可扩展）
TAG_THEME_MAP = {
    "思乡": "思乡",
    "边塞": "边塞",
    "宫怨": "宫怨",
    "怀古": "怀古",
    "咏物": "咏物",
    "山水": "山水",
    "爱情": "爱情",
    "闺怨": "宫怨",      # 归入宫怨
    "离别": "离别",
    "饮酒": "饮酒",
    "春天": "春天",
    "秋天": "秋天",
    "冬天": "冬天",
    "夏天": "夏天",
    "战争": "边塞",      # 战争多属边塞诗
    "爱国": "边塞",      # 多与边塞重合
    "悼亡": "爱情",      # 可单独保留，此处简化
}

def extract_form_from_tags(tags):
    """从 tags 中提取体裁（优先级：绝句 > 律诗）"""
    for tag in tags:
        if "言" in tag:
            return tag
    return None

def extract_theme_from_tags(tags):
    """从 tags 中提取主题（优先使用人工标注）"""
    tag_theme = []
    number = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
    for tag in tags:
        if not any(num in tag for num in number):
            tag_theme.append(tag)
    return "、".join(tag_theme) if len(tag_theme)!=0 else None
    

def infer_theme_fallback(title, paragraphs):
    """fallback：从文本关键词匹配主题"""
    text = title + "".join(paragraphs)
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return theme
    return "某个主题"

def process_poem(poem):
    title = poem.get("title", "")
    paragraphs = poem.get("paragraphs", [])
    tags = poem.get("tags", [])

    if not paragraphs or len("".join(paragraphs)) < 10:
        return None

    # 1. 提取体裁
    form = extract_form_from_tags(tags)
    if not form:
        text = "".join(paragraphs)
        clean_text = re.sub(r"[，。！？；：\s]", "", text)
        total_chars = len(clean_text)
        num_lines = len(paragraphs)
        if num_lines == 4:
            if total_chars == 20:
                form = "五言绝句"
            elif total_chars == 28:
                form = "七言绝句"
        elif num_lines == 8:
            if total_chars == 40:
                form = "五言律诗"
            elif total_chars == 56:
                form = "七言律诗"
        else:
            form = "唐诗"

    # 2. 提取主题（优先用 tags）
    theme = extract_theme_from_tags(tags)
    if not theme:
        theme = infer_theme_fallback(title, paragraphs)

    instruction = f"写一首关于{theme}的{form}"
    output = "".join(paragraphs)

    

    def clean_instruction(instruction, output):
        # 替换 None
        if "None" in instruction:
            instruction = instruction.replace("None", "唐诗")
        
        # 仅当 instruction 中没有“言”字（即体裁未明确）时，才推断
        if "言" not in instruction:
            # 去除标点，获取第一句纯文字
            first_line = output.split("，")[0].split("。")[0].split("？")[0].split("！")[0]
            clean_first_line = re.sub(r"[，。？！；：\s]", "", first_line)
            char_count = len(clean_first_line)
            
            # 推断体裁
            if char_count == 5:
                form = "五言唐诗"
            elif char_count == 7:
                form = "七言唐诗"
            else:
                form = "唐诗"  # 兜底
            
            # 替换“唐诗”为具体体裁
            instruction = instruction.replace("唐诗", form)
        
        return instruction
    return {
        "instruction": clean_instruction(instruction,output),
        "output": output
    }

def main():
    input_file = "/home/dsl/learn/poem/chinese-poetry-master/全唐诗/唐诗三百首.json"
    output_file = "poem_train.jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        poems = json.load(f)

    valid_count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for poem in poems:
            item = process_poem(poem)
            number = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
            if item and any( num in item["instruction"] for num in number):
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                valid_count += 1

    print(f"✅ 已处理 {len(poems)} 首诗，成功生成 {valid_count} 条训练样本，保存至 {output_file}")

if __name__ == "__main__":
    main()