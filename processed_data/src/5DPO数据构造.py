import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from utils import run_llm_inference

# ========================
# DPO 反例生成 Prompt
# ========================
REJECTED_PROMPT_TEMPLATE = """你是一个故意不遵守指令的诗人。请根据以下要求，生成一首**不符合指令**的唐诗。

要求：
1. 故意违反体裁（如要求五言却写七言，或反之）
2. 故意违反句数（如要求4句却写2句或8句以上）
3. 故意偏离主题（如要求“山水”却写“战争”或现代生活）
4. 语言现代、不押韵、逻辑混乱、用词白话
5. 可以包含乱码、重复、无意义句子

以下是几个高质量示例：

指令：写一首[五言绝句] 要求2句 主题为: 怀人、哀悼、文人
正确输出：八韵与五字，俱为时所先。幽魂应自慰，李白墓相连。
错误输出（你要模仿的风格）：今天好想你啊，李白你在那边还好吗？我昨天去看了你的墓，感觉特别伤心。现代人真不懂古诗，只会发朋友圈。

指令：写一首[七言律诗] 要求4句 主题为: 孤洁、隐逸、自然、兴亡之思
正确输出：暖傍渔船睡不惊，可怜孤洁似华亭。晚来湾浦冲平碧，晴过汀洲拂浅青。翡翠静中修羽翼，鸳鸯闲处事仪形。何如飞入汉宫里，留与兴亡作典经。
错误输出（你要模仿的风格）：隐居好累啊，WiFi信号太差了。我想回城里打游戏，山里的蚊子太多了。古代人真傻，干嘛要隐居，不如去上班。

指令：写一首[七言绝句] 要求2句 主题为: 咏物、山水画、文人风流
正确输出：小山破体闲支策，落日梨花照空壁。诗堪记室妬风流，画与将军作勍敌。
错误输出（你要模仿的风格）：这幅山水画不错，挂在客厅挺好看。文人风流就是装逼，还不如我刷抖音。

现在请处理以下指令：
指令：{instruction}
错误输出（直接输出诗句，不要解释）："""

def process_single_sample(sample):
    """处理单个样本，生成 rejected"""
    instruction = sample.get("instruction", "")
    chosen = sample.get("output", "")
    
    if not instruction or not chosen:
        return None
    
    # 构造 prompt
    prompt = REJECTED_PROMPT_TEMPLATE.format(instruction=instruction)
    
    try:
        rejected = run_llm_inference(prompt, 0, 2560,"qwen-max")
        # 清理输出
        rejected = rejected.strip().split("\n")[0]  # 只取第一行（防多余内容）
        return {
            "instruction": instruction,
            "chosen": chosen,
            "rejected": rejected
        }
    except Exception as e:
        print(f"生成 rejected 失败: {instruction} | 错误: {e}")
        return None

def main():
    input_file = "/home/dsl/learn/poem/processed_data/data_new/new_train_cleaned.jsonl"  # 你的原始训练数据
    output_file = "/home/dsl/learn/poem/processed_data/data_new/dpo_preference_data.jsonl"
    max_samples = 5000  # 可调整

    # Step 1: 读取原始数据
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"✅ 读取 {len(samples)} 条原始样本")

    # Step 2: 打乱并抽样
    random.shuffle(samples)
    samples_to_process = samples[:max_samples]
    print(f"✅ 抽取 {len(samples_to_process)} 条用于生成 rejected")

    # Step 3: 并发生成 rejected
    results = []
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(process_single_sample, sample) for sample in samples_to_process]
        for f in tqdm(as_completed(futures), total=len(futures), desc="生成 rejected"):
            item = f.result()
            if item:
                results.append(item)
    
    # Step 4: 保存 DPO 偏好数据
    with open(output_file, "w", encoding="utf-8") as out_f:
        for item in results:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"🎉 生成 {len(results)} 条 DPO 偏好数据，保存至 {output_file}")

if __name__ == "__main__":
    main()