import json
data = []
with open("/home/dsl/learn/poem/processed_data/data_dpo/dpo_preference_data.jsonl","r",encoding="utf-8") as f:
    for line in  f:
        line = line.strip()
        if line:
            js = json.loads(line)
            if js["rejected"] != "":
                data.append(js)
for da in data:
    with open("/home/dsl/learn/poem/processed_data/data_dpo/dpo_preference_data_clean.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(da,ensure_ascii=False)+"\n")
        

