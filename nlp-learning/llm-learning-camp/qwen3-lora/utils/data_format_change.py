import json

input_path = "/hy-tmp/workspace/qwen-lora/data/train_medical.json"
output_path = "/hy-tmp/workspace/qwen-lora/LLaMA-Factory/data/alpaca_zh_medical.json"

alpaca_data = []

with open(input_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue  # 跳过空行
        
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"第 {line_idx + 1} 行 JSON 解析失败，已跳过")
            continue

        # 映射并对齐到 Alpaca 核心三字段
        # 兼容常见原始字段名：instruction/ask/question/prompt, input, output/answer/response
        instruction = item.get("instruction") or item.get("question") or item.get("ask") or item.get("prompt") or ""
        input_text = item.get("input") or ""
        output_text = item.get("output") or item.get("answer") or item.get("response") or ""

        # 过滤掉缺失问答的脏数据
        if not instruction and not input_text:
            continue
        if not output_text:
            continue

        alpaca_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

print(f"成功转换有效样本数量: {len(alpaca_data)}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)