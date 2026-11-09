import json
import os
import re
import time

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 统一 Schema 定义
SCHEMA = {
    '人物': ['姓名', '性别', '出生日期', '出生地', '职业', '获得奖项'],
    '书籍': ['书名', '作者', '类型', '发行时间', '定价'],
    '电视剧': ['电视剧名称', '导演', '演员', '题材', '出品方']
}

# 2. Few-shot 示例：用来强行约束 0.8B 模型的输出格式与字段结构
ie_examples = {
    '人物': [
        {
            'content': '岳云鹏，本名岳龙刚，1985年4月15日出生于河南省濮阳市南乐县，中国内地相声、影视男演员。',
            'answers': {
                '姓名': ['岳云鹏'],
                '性别': ['男'],
                '出生日期': ['1985年4月15日'],
                '出生地': ['河南省濮阳市南乐县'],
                '职业': ['相声演员', '影视演员'],
                '获得奖项': ['原文中未提及']
            }
        }
    ],
    '书籍': [
        {
            'content': '《三体》是刘慈欣创作的长篇科幻小说系列，由《三体》《三体2：黑暗森林》《三体3：死神永生》组成，第一部于2006年5月起在《科幻世界》杂志上连载，第二部于2008年5月首次出版，第三部则于2010年11月出版。',
            'answers': {
                '书名': ['《三体》'],
                '作者': ['刘慈欣'],
                '类型': ['长篇科幻小说'],
                '发行时间': ['2006年5月', '2008年5月', '2010年11月'],
                '定价': ['原文中未提及']
            }
        }
    ]
}

def init_prompts():
    """将 Schema 规则与 ie_examples 共同注入历史消息"""
    schema_str = json.dumps(SCHEMA, ensure_ascii=False)
    system_instruction = (
        f"你是一个信息抽取任务的专家。系统预定义了以下 Schema 属性库：\n{schema_str}\n\n"
        "【严格约束】：\n"
        "1. 必须且只能从当前文本所属类别的 Schema 列表中抽取属性；\n"
        "2. 严禁自创、拓展任何 Schema 以外的多余字段（如'服役经历'、'作品'等）；\n"
        "3. 所有属性值必须统一为列表 list[str]；\n"
        "4. Schema 包含但原文未提及的属性，必须严格填充为 ['原文中未提及']；\n"
        "5. 严禁输出任何思考过程或解释，起手直接输出 ```json ... ``` 单行代码块。"
    )
    
    pre_history = [{'role': 'system', 'content': system_instruction}]

    # 规范的 user-assistant 成对 Few-Shot
    for example_list in ie_examples.values():
        for example in example_list:
            sentence = example['content']
            answers = json.dumps(example['answers'], ensure_ascii=False)
            pre_history.append({'role': 'user', 'content': sentence})
            pre_history.append({'role': 'assistant', 'content': f"```json\n{answers}\n```"})
            
    return pre_history

def enforce_schema_alignment(parsed_dict: dict, schema: dict) -> dict:
    """严格过滤非 Schema 字段，并补齐漏掉的字段"""
    if not isinstance(parsed_dict, dict):
        return parsed_dict

    matched_category = None
    if any(k in parsed_dict for k in ['电视剧名称', '题材', '出品方']):
        matched_category = '电视剧'
    elif any(k in parsed_dict for k in ['书名', '定价']):
        matched_category = '书籍'
    elif any(k in parsed_dict for k in ['姓名', '出生地点', '出生地', '职业']):
        matched_category = '人物'

    if not matched_category:
        return parsed_dict

    target_keys = schema[matched_category]
    strict_result = {}

    if matched_category == '人物' and '出生地' in parsed_dict and '出生地点' not in parsed_dict:
        parsed_dict['出生地点'] = parsed_dict['出生地']

    for key in target_keys:
        val = parsed_dict.get(key)
        if val is None or val == [] or val == "":
            strict_result[key] = ["原文中未提及"]
        elif isinstance(val, list):
            strict_result[key] = val
        else:
            strict_result[key] = [str(val)]

    return strict_result
def clean_response(response: str):
    """清洗模型输出并解析为 dict"""
    json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, flags=re.DOTALL)
    target_str = json_blocks[0] if json_blocks else response.strip()
    target_str = target_str.replace('、', ',')

    parsed = None
    try:
        parsed = json.loads(target_str)
    except json.JSONDecodeError:
        brace_match = re.search(r'\{.*\}', target_str, flags=re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

    if isinstance(parsed, dict):
        # 经过强 Schema 校验器清洗
        aligned_dict = enforce_schema_alignment(parsed, SCHEMA)
        return aligned_dict
    return response

def predict(sentences, pre_history, model, tokenizer):
    for idx, sent in enumerate(sentences, 1):
        print(f"\n[{idx}] 待抽取句子: {sent}")
        start_time = time.time()

        messages = list(pre_history)
        messages.append({'role': 'user', 'content': sent})

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # 核心引导：Pre-fill 强制跳过思考，直接从 json 括号开始生成
        prompt_with_prefix = text + "```json\n{"
        # 修复：确保将 prompt_with_prefix 送入 tokenizer
        model_inputs = tokenizer([prompt_with_prefix], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        # 修复：仅做一次截取
        pure_ids = [
            out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
        ]
        generated_suffix = tokenizer.batch_decode(pure_ids, skip_special_tokens=True)[0]
        end_time = time.time()

        # 拼接回完整 JSON 代码块，确保下游正则正常提取
        full_response = "```json\n{" + generated_suffix
        parsed_result = clean_response(full_response)
        
        print('抽取结果:')
        if isinstance(parsed_result, dict):
            print(json.dumps(parsed_result, ensure_ascii=False, separators=(',', ':')))
        else:
            print(parsed_result)
            
        print(f"cost time: {end_time - start_time:.2f}s", flush=True)
        print('-' * 50)

if __name__ == '__main__':
    # 1. 确定计算设备与精度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}")

    compute_dtype = (
        torch.bfloat16 
        if (device == "cuda" and torch.cuda.is_bf16_supported()) 
        else torch.float16
    )

    # 2. 准备模型本地缓存路径
    model_id = "Qwen/Qwen3.5-0.8B"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.abspath(os.path.join(current_dir, "../data/qwen"))

    print(f"正在准备模型权重: {model_id} ...")
    # snapshot_download 会先检查 cache_dir，存在则直接秒级返回本地路径，不存在则自动下载
    model_dir = snapshot_download(model_id=model_id, cache_dir=cache_dir)
    print(f"模型权重就绪，本地目录: {model_dir}")

    # 3. 分词器与模型统一从本地 model_dir 加载（彻底断开远程网络依赖）
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, 
        trust_remote_code=True,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=compute_dtype if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()

    # 4. 待预测样本
    sentences = [
        '张译(原名张毅)，1978年2月17日出生于黑龙江省哈尔滨市，中国内地男演员。1997年至2006年服役于北京军区政治部战友话剧团。2006年，主演军事励志题材电视剧《士兵突击》。',
        '《琅琊榜》是由山东影视传媒集团、山东影视制作有限公司、北京儒意欣欣影业投资有限公司、北京和颂天地影视文化有限公司、北京圣基影业有限公司、东阳正午阳光影视有限公司联合出品，由孔笙、李雪执导，胡歌、刘涛、王凯、黄维德、陈龙、吴磊、高鑫等主演的古装剧。'
    ]

    # ================= 预热模型：让算子回退警告在此处先触发完毕 =================
    print("正在预热模型并初始化算子...")
    dummy_input = tokenizer("warmup", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**dummy_input, max_new_tokens=2)
    print("=" * 50 + "\n模型就绪，开始执行正式执行抽样：" )

    # 5. 执行推理
    pre_history = init_prompts()
    predict(sentences, pre_history, model, tokenizer)