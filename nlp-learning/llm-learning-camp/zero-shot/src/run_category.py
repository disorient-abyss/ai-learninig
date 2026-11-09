import os
import time

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download


def test():
    model_id = "Qwen/Qwen3.5-0.8B"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.abspath(os.path.join(current_dir, "../data/qwen"))

    # 自动复用本地缓存
    print(f"正在准备模型权重: {model_id} ...")
    model_dir = snapshot_download(model_id=model_id, cache_dir=cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # 加载模型并切换至评估模式
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=compute_dtype if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # ================= 预热模型：让算子回退警告在此处先触发完毕 =================
    print("正在预热模型并初始化算子...")
    dummy_input = tokenizer("warmup", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**dummy_input, max_new_tokens=2)
    print("=" * 50 + "\n模型就绪，开始执行正式分类推理：" )

    sentences = [
        "《琅琊榜》是由山东影视传媒集团,山东影视制作有限公司, 北京儒意欣欣影业投资有限公司,北京和颂天地影视文化有限公司, 北京圣基影业有限公司, 东阳正午阳光影视有限公司联合出品, 由孔笙, 李雪执导, 胡歌, 刘涛, 王凯, 黄维德, 陈龙, 吴磊, 高鑫等主演的古装剧。",
        "《满江红》是由张艺谋执导, 沈腾, 易烊千玺, 张译, 雷佳音, 岳云鹏, 王佳怡领衔主演, 潘斌龙, 余皑磊主演, 郭京飞, 欧豪友情出演, 魏翔, 张弛, 黄炎特别出演, 许静雅, 蒋鹏宇, 林博洋, 飞凡, 任思诺, 陈永胜出演的悬疑喜剧电影。",
        "布宜诺斯艾利斯（Buenos Aires,华人常简称为布宜诺斯）是阿根廷共和国（the Republic of Argentina, Republica Argentina）的首都和最大城市，位于拉普拉塔河南岸，南美洲东南部，河对岸为乌拉圭东岸共和国。",
        "加拿大（英语/法语: Canada），首都渥太华，位于北美洲北部。东临大西洋，西濒太平洋，西北部邻美国阿拉斯加州，南接美国本土，北靠北冰洋。气候大部分为亚寒带针叶林气候和湿润大陆性气候，北部极地区域为极地长寒气候。",
        "张译（原名张毅），1978年2月17日出生于黑龙江省哈尔滨市，中国内地演员。1997年至2006年服役于北京军区政治部战友话剧团。2006年，主演军事励志题材电视剧《士兵突击》。",
    ]

    labels = ["人物", "书籍", "电视剧", "电影", "城市", "国家"]

    with torch.no_grad():
        for sent in sentences:
            print('-' * 50)
            print(f"sentence: {sent}", flush=True)

            start_time = time.time()

            # Few-shot 对话示例：引导 0.8B 迅速掌握输出风格
            messages = [
                {
                    "role": "system",
                    "content": "你是一个文本分类器。直接输出类别名称，不要输出任何标点或分析解释。",
                },
                {"role": "user", "content": "《红楼梦》是清代作家曹雪芹创作的章回体长篇小说。"},
                {"role": "assistant", "content": "书籍"},
                {"role": "user", "content": "北京市是中华人民共和国的首都。"},
                {"role": "assistant", "content": "城市"},
                {"role": "user", "content": f"请将这段文本分类到[{', '.join(labels)}]：\n{sent}"},
            ]

            # 兼容处理：若当前 tokenizer 支持 enable_thinking 参数则关闭思考链
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

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # 生成配置：max_new_tokens 控制在 8 以内足够输出 2~4 个汉字类别
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            raw_response = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            end_time = time.time()

            # 清洗思考标签：若包含 </think>，取其后面的实际回复内容
            clean_text = raw_response
            if "</think>" in clean_text:
                clean_text = clean_text.split("</think>")[-1].strip()

            # 优先从清洗文本中直接匹配预设的 6 个标签
            final_label = None
            for label in labels:
                if label in clean_text:
                    final_label = label
                    break

            if not final_label:
                # 兜底截取首行并去除符号
                first_line = clean_text.split("\n")[0].strip()
                final_label = first_line.strip("。，：. ") or "[无输出]"

            # 确保每轮推理完成后立即完整输出
            print(f"\nresponse: {final_label}\n", flush=True)
            print(f"cost time: {end_time - start_time}", flush=True)



if __name__ == "__main__":
    test()