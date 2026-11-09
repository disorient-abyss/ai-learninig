import time

import requests

URL = "http://127.0.0.1:8103/"


def chat(session: requests.Session, prompt: str, history: list, enable_thinking: bool = False):
    """发送一轮对话请求并返回解析结果与耗时"""
    payload = {
        "prompt": prompt,
        "history": history,
        "max_length": 1024,
        "top_p": 0.95 if enable_thinking else 0.8,
        "temperature": 0.6 if enable_thinking else 0.7,
        "enable_thinking": enable_thinking,
    }

    start_time = time.perf_counter()
    resp = session.post(URL, json=payload, timeout=60)
    resp.raise_for_status()  # 若 HTTP 状态码不是 2xx 则主动抛出异常
    cost_time = time.perf_counter() - start_time

    data = resp.json()
    return data, cost_time


def main():
    # 模拟多轮对话提问列表
    prompts = [
        "宝宝白天爱磨牙会是哪些情况呢？",
        "那需要去医院检查吗？"
    ]
    
    # 是否开启深度思考链（测试 Qwen3 思考模式设为 True）
    enable_thinking = False
    history = []

    # 使用 Session 保持长连接，降低网络延迟
    with requests.Session() as session:
        for round_idx, prompt in enumerate(prompts, 1):
            print(f"\n{'=' * 20} 第 {round_idx} 轮对话 {'=' * 20}")
            print(f"用户问题: {prompt}")

            result, cost = chat(session, prompt, history, enable_thinking=enable_thinking)
            
            # 若包含思考过程且内容不为空，打印思考链
            if enable_thinking and result.get("thinking"):
                print(f"\n[Thinking]:\n{result['thinking']}\n")

            print(f"模型回答: {result['response']}")
            print(f"单轮耗时: {cost:.2f} s")

            # 滚动更新上下文历史
            history = result.get("history", [])


if __name__ == "__main__":
    main()