import time

import requests


def main():
    # 客户端连接本机服务，建议使用 127.0.0.1
    url = "http://127.0.0.1:5000/v1/main_server/"
    data = {
        "uid": "AI-3-20260411", 
        "text": "公共英语(PETS)写作中常见的逻辑词汇总"
    }

    # 记录开始时间
    start_time = time.time()

    # 向服务发起 POST 请求
    res = requests.post(url, data=data, timeout=5)

    # 计算耗时（注意：start_time 是浮点数，不要加括号）
    cost_time = (time.time() - start_time) * 1000

    # 输出测试结果
    print("状态码:", res.status_code)
    print("输入文本:", data["text"])
    try:
        print("返回内容 (JSON):", res.json())
    except Exception:
        print("返回内容 (Text):", res.text)
    print(f"单条样本网络+推理总耗时: {cost_time:.2f} ms")

if __name__ == "__main__":
    main()