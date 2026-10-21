import os

import fasttext
import jieba
from flask import Flask, jsonify, request

app = Flask(__name__)

# ----------------- 1. 配置与资源初始化 -----------------
MODEL_PATH = './model_bin/toutiao_fasttext_jieba_1792382370.bin'
STOPWORDS_PATH = './data/stopwords.txt'

# 加载⾃定义的停⽤词字典
stopwords = set()
if os.path.exists(STOPWORDS_PATH):
    with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        stopwords = {line.strip() for line in f if line.strip()}
    print(f"停用词加载完成，共 {len(stopwords)} 个词。")
else:
    print(f"警告: 停用词文件 {STOPWORDS_PATH} 未找到，跳过过滤。")

# 实例化 fasttext 对象, 并加载模型参数⽤于推断, 提供服务请求
print("正在加载 FastText 模型...")
model = fasttext.load_model(MODEL_PATH)
print("FastText 模型实例化完毕。")

# ----------------- 2. 文本预处理函数 -----------------
def preprocess_text(text: str) -> str:
    """分词并剔除停用词与多余空白, 因为前⾯加载的是基于词的模型, 所以这⾥⽤ jieba 进⾏分词"""
    words = jieba.lcut(text)
    filtered = [w.strip() for w in words if w.strip() and w not in stopwords]
    return " ".join(filtered)


# ----------------- 3. 路由与推断逻辑 -----------------
@app.route('/v1/main_server/', methods=["POST"])
def main_server():
    # 兼容 JSON 与 Form 提交
    data = request.get_json(silent=True) or request.form

    uid = data.get('uid')
    text = data.get('text')

    if not text:
        return jsonify({
            "code": 400,
            "msg": "缺少必要参数: text"
        }), 400

    try:
        # 对请求⽂本进⾏处理
        processed_text = preprocess_text(text)
        # k=1 表示取置信度最高的一个类别
        labels, probs = model.predict(processed_text, k=1)

        raw_label = labels[0]
        confidence = float(probs[0])
        # 移除 FastText 默认的 __label__ 前缀
        clean_label = raw_label.replace('__label__', '')

        return jsonify({
            "code": 200,
            "msg": "success",
            "data": {
                "uid": uid,
                "label": clean_label,
                "probability": round(confidence, 4)
            }
        })
    except (ValueError, RuntimeError) as e:
        return jsonify({
            "code": 500,
            "msg": f"推理异常: {e}"
        }), 500


# ----------------- 4. 服务入口 -----------------
if __name__ == '__main__':
    # 开发调试环境下启动
    # host='0.0.0.0' 允许外部网络访问
    app.run(host='0.0.0.0', port=5000, debug=False)