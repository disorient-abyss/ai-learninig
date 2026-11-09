import os

from modelscope import snapshot_download

# 指定模型 ID 与存放路径（确保在 /hy-tmp 数据盘下，避免撑爆系统盘）
model_id = 'Qwen/Qwen3.5-4B'  # 请替换为你要下载的实际模型名称
save_dir = os.path.join('/hy-tmp/qwen-lora/workspace/models', model_id.split('/')[-1])

model_dir = snapshot_download(
    model_id=model_id,
    local_dir=save_dir
)

print("模型已保存至:", model_dir)