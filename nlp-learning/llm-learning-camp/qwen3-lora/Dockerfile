# 1. 基础镜像：使用 vast.ai 官方 PyTorch 镜像
FROM vastai/pytorch:latest

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 2. 设置 CUDA 架构兼容：8.6 (RTX 3090/A6000) 与 8.9 (RTX 4070S/4090)
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"

# 3. 安装必要的系统编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 升级 pip 并预装加速库依赖（避免 flash-attn 构建阶段缺少 wheel 依赖）
RUN pip install --upgrade --no-cache-dir pip setuptools wheel packaging

# 5. 安装 Flash-Attention 2（大模型微调显存优化必备）
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# 6. 克隆 LLaMA-Factory 源码
WORKDIR /root
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /root/LLaMA-Factory

# 7. 安装微调全套套件（包含 deepspeed、bitsandbytes、metrics）
WORKDIR /root/LLaMA-Factory
RUN pip install --no-cache-dir -e ".[metrics,deepspeed,bitsandbytes]"

# 8. （可选）如果你有自定义数据集注册需求，可在此将本地 dataset_info.json 预拷入，或者留空后续挂载
# 默认进入工作目录
WORKDIR /root/LLaMA-Factory
