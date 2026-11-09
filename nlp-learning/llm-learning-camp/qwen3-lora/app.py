import datetime
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 配置与全局对象
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/disorient/Projects/learning-AI-Projects/nlp-learning/llm-learning-camp/qwen3-lora/LLaMA-Factory/saves/qwen3.5-4b_sft_merged",
)
generate_lock = threading.Lock()
model_resources: dict[str, Any] = {}


# 2. Pydantic 校验与数据契约
class ChatRequest(BaseModel):
    prompt: str
    history: list[dict[str, str]] = Field(default_factory=list)
    max_length: int = Field(default=2048, ge=1, le=8192)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int = Field(default=20, ge=0)
    enable_thinking: bool = False


# 3. 使用 lifespan 管理模型初始化与释放
@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    # 获取 Qwen 关键终止符 token ids，避免生成失控溢出
    stop_token_ids = []
    for token in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and isinstance(tid, int):
            stop_token_ids.append(tid)

    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            stop_token_ids.extend(tokenizer.eos_token_id)
        else:
            stop_token_ids.append(tokenizer.eos_token_id)

    stop_token_ids = list(set(stop_token_ids))

    model_resources["tokenizer"] = tokenizer
    model_resources["model"] = model
    model_resources["stop_token_ids"] = stop_token_ids

    yield

    model_resources.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Qwen3 Inference Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def split_thinking(output_ids: list[int], tokenizer: AutoTokenizer, enable_thinking: bool):
    """鲁棒地拆分思考过程与正式回答"""
    full_output = tokenizer.decode(output_ids, skip_special_tokens=False).strip()

    # 移除可能残留的终止符
    for stop_str in ["<|im_end|>", "<|endoftext|>"]:
        if full_output.endswith(stop_str):
            full_output = full_output[:-len(stop_str)].strip()

    if enable_thinking and "</think>" in full_output:
        parts = full_output.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        response = parts[1].strip()
    else:
        thinking = ""
        response = full_output.replace("<think>", "").replace("</think>", "").strip()

    return thinking, response


@app.post("/")
def create_item(req: ChatRequest):
    tokenizer: AutoTokenizer = model_resources.get("tokenizer")
    model: AutoModelForCausalLM = model_resources.get("model")
    stop_token_ids: list[int] = model_resources.get("stop_token_ids", [])

    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    # 过滤掉 history 中可能损坏或空的轮次
    valid_history = [
        {"role": m["role"], "content": m["content"]}
        for m in req.history
        if m.get("role") in ["user", "assistant", "system"] and m.get("content")
    ]
    messages = valid_history + [{"role": "user", "content": req.prompt}]

    # 构建输入 Prompt：使用 conversation 关键字传参，并做异常回退兼容
    try:
        text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=req.enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = model_inputs.input_ids.shape[-1]

    # 设置超参策略
    top_p = req.top_p if req.top_p is not None else (0.95 if req.enable_thinking else 0.8)
    temperature = req.temperature if req.temperature is not None else (0.6 if req.enable_thinking else 0.7)

    gen_kwargs = {
        "max_new_tokens": req.max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "top_k": req.top_k,
    }
    if stop_token_ids:
        gen_kwargs["eos_token_id"] = stop_token_ids
    if tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with generate_lock, torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **gen_kwargs,
        )
        output_ids = generated_ids[0][input_length:].tolist()

    # 拆解思考链与实际回答
    thinking, response = split_thinking(output_ids, tokenizer, req.enable_thinking)

    # 拼接多轮对话历史
    updated_history = messages + [{"role": "assistant", "content": response}]
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{current_time}] prompt: {req.prompt[:30]}... | resp_len: {len(response)} | think_len: {len(thinking)}")

    return {
        "response": response,
        "thinking": thinking,
        "history": updated_history,
        "status": 200,
        "time": current_time,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8103, workers=1)