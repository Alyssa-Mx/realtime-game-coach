#!/usr/bin/env python3
"""把 LoRA 合进基座、存成一个普通模型 —— 给 vLLM 用（也省得每次加载都合一遍）。

W_新 = W + BA，数学等价，不掉精度（bf16 舍入点会变，实测 1860 条判定全部打平）。
存出来之后它就是个普通 Qwen3-VL，vLLM 不需要 LoRA 支持。
用法: python train/merge_lora.py [adapter=data/sft/v7/final] [out=data/sft/v7/merged]
"""
import os, sys, shutil, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

BASE = os.environ.get('BASE_MODEL', 'Qwen/Qwen3-VL-8B-Instruct')
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else 'data/sft/v7/final'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data/sft/v7/merged'
ADAPTER = ADAPTER if ADAPTER.startswith('/') else T.R + '/' + ADAPTER
OUT = OUT if OUT.startswith('/') else T.R + '/' + OUT

print('base   ', BASE, flush=True)
print('adapter', ADAPTER, flush=True)
print('out    ', OUT, flush=True)
m = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16)
print('base loaded', flush=True)
m = PeftModel.from_pretrained(m, ADAPTER).merge_and_unload()
print('merged', flush=True)
os.makedirs(OUT, exist_ok=True)
m.save_pretrained(OUT, safe_serialization=True)
AutoProcessor.from_pretrained(BASE).save_pretrained(OUT)   # vLLM 要在同目录找到 tokenizer / preprocessor 配置
print('saved', flush=True)
print('MERGE_DONE', flush=True)
