#!/bin/bash
# vllm-omni 起 Qwen3-Omni（thinker+talker），OpenAI 兼容接口一次出 文字+语音
#   环境：vllm-omni 0.18.0 + vLLM 0.18.0（H20 / 驱动 535 可跑），transformers 4.57.6
#   MODEL=<Qwen3-Omni-30B-A3B-Instruct 目录> CVD=0,1 PORT=8324 bash tools/serve_vllm_omni.sh
set -eu
M=${MODEL:?"MODEL=Qwen3-Omni-30B-A3B-Instruct 的本地目录（tokenizer 需要补 extra_special_tokens，见 docs/04）"}
export CUDA_VISIBLE_DEVICES=${CVD:-3,4}   # 默认阶段配置：thinker 在可见卡 0，talker+code2wav 在可见卡 1 → 要两张卡
exec vllm serve "$M" --omni --served-model-name qwen3omni-talker --host 127.0.0.1 --port ${PORT:-8324} \
  --init-timeout 1800 --stage-init-timeout 1800 --max-model-len 32768 --limit-mm-per-prompt '{"image": 8}' --gpu-memory-utilization 0.9
