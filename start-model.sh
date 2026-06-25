#!/bin/bash
# Launch vLLM-Omni serving Qwen3-TTS 1.7B, logging to MODEL_LOG so the vast
# PyWorker can detect readiness ("Application startup complete.").
# Started by supervisor (tts-supervisor.conf) on worker boot.
set -e
export HF_HOME=/opt/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$(dirname "${MODEL_LOG:-/var/log/portal/vllm.log}")"
source /venv/tts/bin/activate

# NOTE: never --enforce-eager (kills CUDA graphs). Default sampling params live
# in the deploy config; do not change them.
exec vllm serve "${MODEL_NAME:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}" \
  --deploy-config /opt/tts/qwen3_17b.yaml \
  --omni --port "${MODEL_SERVER_PORT:-8091}" --trust-remote-code \
  >> "${MODEL_LOG:-/var/log/portal/vllm.log}" 2>&1
