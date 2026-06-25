#!/bin/bash
# vast.ai serverless worker bootstrap (fetched at runtime by the template onstart —
# cache-proof: relies only on the baked model + venv + config, which exist in every
# image version, so a stale cached image still works).
#
# 1) start vLLM-Omni (model/config baked at /opt/hf + /opt/tts) -> log file the
#    PyWorker watches for "Application startup complete."
# 2) hand off to vast's STANDARD serverless PyWorker bootstrap (SSL + reporting + worker.py).
set -e
export HF_HOME=/opt/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /var/log/portal

nohup /venv/tts/bin/vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --deploy-config /opt/tts/qwen3_17b.yaml \
  --omni --port "${MODEL_SERVER_PORT:-8091}" --trust-remote-code \
  >> /var/log/portal/vllm.log 2>&1 &

wget -qO /root/start_server.sh https://raw.githubusercontent.com/vast-ai/pyworker/main/start_server.sh
exec bash /root/start_server.sh
