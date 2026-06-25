# Qwen3-TTS 1.7B (vLLM-Omni) — vast.ai serverless model-server image.
# Bakes the model in so scaled-up workers don't re-download 3.5GB each.
# Based on the vast pytorch image we validated (CUDA 13, /venv/main, supervisor).
#
# Build (needs ~40GB disk during build; final image ~18-22GB):
#   docker build -t <registry>/roleplaychats-tts:1.7b deploy/tts-serverless
#   docker push <registry>/roleplaychats-tts:1.7b
FROM vastai/pytorch:cuda-13.0.3-auto

ENV HF_HOME=/opt/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    MODEL_SERVER_PORT=8091 \
    MODEL_LOG=/var/log/portal/vllm.log

# 1) Dedicated venv with matched vllm + vllm-omni (minor versions MUST match).
#    uv lives at /usr/local/bin/uv in this base image (NOT /venv/main/bin).
RUN /usr/local/bin/uv venv /venv/tts --python 3.12 && \
    /usr/local/bin/uv pip install --python /venv/tts/bin/python \
      vllm==0.22.1 vllm-omni==0.22.0 hf_transfer huggingface_hub

# 2) Bake the model weights (talker + Code2Wav) into the image.
RUN /venv/tts/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_NAME}')"

# 3) Tuned single-GPU deploy config (talker max_model_len 4096 -> 2048: 13.55x concurrency).
#    Locate the stock config by path (no python import — the CI builder has no GPU).
RUN CFG=$(find /venv/tts -path '*/vllm_omni/deploy/qwen3_tts.yaml' | head -1) && \
    test -n "$CFG" && mkdir -p /opt/tts && cp "$CFG" /opt/tts/qwen3_17b.yaml && \
    sed -i 's/max_model_len: 4096/max_model_len: 2048/' /opt/tts/qwen3_17b.yaml

# 4) Model-server launcher + a tiny reference clip for the autoscaler benchmark
#    (fetched + clipped to 10s at build time; no binary checked into the repo).
COPY start-model.sh /opt/tts/start-model.sh
RUN chmod +x /opt/tts/start-model.sh && \
    curl -4 -L -s -o /tmp/ref.flac 'https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac' && \
    ffmpeg -y -i /tmp/ref.flac -t 10 -ar 24000 -ac 1 /opt/tts/bench_ref.wav && rm -f /tmp/ref.flac

# 5) Supervisor program: start vLLM-Omni on boot, logging to MODEL_LOG so the
#    PyWorker's LogActionConfig can detect "Application startup complete."
COPY tts-supervisor.conf /etc/supervisor/conf.d/tts.conf

EXPOSE 8091
