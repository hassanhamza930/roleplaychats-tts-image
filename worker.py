"""
Vast.ai Serverless PyWorker for Qwen3-TTS (vLLM-Omni) voice cloning.

vLLM-Omni exposes an OpenAI-compatible speech API; this PyWorker proxies
POST /v1/audio/speech to the local vLLM-Omni server, computes per-request
workload for the autoscaler, and reports readiness from the model log.

Deploy: this file + requirements.txt live in a public git repo; the vast
serverless template clones it (PYWORKER_REPO) and runs `python worker.py`.
See deploy/tts-serverless/README.md.
"""
import os
import base64

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# Local vLLM-Omni server (started by the image; see start-model.sh / Dockerfile).
MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "8091"))
MODEL_LOG_FILE = os.environ.get("MODEL_LOG", "/var/log/portal/vllm.log")
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# vLLM-Omni readiness / failure markers (verified in our runs).
MODEL_LOAD_LOG_MSG = ["Application startup complete."]
MODEL_ERROR_LOG_MSGS = [
    "INFO exited: vllm",
    "RuntimeError: Engine",
    "Traceback (most recent call last):",
]
MODEL_INFO_LOG_MSGS = ['"message":"Download']

# A short reference clip baked into the image, used only for the autoscaler's
# capacity benchmark (a Base-task voice clone, matching real traffic shape).
BENCH_REF_PATH = os.environ.get("BENCH_REF_PATH", "/opt/tts/bench_ref.wav")
BENCH_REF_TEXT = "the quick brown fox jumps over the lazy dog"


def _bench_ref_data_url() -> str:
    with open(BENCH_REF_PATH, "rb") as f:
        return "data:audio/wav;base64," + base64.b64encode(f.read()).decode()


def speech_benchmark_generator() -> dict:
    """A representative voice-clone request for throughput estimation."""
    return {
        "input": "Hello there, this is a benchmark utterance for capacity estimation.",
        "task_type": "Base",
        "ref_audio": _bench_ref_data_url(),
        "ref_text": BENCH_REF_TEXT,
        "non_streaming_mode": True,
    }


# Workload ~ output audio length ~ input text length (TTS is decode-bound on the
# codec stream). Floor at 1 so empty inputs still count as a unit of work.
def speech_workload(data: dict) -> float:
    return float(max(len(str(data.get("input", "")) or ""), 1))


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/v1/audio/speech",
            allow_parallel_requests=True,  # vLLM-Omni continuous batching
            workload_calculator=speech_workload,
            max_queue_time=120.0,
            benchmark_config=BenchmarkConfig(
                generator=speech_benchmark_generator,
                concurrency=8,
                runs=3,
            ),
        ),
        # Streaming variant (low time-to-first-audio); same backend.
        HandlerConfig(
            route="/v1/audio/speech/stream",
            allow_parallel_requests=True,
            workload_calculator=speech_workload,
            max_queue_time=120.0,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS,
    ),
)

Worker(worker_config).run()
