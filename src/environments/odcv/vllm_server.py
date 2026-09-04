# ABOUTME: Starts, health-checks and stops a vLLM OpenAI server for the ODCV policy (base model + optional LoRA) inside a SLURM job
# ABOUTME: Use: with VllmServer(base, lora_path=..., lora_name=..., tensor_parallel=2) as s: ... s.base_url
"""
The serving half of a rollout round, as a context manager so a crash in the
collector still kills the server and frees the GPUs for the training half.
Arguments mirror scripts/slurm/collect_odcv_rollouts_job.sh: Qwen3.6 needs the
qwen3 reasoning parser, the qwen3_xml tool parser and a long context.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


class VllmServer:
    def __init__(
        self,
        base_model: str,
        *,
        lora_path: str | None = None,
        lora_name: str = "policy",
        tensor_parallel: int = 2,
        port: int = 8000,
        max_model_len: int = 65536,
        max_num_seqs: int = 16,
        gpu_fraction: float = 0.90,
        gpus: str | None = None,
        log_path: str | Path = "vllm.log",
        max_lora_rank: int = 64,
        startup_timeout_s: int = 1500,
    ):
        self.base_model, self.lora_path, self.lora_name = (
            base_model,
            lora_path,
            lora_name,
        )
        self.tp, self.port, self.max_model_len, self.max_num_seqs = (
            tensor_parallel,
            port,
            max_model_len,
            max_num_seqs,
        )
        self.gpu_fraction, self.gpus, self.log_path = gpu_fraction, gpus, Path(log_path)
        self.max_lora_rank, self.startup_timeout_s = max_lora_rank, startup_timeout_s
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def served_name(self) -> str:
        return self.lora_name if self.lora_path else "base"

    def start(self) -> "VllmServer":
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.base_model,
            "--served-model-name",
            "base",
            "--tensor-parallel-size",
            str(self.tp),
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.gpu_fraction),
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_xml",
        ]
        if self.lora_path:
            cmd += [
                "--enable-lora",
                "--lora-modules",
                f"{self.lora_name}={self.lora_path}",
                "--max-lora-rank",
                str(self.max_lora_rank),
            ]
        env = dict(os.environ)
        if self.gpus:
            env["CUDA_VISIBLE_DEVICES"] = self.gpus
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            cmd, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT, env=env
        )
        t0 = time.time()
        while time.time() - t0 < self.startup_timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with {self.proc.returncode}; see {self.log_path}"
                )
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=2
                )
                print(
                    f"vLLM healthy after {time.time() - t0:.0f}s (served: {self.served_name})",
                    flush=True,
                )
                return self
            except Exception:
                time.sleep(5)
        self.stop()
        raise RuntimeError(
            f"vLLM not healthy after {self.startup_timeout_s}s; see {self.log_path}"
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
