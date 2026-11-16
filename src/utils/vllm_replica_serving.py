# ABOUTME: Serves a policy model as one vLLM OpenAI server per GPU group (replica), on ports that are free right now, for the ICRL pipelines
# ABOUTME: Use: with served(cfg, out_dir, adapter_dir_or_None, log=log) as servers: ... servers[i].base_url / .served_name  (cfg.policy.serve.replicas = ["0,1", "2,3"])
from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path


def free_ports(n: int, start: int) -> list[int]:
    """n ports, 10 apart from `start` upward, that nothing on this host is listening on."""
    import socket

    out, p = [], start
    while len(out) < n:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", p))  # all interfaces, as vLLM binds
                out.append(p)
            except OSError:
                pass
        p += 10
        if p > 60000:
            raise RuntimeError("no free ports found")
    return out


@contextlib.contextmanager
def served(cfg, out_dir: Path, adapter: Path | None, log=print):
    """Start one vLLM server per replica GPU group, concurrently; yield the list."""
    from src.environments.odcv.vllm_server import VllmServer

    s = cfg.policy.serve
    replicas = list(s.get("replicas") or [str(s.gpus)])
    # Nodes are shared (other jobs, other users): logs are per job and every
    # replica gets a port that is free right now.
    job = os.environ.get("SLURM_JOB_ID")
    tag = f"job{job}_" if job else ""
    ports = free_ports(
        len(replicas), start=8000 + (int(job) % 90) * 100 if job else 8000
    )
    servers = [
        VllmServer(
            str(cfg.policy.model),
            lora_path=str(adapter) if adapter else None,
            lora_name="policy",
            tensor_parallel=int(s.tensor_parallel),
            gpus=str(g),
            port=ports[i],
            max_model_len=int(s.max_model_len),
            max_num_seqs=int(s.max_num_seqs),
            gpu_fraction=float(s.gpu_fraction),
            log_path=out_dir / f"vllm_{tag}{i}.log",
            max_lora_rank=max(64, int((cfg.get("rl") or cfg.get("finetune") or {}).get("lora_r", 16))),
            reasoning_parser=s.reasoning_parser,
            tool_call_parser=s.tool_call_parser,
        )
        for i, g in enumerate(replicas)
    ]
    errors: list[BaseException] = []

    def start(srv):
        try:
            srv.start()
        except BaseException as e:  # noqa: BLE001 - re-raised below
            errors.append(e)

    t0 = time.time()
    threads = [threading.Thread(target=start, args=(srv,)) for srv in servers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        if errors:
            raise errors[0]
        log(
            f"{len(servers)} vLLM server(s) up in {time.time() - t0:.0f}s on GPU groups {replicas}"
        )
        yield servers
    finally:
        for srv in servers:
            srv.stop()
