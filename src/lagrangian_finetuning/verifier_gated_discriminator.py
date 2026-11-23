# ABOUTME: The VARL discriminator (Damani et al. 2026, arXiv 2607.01181): an LM backbone with a binary head trained to tell demonstrations from the policy's verifier-passing outputs, with replay buffers and accuracy-capped updates, served over HTTP so all GRPO ranks share one copy
# ABOUTME: Use: with discriminator_server(cfg, out_dir, log) as url: ...; from a reward function: DiscriminatorClient(url).step(texts, passed) -> {"scores": [...], "warmup": bool}
"""
Our reimplementation of VARL's discriminator from the paper's Appendix A and C
(no code is public). What follows the paper:

  * D is a classification head on a generative LM backbone, trained with binary
    cross-entropy: demonstrations are labelled 1, policy outputs 0.
  * Verifier filtering: only policy outputs that pass the verifier train D.
  * Two replay buffers, a FIFO of recent passing outputs and a reservoir sample
    of all passing outputs so far; negatives are drawn half from each.
  * Batches are balanced between demonstrations and policy outputs.
  * D is updated only when its accuracy on the latest batch is below a cap (0.8).
  * A warm-up of `warmup_steps` policy steps during which only D trains.
  * D sees phi(y), the benchmark's monitor text, never the chain of thought.

Where it differs (see the config for the values): the backbone is trained
through a LoRA adapter rather than fully, so the learning rate and batch size
are ours; and D is updated once per call (one call per GRPO rank per step)
instead of once per step on a 1,024-sample batch.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path


class ReplayBuffers:
    """A FIFO of the most recent items and a reservoir sample of every item seen; sample() draws half from each."""

    def __init__(self, fifo_size: int, reservoir_size: int, seed: int = 0):
        self.fifo: deque = deque(maxlen=int(fifo_size))
        self.reservoir: list = []
        self.reservoir_size, self.seen, self.rng = (
            int(reservoir_size),
            0,
            random.Random(seed),
        )

    def add(self, items: list) -> None:
        for x in items:
            self.fifo.append(x)
            self.seen += 1
            if len(self.reservoir) < self.reservoir_size:
                self.reservoir.append(x)
            else:
                j = self.rng.randrange(self.seen)
                if j < self.reservoir_size:
                    self.reservoir[j] = x

    def __len__(self) -> int:
        return len(self.fifo)

    def state(self) -> dict:
        return {
            "fifo": list(self.fifo),
            "reservoir": list(self.reservoir),
            "seen": self.seen,
            "rng": self.rng.getstate(),
        }

    def load_state(self, st: dict) -> None:
        self.fifo.clear()
        self.fifo.extend(st["fifo"])
        self.reservoir, self.seen = list(st["reservoir"]), int(st["seen"])
        self.rng.setstate(st["rng"])

    def sample(self, n: int) -> list:
        if not self.fifo:
            return []
        half = n // 2
        return [self.rng.choice(self.fifo) for _ in range(n - half)] + [
            self.rng.choice(self.reservoir) for _ in range(half)
        ]


def should_update(accuracy: float | None, threshold: float) -> bool:
    """VARL's accuracy cap: train D only while it is not already winning."""
    return accuracy is not None and accuracy < threshold


class VarlDiscriminator:
    """Holds D, the demonstrations and the buffers; step() scores a batch and (maybe) trains on it."""

    def __init__(
        self,
        score_fn,
        update_fn,
        demos: list[str],
        *,
        warmup_calls: int,
        accuracy_threshold: float,
        batch_size: int,
        fifo_size: int,
        reservoir_size: int,
        seed: int = 0,
        updates_per_call: int = 1,
    ):
        # updates_per_call > 1 when one generation feeds several policy steps, so D trains as often
        # per policy step as with one update per generation
        self.updates_per_call = max(1, int(updates_per_call))
        self.score_fn, self.update_fn, self.demos = score_fn, update_fn, demos
        self.warmup_calls, self.threshold, self.batch_size = (
            int(warmup_calls),
            float(accuracy_threshold),
            int(batch_size),
        )
        self.buffers = ReplayBuffers(fifo_size, reservoir_size, seed)
        self.rng, self.calls, self.updates = random.Random(seed + 1), 0, 0
        self.lock = threading.Lock()
        self.after_call = None  # called with the call count, inside the lock, once a call is fully applied

    def state(self) -> dict:
        """Everything but the model: counters, the sampling RNG and both buffers."""
        return {
            "calls": self.calls,
            "updates": self.updates,
            "rng": self.rng.getstate(),
            "buffers": self.buffers.state(),
        }

    def load_state(self, st: dict) -> None:
        self.calls, self.updates = int(st["calls"]), int(st["updates"])
        self.rng.setstate(st["rng"])
        self.buffers.load_state(st["buffers"])

    def step(self, texts: list[str], passed: list[bool]) -> dict:
        with self.lock:
            self.calls += 1
            scores = [float(s) for s in self.score_fn(texts)] if texts else []
            negatives = [t for t, p in zip(texts, passed) if p]
            accuracy = loss = None
            if negatives:
                # accuracy on the latest batch: passing policy outputs should score < 0.5,
                # an equal number of demonstrations > 0.5
                neg_scores = [s for s, p in zip(scores, passed) if p]
                pos = [self.rng.choice(self.demos) for _ in negatives]
                pos_scores = [float(s) for s in self.score_fn(pos)]
                right = sum(s < 0.5 for s in neg_scores) + sum(
                    s >= 0.5 for s in pos_scores
                )
                accuracy = right / (2 * len(negatives))
                self.buffers.add(negatives)
            updated = False
            if len(self.buffers) and (
                should_update(accuracy, self.threshold)
                or (accuracy is None and self.updates == 0)
            ):
                half = self.batch_size // 2
                for _ in range(self.updates_per_call):
                    loss = float(
                        self.update_fn(
                            [self.rng.choice(self.demos) for _ in range(half)],
                            self.buffers.sample(half),
                        )
                    )
                    self.updates += 1
                updated = True
            if self.after_call is not None:
                self.after_call(self.calls)
            return {
                "scores": scores,
                "warmup": self.calls <= self.warmup_calls,
                "accuracy": accuracy,
                "updated": updated,
                "loss": loss,
                "updates": self.updates,
                "buffer": len(self.buffers),
            }


class LmDiscriminatorModel:
    """LM backbone (LoRA) + linear head on the last token's hidden state; score() -> P(demonstration)."""

    def __init__(
        self,
        model_name: str,
        device: str,
        *,
        max_length: int,
        lora_r: int,
        lora_alpha: int,
        lr: float,
        score_batch: int = 16,
    ):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.max_length, self.score_batch = (
            torch,
            device,
            int(max_length),
            int(score_batch),
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side, self.tok.truncation_side = (
            "left",
            "left",
        )  # last token = end of the reply
        base = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
        base.gradient_checkpointing_enable()
        self.backbone = get_peft_model(
            base,
            LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        self.head = torch.nn.Linear(base.config.hidden_size, 1).to(device)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        params = [p for p in self.backbone.parameters() if p.requires_grad] + list(
            self.head.parameters()
        )
        self.opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)

    def _logits(self, texts: list[str]):
        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        hidden = self.backbone(**enc).last_hidden_state[:, -1, :]
        return self.head(hidden.float()).squeeze(-1)

    def score(self, texts: list[str]) -> list[float]:
        torch, out = self.torch, []
        self.backbone.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.score_batch):
                out += torch.sigmoid(
                    self._logits(texts[i : i + self.score_batch])
                ).tolist()
        return out

    def update(self, positives: list[str], negatives: list[str]) -> float:
        torch = self.torch
        self.backbone.train()
        texts = positives + negatives
        labels = torch.tensor(
            [1.0] * len(positives) + [0.0] * len(negatives), device=self.device
        )
        self.opt.zero_grad()
        total, n = 0.0, len(texts)
        for i in range(
            0, n, self.score_batch
        ):  # accumulate over chunks to bound memory
            logits = self._logits(texts[i : i + self.score_batch])
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels[i : i + self.score_batch], reduction="sum"
                )
                / n
            )
            loss.backward()
            total += loss.item()
        self.opt.step()
        return total

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(out_dir / "backbone_adapter"))
        self.torch.save(self.head.state_dict(), out_dir / "head.pt")

    def training_state(self) -> dict:
        """The trained weights (LoRA + head) and the optimizer, enough to continue training exactly."""
        return {
            "lora": {
                n: p.detach().cpu()
                for n, p in self.backbone.named_parameters()
                if p.requires_grad
            },
            "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
            "optimizer": self.opt.state_dict(),
        }

    def load_training_state(self, st: dict) -> None:
        named = dict(self.backbone.named_parameters())
        missing = set(st["lora"]) ^ {n for n, p in named.items() if p.requires_grad}
        if missing:
            raise RuntimeError(f"discriminator checkpoint does not match the model: {sorted(missing)[:5]}")
        with self.torch.no_grad():
            for n, w in st["lora"].items():
                named[n].copy_(w.to(named[n].device, named[n].dtype))
        self.head.load_state_dict(st["head"])
        self.opt.load_state_dict(st["optimizer"])


def calls_at_policy_step(step: int, n_ranks: int, per_gen: int) -> int:
    """step() is called once per rank per generation, and a generation feeds per_gen policy steps."""
    if step % per_gen:
        raise ValueError(f"policy step {step} is not at a generation boundary (every {per_gen} steps)")
    return step * n_ranks // per_gen


def save_checkpoint(ckpt_dir: Path, disc: VarlDiscriminator, model_state: dict, keep: int = 3) -> Path:
    """Write D's whole training state for this call count; keeps the newest `keep` checkpoints."""
    import torch

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"calls-{disc.calls}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"discriminator": disc.state(), "model": model_state}, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint under the final name
    old = sorted(ckpt_dir.glob("calls-*.pt"), key=lambda q: int(q.stem.split("-")[1]))
    for q in old[:-keep]:
        q.unlink()
    return path


def load_checkpoint(ckpt_dir: Path, calls: int) -> dict:
    import torch

    path = ckpt_dir / f"calls-{calls}.pt"
    if not path.exists():
        have = sorted(q.name for q in ckpt_dir.glob("calls-*.pt"))
        raise FileNotFoundError(f"no discriminator checkpoint {path} (have: {have}); refusing to restart D from scratch")
    return torch.load(path, weights_only=False)


class DiscriminatorClient:
    def __init__(self, url: str, timeout_s: float = 600):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def step(self, texts: list[str], passed: list[bool]) -> dict:
        return self._post(
            "/step", {"texts": texts, "passed": [bool(p) for p in passed]}
        )

    def score(self, texts: list[str]) -> list[float]:
        return self._post("/score", {"texts": texts})["scores"]

    def save(self) -> dict:
        return self._post("/save", {})


def _serve(cfg_path: Path, out_dir: Path, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from omegaconf import OmegaConf

    from src.environments.single_turn_benchmark import get_benchmark

    cfg = OmegaConf.load(cfg_path)
    v = cfg.varl
    bench = get_benchmark(str(cfg.benchmark))
    demos = [
        bench.monitor_text(d, d["response"])
        for d in bench.load_items(cfg, "demonstrations")
    ]
    model = LmDiscriminatorModel(
        str(v.get("model", cfg.policy.model)),
        "cuda:0",
        max_length=int(v.max_length),
        lora_r=int(v.lora_r),
        lora_alpha=int(v.lora_alpha),
        lr=float(v.lr),
        score_batch=int(v.get("score_batch", 16)),
    )
    n_ranks = len([g for g in str(cfg.rl.train_gpus).split(",") if g.strip()])
    per_gen = max(1, int(cfg.rl.get("steps_per_generation", cfg.rl.grad_accum)) // int(cfg.rl.grad_accum))
    disc = VarlDiscriminator(
        model.score,
        model.update,
        demos,
        # step() is called once per rank per generation; a generation may feed several policy steps
        warmup_calls=max(1, int(v.warmup_steps) * n_ranks // per_gen),
        updates_per_call=per_gen,
        accuracy_threshold=float(v.accuracy_threshold),
        batch_size=int(v.batch_size),
        fifo_size=int(v.fifo_size),
        reservoir_size=int(v.reservoir_size),
        seed=int(cfg.rl.seed),
    )
    # D is checkpointed at the same policy steps as the policy (rl.save_steps), so a stopped run can continue
    ckpt_dir = out_dir / "discriminator_checkpoints"
    every = calls_at_policy_step(int(cfg.rl.save_steps), n_ranks, per_gen)
    resume = cfg.rl.get("resume_from_checkpoint")
    if resume:
        at = calls_at_policy_step(int(Path(str(resume)).name.split("-")[-1]), n_ranks, per_gen)
        st = load_checkpoint(ckpt_dir, at)
        model.load_training_state(st["model"])
        disc.load_state(st["discriminator"])
        print(f"discriminator: resumed from {ckpt_dir / f'calls-{at}.pt'} ({disc.updates} updates so far)", flush=True)

    def _maybe_checkpoint(calls: int) -> None:
        if calls % every == 0:
            save_checkpoint(ckpt_dir, disc, model.training_state())

    disc.after_call = _maybe_checkpoint
    log_f = open(out_dir / "discriminator_steps.jsonl", "a")
    print(
        f"discriminator: {len(demos)} demonstrations, warm-up {disc.warmup_calls} calls ({n_ranks} ranks)",
        flush=True,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send({"ok": True, "calls": disc.calls, "updates": disc.updates})

        def do_POST(self):
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            )
            if self.path == "/step":
                res = disc.step(body["texts"], body["passed"])
                log_f.write(
                    json.dumps(
                        {k: res[k] for k in res if k != "scores"}
                        | {
                            "call": disc.calls,
                            "n": len(body["texts"]),
                            "n_passed": sum(body["passed"]),
                            "mean_score_passed": (
                                sum(
                                    s
                                    for s, p in zip(res["scores"], body["passed"])
                                    if p
                                )
                                / max(1, sum(body["passed"]))
                            ),
                        }
                    )
                    + "\n"
                )
                log_f.flush()
                self._send(res)
            elif self.path == "/score":
                with disc.lock:
                    self._send({"scores": model.score(body["texts"])})
            elif self.path == "/save":
                with disc.lock:
                    model.save(out_dir / "discriminator")
                self._send({"saved": str(out_dir / "discriminator")})
            else:
                self.send_error(404)

    (out_dir / "discriminator_port").write_text(str(port))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


@contextlib.contextmanager
def discriminator_server(cfg, out_dir: Path, log=print):
    """Start D on cfg.varl.gpu as a child process; yields its URL; saves D and stops it on exit."""
    from omegaconf import OmegaConf

    from src.utils.vllm_replica_serving import free_ports

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "discriminator_config.yaml"
    cfg_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    job = os.environ.get("SLURM_JOB_ID")
    port = free_ports(1, start=9000 + (int(job) % 90) * 100 if job else 9000)[0]
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg.varl.gpu))
    cmd = [
        sys.executable,
        "-m",
        "src.lagrangian_finetuning.verifier_gated_discriminator",
        "--config",
        str(cfg_path),
        "--out",
        str(out_dir),
        "--port",
        str(port),
    ]
    lf = open(out_dir / "discriminator.log", "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        stdout=lf,
        stderr=subprocess.STDOUT,
    )
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"discriminator exited ({proc.returncode}); see {out_dir / 'discriminator.log'}:\n"
                    + (out_dir / "discriminator.log").read_text()[-3000:]
                )
            try:
                urllib.request.urlopen(url, timeout=5).read()
                break
            except Exception:
                if time.time() - t0 > 1800:
                    raise RuntimeError("discriminator did not come up in 30 minutes")
                time.sleep(5)
        log(
            f"discriminator up at {url} on GPU {cfg.varl.gpu} in {time.time() - t0:.0f}s"
        )
        yield url
        with contextlib.suppress(Exception):
            DiscriminatorClient(url).save()
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
        lf.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    _serve(a.config, a.out, a.port)
