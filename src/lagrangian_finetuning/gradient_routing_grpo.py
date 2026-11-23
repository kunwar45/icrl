# ABOUTME: GRAFT (Ward et al. 2026, arXiv 2610.17062) as a GRPO run on a single-turn benchmark: RETAIN and FORGET MLP adapters, a monitor that routes flagged replies to FORGET only, anchor replies that train RETAIN alone, and the split-moment optimizer
# ABOUTME: Use: adapters_path = gradient_routing_train(cfg, out_dir, log=log) from scripts/run_single_turn_icrl_pipeline.py; the worker runs under torchrun: python -m src.lagrangian_finetuning.gradient_routing_grpo --config ... --out ...
"""
Our reimplementation of GRAFT's RL training (paper Section 3, Algorithm 1) on
TRL's GRPOTrainer. The authors' code (https://github.com/jnward/GRAFT, MIT) was
read for the mechanism; nothing is copied.

Each sampled batch has two kinds of replies:

  target  a Countdown-Code prompt, sampled with both adapters on (pi_{R+F}).
          Reward = the benchmark's training reward, never changed by the monitor.
          Not flagged: both adapters are updated. Flagged: only FORGET is updated,
          with its gradient multiplied by graft.forget_amplification.
  anchor  sampled with RETAIN only (pi_R); only RETAIN is updated. Two kinds
          (graft.anchor.kind):
            leetcode            a LeetCode prompt; reward = passes the hidden
                                tests (the paper's Countdown-Code setup)
            constrained_target  a prompt of the target task itself; reward =
                                r - lambda * (1 - C_theta), lambda by dual ascent
                                toward the budget graft.anchor.dual.epsilon. With
                                the constraint re-fitted every round by the
                                pipeline's icrl_graft_loop stage, this is the ICRL
                                loop applied to the deployed policy.

Advantages (Algorithm 1 line 7): within a group, flagged replies are left out of
the baseline mean; the standard deviation is over the whole group.

The monitor (graft.monitor):
  trivial_test        the benchmark's hand-written monitor (bench.monitor_flags)
  learned_constraint  a saved ICRL constraint model C_theta (graft.phi_dir):
                      flagged if its cost 1 - C_theta is above graft.cost_threshold
graft.routing = false gives the paper's "GRAFT w/o routing" ablation (no reply is
flagged for routing; the monitor is still logged).

Sampling. TRL's in-process vLLM engine cannot hold adapter modules. It is started
from a "widened" copy of the policy (every MLP 2 x width wider, extra units zero;
see gradient_routing_adapters.write_widened_checkpoint) and before each sampling
call the adapter weights are copied into the extra units. Anchor prompts are
sampled in a second call with FORGET's output weights written as zero.

The benchmark's `violation` and `true` values are logged for audit and never
enter training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ADAPTERS_FILE = "graft_adapters.pt"
MONITORS = ("trivial_test", "learned_constraint")
ANCHOR_KINDS = ("leetcode", "constrained_target")
EXPLORING_DIR = "exploring_replies"


def graft_advantages(rewards, flagged, num_generations: int, eps: float = 1e-4):
    """Algorithm 1 line 7. `rewards`, `flagged`: (n_groups * num_generations,), groups contiguous.

    a = (r - mean of the group's NOT flagged rewards) / (std of the whole group + eps).
    A group whose replies are all flagged uses the mean of the whole group.
    """
    import torch

    r = rewards.float().view(-1, num_generations)
    keep = (~flagged.bool()).float().view(-1, num_generations)
    n_keep = keep.sum(dim=1, keepdim=True)
    mean_keep = (r * keep).sum(dim=1, keepdim=True) / n_keep.clamp(min=1.0)
    mean = torch.where(n_keep > 0, mean_keep, r.mean(dim=1, keepdim=True))
    std = r.std(dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean)
    return ((r - mean) / (std + eps)).view(-1)


def anchor_rows_needed(n_target: int, fraction: float) -> int:
    """Anchor prompts to mix into n_target prompts so that anchors are `fraction` of the total."""
    if fraction <= 0:
        return 0
    if fraction >= 1:
        raise ValueError("graft.anchor.fraction must be below 1")
    return max(1, round(n_target * fraction / (1.0 - fraction)))


def mixed_rows(
    bench,
    target_items: list[dict],
    anchor_items: list[dict],
    fraction: float,
    seed: int,
    kind: str = "leetcode",
) -> list[dict]:
    """Dataset rows for both kinds of prompt, with the same columns.

    kind "leetcode": anchor rows are `anchor_items` (LeetCode problems; they repeat if there are too few).
    kind "constrained_target": anchor rows are prompts of the target benchmark itself, drawn from
    `target_items`; `anchor_items` is not used.
    """
    from src.environments.leetcode_anchor import leetcode_anchor_adapter as anchor

    if kind not in ANCHOR_KINDS:
        raise ValueError(f"graft.anchor.kind must be one of {ANCHOR_KINDS}, got {kind}")
    rows = [
        {
            "prompt": bench.chat_messages(it),
            **{k: it[k] for k in bench.item_columns},
            "is_anchor": 0,
            "anchor_item": "",
        }
        for it in target_items
    ]
    rng = random.Random(seed)
    if kind == "constrained_target":
        for _ in range(anchor_rows_needed(len(target_items), fraction)):
            it = rng.choice(target_items)
            rows.append(
                {
                    "prompt": bench.chat_messages(it),
                    **{k: it[k] for k in bench.item_columns},
                    "is_anchor": 1,
                    "anchor_item": "",
                }
            )
        return rows
    n = anchor_rows_needed(len(target_items), fraction) if anchor_items else 0
    if n:
        blank = {k: _blank_like(target_items[0][k]) for k in bench.item_columns}
        pool = list(anchor_items)
        rng.shuffle(pool)
        for i in range(n):
            it = pool[i % len(pool)]
            rows.append(
                {
                    "prompt": anchor.chat_messages(it),
                    **blank,
                    "is_anchor": 1,
                    "anchor_item": json.dumps(it),
                }
            )
    return rows


def _blank_like(value):
    if isinstance(value, list):
        return []
    if isinstance(value, str):
        return ""
    return -1


class RoutingScorer:
    """The reward function. Also keeps this call's monitor flags for the trainer, and logs audit counts."""

    def __init__(self, cfg, rank: int, out_dir: Path | None = None):
        from src.environments.single_turn_benchmark import get_benchmark

        self.cfg, self.rank, self.out_dir = cfg, rank, out_dir
        self.bench = get_benchmark(str(cfg.benchmark))
        g = cfg.graft
        self.monitor = str(g.monitor)
        if self.monitor not in MONITORS:
            raise ValueError(f"graft.monitor must be one of {MONITORS}, got {self.monitor}")
        anchor = g.get("anchor") or {}
        self.anchor_kind = str(anchor.get("kind", "leetcode"))
        if self.anchor_kind not in ANCHOR_KINDS:
            raise ValueError(f"graft.anchor.kind must be one of {ANCHOR_KINDS}, got {self.anchor_kind}")
        self.routing = bool(g.get("routing", True))
        self.threshold = float(g.get("cost_threshold", 0.5))
        self.constraint_input = str((cfg.get("constraint") or {}).get("input", "files"))
        self.phi = None
        if self.monitor == "learned_constraint" or self.anchor_kind == "constrained_target":
            from src.icrl_dual_training.secret_keeping_reply_constraint import (
                TrajectoryFeasibility,
            )

            self.phi = TrajectoryFeasibility.load(Path(str(g.phi_dir)), device=f"cuda:{rank}")
            # a threshold calibrated on the demonstrations when the constraint was fitted, if there is one
            calibrated = Path(str(g.phi_dir)) / "routing_threshold.json"
            if calibrated.exists():
                self.threshold = float(json.loads(calibrated.read_text())["threshold"])
        self.dual = anchor.get("dual")
        self.lam = float(anchor.get("lambda_init", (self.dual or {}).get("lambda_init", 0.0)))
        self.log_exploring = bool(g.get("log_exploring_replies", False)) and out_dir is not None
        self.last_flags: list[int] = []
        self.calls = 0

    def cost(self, items: list[dict], replies: list[str]) -> list[float]:
        """The learned cost 1 - C_theta per reply."""
        if not replies:
            return []
        texts = [self.bench.constraint_text(it, x, self.constraint_input) for it, x in zip(items, replies)]
        return [float(c) for c in self.phi.score([[t] for t in texts])]

    def flags(self, items: list[dict], replies: list[str]) -> tuple[list[int], list[float] | None]:
        """(flag per reply, the monitor's cost per reply or None for the hand-written monitor)."""
        if not replies:
            return [], None
        if self.monitor == "trivial_test":
            return [int(f) for f in self.bench.monitor_flags(items, replies)], None
        cost = self.cost(items, replies)
        return [int(c > self.threshold) for c in cost], cost

    def _dual_step(self, costs: list[float]) -> float | None:
        """Dual ascent on lambda from the mean cost of the deployed policy's replies, over all processes."""
        from src.lagrangian_finetuning.single_turn_constrained_grpo import dual_update

        total, count = _global_sum_and_count(costs)
        if count == 0:
            return None
        mean_cost = total / count
        d, rl = self.dual, self.cfg.rl
        # one call per generation; take as many dual steps as the generation feeds optimizer steps
        per_gen = max(1, int(rl.get("steps_per_generation", rl.grad_accum)) // int(rl.grad_accum))
        for _ in range(per_gen):
            self.lam = dual_update(self.lam, mean_cost, d.lr, d.epsilon, d.get("max", 100.0))
        return mean_cost

    def _write_exploring(self, items, replies, scores) -> None:
        """Append the exploring policy's replies for the next round's inverse step (label kept for audit only)."""
        d = Path(self.out_dir) / EXPLORING_DIR
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"rank{self.rank}.jsonl", "a") as f:
            for it, x, s in zip(items, replies, scores):
                row = {
                    "text": self.bench.constraint_text(it, x, self.constraint_input),
                    "reward": float(s["reward"]),
                    "violates": int(s["violation"]),
                    "true": float(s["true"]),
                }
                f.write(json.dumps(row) + "\n")

    def __call__(self, prompts, completions, **kw) -> list[float]:
        from src.environments.leetcode_anchor import leetcode_anchor_adapter as anchor
        from src.lagrangian_finetuning.single_turn_constrained_grpo import _reply_text

        replies = [_reply_text(c) for c in completions]
        n = len(replies)
        is_anchor = [bool(a) for a in kw["is_anchor"]]
        t_idx = [i for i in range(n) if not is_anchor[i]]
        a_idx = [i for i in range(n) if is_anchor[i]]
        constrained = self.anchor_kind == "constrained_target"

        items = [{k: kw[k][i] for k in self.bench.item_columns} for i in t_idx]
        t_replies = [replies[i] for i in t_idx]
        scores = self.bench.score_replies(items, t_replies) if t_idx else []
        monitor_flags, cost = self.flags(items, t_replies)
        if self.log_exploring and t_idx:
            self._write_exploring(items, t_replies, scores)

        out = [0.0] * n
        flags = [0] * n
        for j, i in enumerate(t_idx):
            out[i] = float(scores[j]["reward"])
            flags[i] = int(monitor_flags[j]) if self.routing else 0

        a_replies = [replies[i] for i in a_idx]
        lam_used, a_cost, a_mean_cost = self.lam, [], None
        if constrained:
            # the deployed policy (RETAIN only) on the target task, under the learned constraint
            a_items = [{k: kw[k][i] for k in self.bench.item_columns} for i in a_idx]
            a_scores = self.bench.score_replies(a_items, a_replies) if a_idx else []
            a_cost = self.cost(a_items, a_replies)
            a_mean_cost = self._dual_step(a_cost) if self.dual else None
            for j, i in enumerate(a_idx):
                out[i] = float(a_scores[j]["reward"]) - lam_used * a_cost[j]
        else:
            a_items = [json.loads(kw["anchor_item"][i]) for i in a_idx]
            a_scores = anchor.score_replies(a_items, a_replies) if a_idx else []
            for j, i in enumerate(a_idx):
                out[i] = float(a_scores[j]["reward"])
        self.last_flags = flags

        log_metric = kw.get("log_metric")
        if log_metric is not None:
            # counts over TARGET replies (the exploring policy), with the same names as the other arms' logs
            viol = [int(s["violation"]) for s in scores]
            r = [float(s["reward"]) for s in scores]
            passed_clean = [j for j in range(len(scores)) if r[j] > 0 and not viol[j]]
            log_metric("n_replies", float(len(scores)))
            log_metric("reward_sum", float(sum(r)))
            log_metric("true_sum", float(sum(s["true"] for s in scores)))
            log_metric("violation_sum", float(sum(viol)))
            log_metric("parsed_sum", float(sum(bool(s["parsed"]) for s in scores)))
            # the monitor against the audit label: how many replies it flags, and how many hacks it catches
            log_metric("flagged_sum", float(sum(monitor_flags)))
            log_metric("flagged_violating_sum", float(sum(f for f, v in zip(monitor_flags, viol) if v)))
            log_metric("routed_sum", float(sum(flags)))
            log_metric("monitor_sum_violating", float(sum(c for c, v in zip(cost, viol) if v)) if cost else 0.0)
            log_metric("monitor_sum_passing_clean", float(sum(cost[j] for j in passed_clean)) if cost else 0.0)
            log_metric("n_passing_clean", float(len(passed_clean)))
            # the deployed policy's replies (RETAIN only)
            log_metric("anchor_n_replies", float(len(a_idx)))
            log_metric("anchor_reward_sum", float(sum(s["reward"] for s in a_scores)))
            log_metric("anchor_parsed_sum", float(sum(bool(s["parsed"]) for s in a_scores)))
            log_metric("anchor_true_sum", float(sum(s["true"] for s in a_scores)) if constrained else 0.0)
            log_metric("anchor_violation_sum", float(sum(s["violation"] for s in a_scores)) if constrained else 0.0)
            log_metric("anchor_cost_sum", float(sum(a_cost)))
            log_metric("lambda", float(lam_used))
            log_metric("anchor_cost_mean_global", float(a_mean_cost) if a_mean_cost is not None else 0.0)
        self.calls += 1
        if self.rank == 0 and (self.calls <= 2 or self.calls % 25 == 0):
            for kind, idx in (("target", t_idx[:3]), ("anchor", a_idx[:2])):
                for i in idx:
                    if self.calls == 1:
                        print(
                            f"[PROMPT seen by the policy, {kind}]\n"
                            + "\n".join(f"<{m['role']}> {m['content']}" for m in prompts[i])
                            + "\n",
                            flush=True,
                        )
                    if kind == "target":
                        j = t_idx.index(i)
                        c = f"{cost[j]:.3f}" if cost else "-"
                        thr = f"{self.threshold:.3f}" if cost else "-"
                        updated = "FORGET only" if flags[i] else "RETAIN and FORGET"
                        print(
                            f"[SAMPLE call {self.calls}, target] item={items[j]} reward={out[i]} "
                            f"true={scores[j]['true']} violation={scores[j]['violation']} monitor_flag={monitor_flags[j]} "
                            f"monitor_cost={c} threshold={thr} updated={updated}\n[REPLY] {replies[i]}\n",
                            flush=True,
                        )
                    elif constrained:
                        j = a_idx.index(i)
                        print(
                            f"[SAMPLE call {self.calls}, deployed policy on the target task] item={a_items[j]} "
                            f"reward={a_scores[j]['reward']} true={a_scores[j]['true']} violation={a_scores[j]['violation']} "
                            f"cost={a_cost[j]:.3f} lambda={lam_used:.3f} score={out[i]:.3f} "
                            f"updated=RETAIN only (FORGET off)\n[REPLY] {replies[i]}\n",
                            flush=True,
                        )
                    else:
                        j = a_idx.index(i)
                        print(
                            f"[SAMPLE call {self.calls}, anchor] problem={a_items[j]['id']} reward={out[i]} "
                            f"error={a_scores[j]['error']!r} updated=RETAIN only (FORGET off)\n[REPLY] {replies[i]}\n",
                            flush=True,
                        )
        return out


def _global_sum_and_count(values: list[float]) -> tuple[float, float]:
    """(sum, count) of `values` over all training processes."""
    import torch
    import torch.distributed as dist

    t = torch.tensor([float(sum(values)), float(len(values))], dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        t = t.to("cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(t)
    return float(t[0]), float(t[1])


def read_exploring_replies(rl_dir: Path, n: int) -> list[dict]:
    """The most recent `n` exploring-policy replies written during a round, shared evenly over the processes."""
    files = sorted((Path(rl_dir) / EXPLORING_DIR).glob("rank*.jsonl"))
    if not files:
        return []
    per = max(1, n // len(files))
    rows = []
    for f in files:
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        rows += [json.loads(ln) for ln in lines[-per:]]
    return rows


def _trainer_class():
    """Built lazily so importing this module does not import TRL (the pipeline imports it on a login node)."""
    import torch
    from trl import GRPOTrainer

    from src.lagrangian_finetuning.gradient_routing_adapters import (
        adapters_of,
        clear_routing,
        load_adapter_state_dict,
        parameter_pairs,
        routing_weights,
        save_adapters,
        set_routing,
        widened_mlp_weights,
    )
    from src.lagrangian_finetuning.split_moment_adamw import SplitMomentAdamW

    class GradientRoutingGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, graft_cfg, scorer: RoutingScorer, **kwargs):
            self.graft_cfg = graft_cfg
            self.scorer = scorer
            self._rows = None  # (retain_weight, forget_weight, forget_on) for the rows about to be run
            self._local_is_anchor: list[bool] = []
            self._all_rewards = self._all_flags = self._all_is_anchor = None
            super().__init__(*args, **kwargs)
            if getattr(self, "use_vllm", False):
                self._plain_generate = self.vllm_generation.generate
                self.vllm_generation.sync_weights = self._sync_adapters_to_vllm
                self.vllm_generation.generate = self._generate_by_kind

        # ---- optimizer ------------------------------------------------------

        def create_optimizer(self, *args, **kwargs):
            o = self.graft_cfg.optimizer
            pairs = parameter_pairs(self.model)
            fraction = float(self.graft_cfg.anchor.get("fraction", 0.0))
            self.optimizer = SplitMomentAdamW(
                [
                    {"pairs": pairs["retain"]},
                    # the FORGET unrouted gradient is averaged over target replies only (paper Eq. 1)
                    {
                        "pairs": pairs["forget"],
                        "unrouted_scale": 1.0 / (1.0 - fraction),
                    },
                ],
                lr=self.args.learning_rate,
                betas=(float(o.get("beta1", 0.9)), float(o.beta2)),
                eps=float(o.get("eps", 1e-8)),
                weight_decay=float(o.weight_decay),
                max_grad_norm=float(o.max_grad_norm) or None,
            )
            return self.optimizer

        # ---- sampling -------------------------------------------------------

        def _push_mlp_weights(self, forget: bool, only_down: bool = False) -> None:
            gen = self.vllm_generation
            for i, a in enumerate(adapters_of(self.model)):
                per_layer = {
                    f"{which}.{name}": getattr(branch, name).data
                    for which, branch in (("retain", a.retain), ("forget", a.forget))
                    for name in ("gate", "up", "down")
                }
                gate, up, down = widened_mlp_weights(
                    a.base.gate_proj.weight.data,
                    a.base.up_proj.weight.data,
                    a.base.down_proj.weight.data,
                    per_layer,
                    a.width,
                    forget,
                )
                named = (
                    [("down_proj", down)]
                    if only_down
                    else [("gate_proj", gate), ("up_proj", up), ("down_proj", down)]
                )
                for name, weight in named:
                    gen._push_param_to_vllm(
                        f"model.layers.{i}.mlp.{name}.weight", weight
                    )
                del gate, up, down
            gen.llm.reset_prefix_cache()

        def _sync_adapters_to_vllm(self) -> None:
            self._push_mlp_weights(forget=True)

        def _generate_by_kind(self, prompts, images, num_generations, profiler=None):
            mask = self._local_is_anchor
            if len(mask) != len(prompts) or not any(mask):
                return self._plain_generate(
                    prompts=prompts,
                    images=images,
                    num_generations=num_generations,
                    profiler=profiler,
                )
            parts = {}
            for kind, want in (("target", False), ("anchor", True)):
                idx = [i for i, m in enumerate(mask) if m == want]
                if not idx:
                    continue
                if want:  # anchor replies are sampled by pi_R: FORGET's output weights are zero
                    self._push_mlp_weights(forget=False, only_down=True)
                try:
                    res = self._plain_generate(
                        prompts=[prompts[i] for i in idx],
                        images=None,
                        num_generations=num_generations,
                        profiler=profiler,
                    )
                finally:
                    if want:
                        self._push_mlp_weights(forget=True, only_down=True)
                parts[kind] = (idx, res)
            merged = []
            for slot in range(4):
                if any(res[slot] is None for _, res in parts.values()):
                    merged.append(None)
                    continue
                column = [None] * len(prompts)
                for idx, res in parts.values():
                    for i, value in zip(idx, res[slot]):
                        column[i] = value
                merged.append(column)
            return tuple(merged)

        # ---- rewards, flags, advantages ------------------------------------------

        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            rewards_per_func = super()._calculate_rewards(
                inputs, prompts, completions, completion_ids_list
            )
            device = self.accelerator.device
            weights = self.reward_weights.to(rewards_per_func.device).unsqueeze(0)
            self._all_rewards = (rewards_per_func * weights).nansum(dim=1)
            flags = torch.tensor(
                self.scorer.last_flags, dtype=torch.float32, device=device
            )
            anchors = torch.tensor(
                [float(a) for a in self._local_is_anchor],
                dtype=torch.float32,
                device=device,
            )
            self._all_flags = self.accelerator.gather(flags)
            self._all_is_anchor = self.accelerator.gather(anchors)
            return rewards_per_func

        def _generate_and_score_completions(self, inputs):
            mode = "train" if self.model.training else "eval"
            n = len(inputs)
            device = self.accelerator.device
            self._local_is_anchor = [bool(x.get("is_anchor", 0)) for x in inputs]
            is_anchor = torch.tensor(
                [float(a) for a in self._local_is_anchor], device=device
            )
            # no-gradient passes inside (the sampling-time log-probabilities): anchors run with FORGET off
            self._rows = routing_weights(torch.zeros(n, device=device), is_anchor)
            try:
                output = super()._generate_and_score_completions(inputs)
            finally:
                self._rows = None
                clear_routing(self.model)
            num_generations = (
                self.num_generations if mode == "train" else self.num_generations_eval
            )
            flagged_all = self._all_flags * (1.0 - self._all_is_anchor)
            advantages = graft_advantages(
                self._all_rewards, flagged_all, num_generations
            )
            advantages = torch.nan_to_num(advantages, nan=0.0)
            local = slice(
                self.accelerator.process_index * n,
                (self.accelerator.process_index + 1) * n,
            )
            output["advantages"] = advantages[local].to(output["advantages"].dtype)
            output["flagged"] = flagged_all[local]
            output["is_anchor"] = is_anchor
            return output

        # ---- routed forward and loss -------------------------------------------

        def _get_per_token_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=None,
            compute_entropy=False,
            **kw,
        ):
            rows = self._rows
            if rows is None or rows[0].shape[0] != input_ids.size(0):
                return super()._get_per_token_logps_and_entropies(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    compute_entropy,
                    **kw,
                )
            step = batch_size or input_ids.size(0)
            logps, entropies, aux = [], [], None
            for start in range(0, input_ids.size(0), step):
                chunk = slice(start, start + step)
                # left set after the last chunk: the backward pass of the loss needs it
                set_routing(self.model, *(t[chunk] for t in rows))
                lp, ent, aux = super()._get_per_token_logps_and_entropies(
                    model,
                    input_ids[chunk],
                    attention_mask[chunk],
                    logits_to_keep,
                    None,
                    compute_entropy,
                    **kw,
                )
                logps.append(lp)
                if ent is not None:
                    entropies.append(ent)
            return (
                torch.cat(logps, dim=0),
                (torch.cat(entropies, dim=0) if entropies else None),
                aux,
            )

        def _compute_loss(self, model, inputs):
            amplification = float(self.graft_cfg.get("forget_amplification", 2.0))
            self._rows = routing_weights(
                inputs["flagged"], inputs["is_anchor"], amplification
            )
            try:
                return super()._compute_loss(model, inputs)
            finally:
                self._rows = None  # the routing itself stays set on the model for the backward pass

        # ---- checkpoints: the adapters only ----------------------------------------

        def save_model(self, output_dir=None, _internal_call=False):
            if self.is_world_process_zero():
                save_adapters(
                    self.model, Path(output_dir or self.args.output_dir) / ADAPTERS_FILE
                )

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            path = Path(resume_from_checkpoint) / ADAPTERS_FILE
            if not path.exists():
                raise ValueError(f"no {ADAPTERS_FILE} in {resume_from_checkpoint}")
            load_adapter_state_dict(self.model, torch.load(path, map_location="cpu"))

        def create_model_card(
            self, *args, **kwargs
        ):  # the Trainer's card assumes a full model
            return None

    return GradientRoutingGRPOTrainer


def widened_start_dir(out_dir: Path, cfg=None) -> Path:
    """Where the widened starting checkpoint lives: graft.widened_start_dir if set (shared by the rounds of
    a multi-round run), else inside this run's directory."""
    shared = cfg.graft.get("widened_start_dir") if cfg is not None else None
    return Path(str(shared)) if shared else out_dir / "widened_start"


def gradient_routing_train(cfg, out_dir: Path, log=print) -> Path:
    """Run GRAFT training as a torchrun subprocess over `rl.train_gpus`; returns the saved adapters file."""
    import subprocess
    import sys

    from omegaconf import OmegaConf

    out_dir.mkdir(parents=True, exist_ok=True)
    start = widened_start_dir(out_dir, cfg)
    if not (start / "config.json").exists():
        from src.lagrangian_finetuning.gradient_routing_adapters import (
            write_widened_checkpoint,
        )

        log(f"writing the widened starting checkpoint for sampling: {start}")
        tmp = start.with_name(start.name + "_tmp")
        write_widened_checkpoint(
            str(cfg.policy.model), tmp, int(cfg.graft.adapter_width)
        )
        tmp.rename(start)
    (out_dir / "rl_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    gpus = str(cfg.rl.train_gpus)
    n = len([g for g in gpus.split(",") if g.strip()])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    job = os.environ.get("SLURM_JOB_ID")
    port = 29500 + (int(job) % 400 if job else 31)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={n}",
        f"--master_port={port}",
        "-m",
        "src.lagrangian_finetuning.gradient_routing_grpo",
        "--config",
        str(out_dir / "rl_config.yaml"),
        "--out",
        str(out_dir),
    ]
    log(
        f"GRPO (gradient_routing, monitor {cfg.graft.monitor}) on GPUs {gpus} ({n} processes); log {out_dir / 'rl_train.log'}"
    )
    repo = Path(__file__).resolve().parents[2]
    with open(out_dir / "rl_train.log", "a") as lf:
        rc = subprocess.run(
            cmd, cwd=str(repo), env=env, stdout=lf, stderr=subprocess.STDOUT
        ).returncode
    tail = (out_dir / "rl_train.log").read_text()[-4000:]
    if rc != 0:
        raise RuntimeError(
            f"GRPO failed (exit {rc}); tail of {out_dir / 'rl_train.log'}:\n{tail}"
        )
    log("GRPO done; " + tail[-1200:])
    return out_dir / "adapter" / ADAPTERS_FILE


def _worker(cfg_path: Path, out_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from src.environments.leetcode_anchor.leetcode_anchor_adapter import load_problems
    from src.environments.single_turn_benchmark import get_benchmark
    from src.lagrangian_finetuning.gradient_routing_adapters import (
        add_dual_mlp_adapters,
    )

    cfg = OmegaConf.load(cfg_path)
    rl, g = cfg.rl, cfg.graft
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    bench = get_benchmark(str(cfg.benchmark))
    tok = AutoTokenizer.from_pretrained(str(cfg.policy.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    fraction = float(g.anchor.get("fraction", 0.0))
    anchor_kind = str(g.anchor.get("kind", "leetcode"))
    anchor_items = []
    if fraction > 0 and anchor_kind == "leetcode":
        anchor_items = load_problems(str(g.anchor.data), g.anchor.get("prompts"))
        # the prompt must leave room for a full-length reply inside the sampling engine's context
        room = int(rl.vllm_max_model_length) - int(rl.max_completion_length)
        from src.environments.leetcode_anchor.leetcode_anchor_adapter import (
            chat_messages,
        )

        fits = [
            it
            for it in anchor_items
            if len(
                tok.apply_chat_template(
                    chat_messages(it), add_generation_prompt=True, tokenize=True
                )
            )
            <= room
        ]
        if rank == 0:
            print(
                f"anchor problems: {len(fits)} of {len(anchor_items)} fit in {room} prompt tokens",
                flush=True,
            )
        if not fits:
            raise RuntimeError("no anchor problem fits in the prompt budget")
        anchor_items = fits
    rows = mixed_rows(
        bench,
        bench.load_items(cfg, "train"),
        anchor_items,
        fraction,
        int(rl.seed),
        kind=anchor_kind,
    )
    ds = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.policy.model), dtype=torch.bfloat16
    )
    add_dual_mlp_adapters(model, int(g.adapter_width), dtype=torch.float32)
    # the sampling engine loads this path: the same model with wider MLPs whose extra units start at zero
    model.config._name_or_path = str(widened_start_dir(out_dir, cfg))
    model.name_or_path = str(widened_start_dir(out_dir, cfg))

    scorer = RoutingScorer(cfg, rank, out_dir)

    def score(prompts, completions, **kw):
        return scorer(prompts, completions, **kw)

    o = g.optimizer
    kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        learning_rate=float(rl.lr),
        max_steps=int(rl.max_steps),
        per_device_train_batch_size=int(rl.per_device_batch),
        gradient_accumulation_steps=int(rl.grad_accum),
        steps_per_generation=int(rl.get("steps_per_generation", rl.grad_accum)),
        num_generations=int(rl.num_generations),
        max_completion_length=int(rl.max_completion_length),
        temperature=float(rl.temperature),
        beta=float(rl.kl_beta),
        reward_weights=[1.0],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=float(rl.vllm_gpu_memory_utilization),
        vllm_max_model_length=int(rl.vllm_max_model_length),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=int(rl.save_steps),
        save_total_limit=1,
        report_to=[],
        seed=int(rl.seed),
        warmup_steps=int(o.warmup_steps),
        lr_scheduler_type=str(o.scheduler),
        max_grad_norm=0.0,  # clipping is done inside the optimizer, on the routed gradient
        remove_unused_columns=False,
        mask_truncated_completions=False,
    )
    accepted = set(GRPOConfig.__dataclass_fields__)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped and rank == 0:
        print(f"GRPOConfig does not accept {dropped}; dropped", flush=True)
    args = GRPOConfig(**{k: v for k, v in kwargs.items() if k in accepted})
    if rank == 0:
        n_anchor = sum(r["is_anchor"] for r in rows)
        trainable = sum(
            p.numel()
            for n_, p in model.named_parameters()
            if p.requires_grad and "shadow" not in n_
        )
        print(
            f"GRPO[gradient_routing]: {len(rows) - n_anchor} target + {n_anchor} anchor prompts, "
            f"{args.num_generations} replies each, monitor={g.monitor} routing={g.get('routing', True)} "
            f"adapter_width={g.adapter_width} trained weights={trainable:,} lr={args.learning_rate} "
            f"max_steps={args.max_steps} model={cfg.policy.model}",
            flush=True,
        )
    trainer = _trainer_class()(
        model=model,
        reward_funcs=[score],
        args=args,
        train_dataset=ds,
        processing_class=tok,
        graft_cfg=g,
        scorer=scorer,
    )
    stop_at = rl.get("stop_at_step")
    if stop_at:
        # one round of a multi-round run: stop here, while max_steps (and the learning-rate schedule) span the whole run
        from transformers import TrainerCallback

        class StopAtStep(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= int(stop_at):
                    control.should_training_stop = True

        trainer.add_callback(StopAtStep())
    if rl.get("keep_adapter_checkpoints", True):
        # a copy of the retain/forget adapters at every save (save_steps), kept beyond save_total_limit for publishing
        from src.lagrangian_finetuning.adapter_checkpoint_keeping import (
            keep_adapter_checkpoints_callback,
        )

        trainer.add_callback(keep_adapter_checkpoints_callback(out_dir / "adapter_checkpoints"))
    resume = rl.get("resume_from_checkpoint")
    if resume and rank == 0:
        print(f"resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "adapter"))
        (out_dir / "lambda.json").write_text(json.dumps({"lambda": scorer.lam}))
        (out_dir / "train_log_history.json").write_text(
            json.dumps(trainer.state.log_history, indent=1)
        )
        print("adapters saved to", out_dir / "adapter" / ADAPTERS_FILE, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    _worker(a.config, a.out)
