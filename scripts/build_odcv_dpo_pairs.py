#!/usr/bin/env python
# ABOUTME: Builds DPO preference pairs (expert = chosen, unsafe = rejected, same ODCV cell) rendered in the policy's chat template
# ABOUTME: Run: python scripts/build_odcv_dpo_pairs.py --set-root data/odcv/numina_control_temp07_audited --output data/odcv/dpo_pairs --tokenizer Qwen/Qwen3.6-27B
"""
The DPO baseline for the ODCV track: does imitating audited expert traces over
unsafe traces of the SAME cell move held-out misalignment, without any learned
constraint? Pairs are built only within a cell (scenario + framing), so the
prompt is identical for chosen and rejected and the preference is purely about
behaviour. Scenarios in the constraint head's held-out split stay held out
here too, so a later evaluation on them is clean.

Each trace is re-rendered through the policy tokenizer's chat template with
real tool_calls / tool-response roles and the model's reasoning as
`reasoning_content`, then split into `prompt` (system + user + generation
prefix) and the completion. Observations are the 300-char head/tail slices the
trace files carry; the tool-call format is exactly what vLLM's qwen3_xml parser
serves at inference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Executes shell commands. Put the command in the 'command' field.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "To mark the task as completion.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]


def call_of(action: str):
    if action.startswith("bash: "):
        return {"name": "bash", "arguments": {"command": action[len("bash: ") :]}}
    if action.startswith("task_complete: "):
        return {
            "name": "task_complete",
            "arguments": {"reason": action[len("task_complete: ") :]},
        }
    return None


def messages_of(trace: dict) -> list[dict]:
    sp = trace["scenario_prompt"]
    system = sp.split("[user]", 1)[0].replace("[system]", "").strip()
    user = sp.split("[user]", 1)[1].strip() if "[user]" in sp else ""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    steps = trace["steps"]
    for i, s in enumerate(steps):
        c = call_of(s["action"])
        reasoning = s.get("reasoning") or ""
        if c is None:
            msgs.append(
                {
                    "role": "assistant",
                    "content": s["action"].removeprefix("message: "),
                    "reasoning_content": reasoning,
                }
            )
            continue
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [{"type": "function", "function": c}],
            }
        )
        # the result of this call is what the NEXT step saw before acting
        if i + 1 < len(steps):
            msgs.append(
                {"role": "tool", "content": steps[i + 1].get("observation") or ""}
            )
    return msgs


def render(tok, msgs: list[dict]) -> tuple[str, str]:
    full = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False)
    prefix = tok.apply_chat_template(
        msgs[:2], tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    if not full.startswith(prefix):
        raise ValueError(
            "chat template prefix does not match; check the template's generation prompt"
        )
    return prefix, full[len(prefix) :]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--set-root",
        type=Path,
        required=True,
        help="audited set with expert/, unsafe/ and split/splits.json",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--max-pairs-per-cell", type=int, default=12)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=6144,
        help="drop a pair whose prompt + longer completion exceeds this",
    )
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    held = set(
        json.load(open(a.set_root / "split" / "splits.json"))["held_out_task_ids"]
    )
    rng = random.Random(a.seed)

    cells: dict[tuple, dict] = defaultdict(lambda: {"expert": [], "unsafe": []})
    for label in ("expert", "unsafe"):
        for p in sorted((a.set_root / label).glob("task_*_trace_*.json")):
            t = json.loads(p.read_text())
            cells[(t["task_id"], t["task_variant"])][label].append((p.name, t))

    a.output.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)
    out = {"train": [], "eval": []}
    for (scenario, variant), v in sorted(cells.items()):
        if not v["expert"] or not v["unsafe"]:
            continue
        split = "eval" if scenario in held else "train"
        rendered = {}
        for label in ("expert", "unsafe"):
            for name, t in v[label]:
                pr, comp = render(tok, messages_of(t))
                rendered[(label, name)] = (
                    pr,
                    comp,
                    len(tok(pr)["input_ids"]),
                    len(tok(comp)["input_ids"]),
                )
        pairs = list(
            itertools.product([n for n, _ in v["expert"]], [n for n, _ in v["unsafe"]])
        )
        rng.shuffle(pairs)
        kept = 0
        for e, u in pairs:
            if kept >= a.max_pairs_per_cell:
                break
            pe, ce, npe, nce = rendered[("expert", e)]
            pu, cu, npu, ncu = rendered[("unsafe", u)]
            if npe + max(nce, ncu) > a.max_tokens:
                stats[f"{split}_dropped_too_long"] += 1
                continue
            out[split].append(
                {
                    "prompt": pe,
                    "chosen": ce,
                    "rejected": cu,
                    "scenario": scenario,
                    "variant": variant,
                    "chosen_file": e,
                    "rejected_file": u,
                    "prompt_tokens": npe,
                    "chosen_tokens": nce,
                    "rejected_tokens": ncu,
                }
            )
            kept += 1
        stats[f"{split}_cells"] += 1
    for split, rows in out.items():
        with open(a.output / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        toks = [
            r["prompt_tokens"] + max(r["chosen_tokens"], r["rejected_tokens"])
            for r in rows
        ]
        print(
            f"{split}: {len(rows)} pairs over {stats[f'{split}_cells']} cells, "
            f"tokens median {sorted(toks)[len(toks) // 2] if toks else 0} max {max(toks) if toks else 0}, "
            f"dropped too long {stats[f'{split}_dropped_too_long']}"
        )
    (a.output / "manifest.json").write_text(
        json.dumps(
            {
                "set_root": str(a.set_root),
                "tokenizer": a.tokenizer,
                "held_out_scenarios": sorted(held),
                "max_pairs_per_cell": a.max_pairs_per_cell,
                "max_tokens": a.max_tokens,
                "seed": a.seed,
                "n_train": len(out["train"]),
                "n_eval": len(out["eval"]),
                "stats": dict(stats),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
