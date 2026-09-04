# ABOUTME: ODCV transcript handling shared by import, DPO pairs and the fine-tuning loop: parse messages_record.txt, render chat-template text, actions-only text
# ABOUTME: Import: from src.environments.odcv.transcripts import transcript_to_steps, steps_to_actions_text, messages_of, render_prompt_completion
"""
One place for the three transformations every ODCV stage needs:

  messages_record.txt  ->  steps      (transcript_to_steps)   what the agent saw and did
  steps                ->  text       (steps_to_actions_text) the constraint head's input
  steps                ->  chat turns (messages_of, render_prompt_completion) what the policy is trained on

The step parser is the same one scripts/import_odcv_rollouts.py carries; the
renderer the same one scripts/build_odcv_dpo_pairs.py carries. They live here
so the fine-tuning loop cannot drift from the data those scripts produced.
"""

from __future__ import annotations

import ast
import json
import re

STEP_RE = re.compile(r"^== Step \d+ ==\s*$", re.M)
FIELD_RE = re.compile(r"^(role|content|reason|call|tool_calls):\s?(.*)$")
SUBMIT_MARKER = '"action": "task_complete"'

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


def parse_fields(block: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    key = None
    for line in block.splitlines():
        m = FIELD_RE.match(line)
        if m:
            key = m.group(1)
            out.setdefault(key, []).append(m.group(2))
        elif key is not None:
            out[key].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def parse_calls(call_text: str) -> list[dict]:
    if not call_text or call_text in ("None", "[]"):
        return []
    try:
        raw = ast.literal_eval(call_text)
    except (ValueError, SyntaxError):
        return [{"name": "unparsed", "args": {"raw": call_text[:500]}}]
    calls = []
    for c in raw if isinstance(raw, list) else [raw]:
        fn = (c or {}).get("function", {}) if isinstance(c, dict) else {}
        name = fn.get("name", "unknown")
        args_s = fn.get("arguments", "")
        try:
            args = json.loads(args_s) if isinstance(args_s, str) else (args_s or {})
        except json.JSONDecodeError:
            args = {"raw": str(args_s)[:500]}
        calls.append(
            {
                "name": name,
                "args": args if isinstance(args, dict) else {"raw": str(args)},
            }
        )
    return calls


def action_string(call: dict) -> str:
    name, args = call["name"], call["args"]
    if name == "bash":
        return f"bash: {args.get('command', args.get('raw', ''))}"
    if name == "task_complete":
        return f"task_complete: {args.get('reason', args.get('raw', ''))}"
    return f"{name}: {json.dumps(args, ensure_ascii=False)}"


def tool_result_text(content: str) -> tuple[str, int | None]:
    try:
        d = json.loads(content)
    except json.JSONDecodeError:
        return content, None
    if not isinstance(d, dict):
        return content, None
    if d.get("action") == "task_complete":
        return f"[task_complete acknowledged: {d.get('status', '')}]", None
    parts = []
    if d.get("stdout"):
        parts.append(d["stdout"].rstrip())
    if d.get("stderr"):
        parts.append("[stderr] " + d["stderr"].rstrip())
    return "\n".join(parts), d.get("returncode")


def truncate(text: str, keep: int) -> str:
    if keep <= 0 or len(text) <= 2 * keep + 30:
        return text
    cut = len(text) - 2 * keep
    return f"{text[:keep]} …[{cut} chars cut]… {text[-keep:]}"


def transcript_to_steps(
    transcript: str, obs_chars: int = 300
) -> tuple[list[dict], dict]:
    """Pair every tool call with what the agent saw BEFORE issuing it (the previous call's result)."""
    blocks = [b for b in STEP_RE.split(transcript) if b.strip()]
    framing: list[str] = []
    pairs: list[dict] = []
    pending: list[int] = []
    n_text_only = 0
    for b in blocks:
        f = parse_fields(b)
        role = f.get("role", "assistant")
        if role in ("system", "user"):
            if f.get("content"):
                framing.append(f"[{role}] {f['content']}")
            continue
        if role == "tool":
            text, rc = tool_result_text(f.get("content", ""))
            if pending:
                i = pending.pop(0)
                pairs[i]["result"], pairs[i]["returncode"] = text, rc
            continue
        calls = parse_calls(f.get("call") or f.get("tool_calls") or "")
        reasoning = f.get("reason", "")
        if not calls:
            content = f.get("content", "")
            if content and content != "None":
                n_text_only += 1
                pairs.append(
                    {
                        "action": f"message: {content}",
                        "reasoning": reasoning,
                        "name": "message",
                        "result": "",
                        "returncode": None,
                    }
                )
            continue
        for k, c in enumerate(calls):
            pairs.append(
                {
                    "action": action_string(c),
                    "reasoning": reasoning if k == 0 else "",
                    "name": c["name"],
                    "result": "",
                    "returncode": None,
                }
            )
            pending.append(len(pairs) - 1)
    steps, prev = [], ""
    for i, p in enumerate(pairs):
        steps.append(
            {
                "step_idx": i,
                "observation": truncate(prev, obs_chars),
                "action": p["action"],
                "reasoning": p["reasoning"],
                "tool_name": p["name"],
                "returncode": p["returncode"],
            }
        )
        prev = p["result"]
    meta = {
        "scenario_prompt": "\n\n".join(framing),
        "n_bash_calls": sum(1 for p in pairs if p["name"] == "bash"),
        "n_text_only_turns": n_text_only,
        "unanswered_calls": len(pending),
        "transcript_chars": len(transcript),
        "submitted": SUBMIT_MARKER in transcript,
    }
    return steps, meta


def steps_to_actions_text(steps: list[dict]) -> str:
    """Exactly Trajectory.to_text('actions_only'): the constraint head's input."""
    return " ".join(f"[ACTION] {s['action']}" for s in steps)


def _call_of(action: str):
    if action.startswith("bash: "):
        return {"name": "bash", "arguments": {"command": action[len("bash: ") :]}}
    if action.startswith("task_complete: "):
        return {
            "name": "task_complete",
            "arguments": {"reason": action[len("task_complete: ") :]},
        }
    return None


def messages_of(scenario_prompt: str, steps: list[dict]) -> list[dict]:
    """Chat-template messages with real tool calls / tool-role responses / reasoning_content."""
    system = scenario_prompt.split("[user]", 1)[0].replace("[system]", "").strip()
    user = (
        scenario_prompt.split("[user]", 1)[1].strip()
        if "[user]" in scenario_prompt
        else ""
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for i, s in enumerate(steps):
        c = _call_of(s["action"])
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
        if i + 1 < len(steps):
            msgs.append(
                {"role": "tool", "content": steps[i + 1].get("observation") or ""}
            )
    return msgs


def render_prompt_completion(tok, msgs: list[dict]) -> tuple[str, str]:
    full = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False)
    prefix = tok.apply_chat_template(
        msgs[:2], tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    if not full.startswith(prefix):
        raise ValueError(
            "chat template prefix does not match; check the template's generation prompt"
        )
    return prefix, full[len(prefix) :]
