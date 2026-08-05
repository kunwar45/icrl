# ABOUTME: The planner-model stages of trajectory generation: propose an optimal plan,
# ABOUTME: refine it against the policies, and revise it from a concrete failure report.
"""
Plans are plain numbered text, not action calls — element ids only exist at
runtime, so the plan stays at the level of intent ("open the record's Actions
menu") and the executor grounds each step against the live page.

All prompt text lives in the run config (prompts.propose_plan / refine_plan /
revise_plan / diversify_plan), so what "optimal" means is part of the run
definition.

Diversity: synthetic plans risk converging on one template, and a constraint
head trained on templated experts can learn "plan-shaped = safe" instead of
"compliant = safe". The refine stage therefore sees recent plans from other
tasks ({previous_plans}), and plan_similarity() gives the runner a cheap
quantitative check to trigger an explicit diversification pass.
"""
from __future__ import annotations

import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def plan_similarity(plan: str, previous_plans: list[str]) -> float:
    """
    Max character-level similarity (0..1) between `plan` and any previous plan.

    SequenceMatcher on normalized text is deliberately crude: it catches the
    failure mode that matters (near-verbatim template reuse across tasks)
    without an embedding model in the loop. Structural echoes that any two
    compliant CRM plans share (a confirm step before a delete) score well
    below the trigger threshold.
    """
    def _norm(s: str) -> str:
        return " ".join(s.lower().split())

    plan_n = _norm(plan)
    return max((SequenceMatcher(None, plan_n, _norm(p)).ratio()
                for p in previous_plans), default=0.0)


def format_previous_plans(previous_plans: list[str]) -> str:
    """Render recent plans for the {previous_plans} template field."""
    if not previous_plans:
        return "(none yet)"
    blocks = []
    for i, p in enumerate(previous_plans, 1):
        blocks.append(f"<previous_plan {i}>\n{p}\n</previous_plan {i}>")
    return "\n\n".join(blocks)


def _render(template: str, fields: dict[str, str]) -> str:
    class _Default(dict):
        def __missing__(self, key):  # noqa: ANN001
            return "{" + key + "}"
    return template.format_map(_Default(fields))


def _call_planner(client, model_cfg: dict, prompt_cfg: dict, fields: dict) -> str:
    response = client.chat.completions.create(
        model=model_cfg["name"],
        messages=[
            {"role": "system", "content": _render(prompt_cfg["system"], fields)},
            {"role": "user", "content": _render(prompt_cfg["user"], fields)},
        ],
        max_tokens=int(model_cfg.get("max_tokens", 2048)),
        temperature=float(model_cfg.get("temperature", 0.7)),
    )
    return (response.choices[0].message.content or "").strip()


def propose_plan(client, model_cfg: dict, prompts: dict, metadata: dict) -> str:
    """Stage 1: sketch what an optimal, policy-compliant trajectory looks like."""
    return _call_planner(client, model_cfg, prompts["propose_plan"], metadata)


def refine_plan(client, model_cfg: dict, prompts: dict, metadata: dict,
                plan: str, previous_plans: list[str]) -> str:
    """Stage 2: critic pass — tighten the plan against the policies and
    optimality, with recent plans in context so it does not converge on one
    template."""
    return _call_planner(client, model_cfg, prompts["refine_plan"],
                         dict(metadata, plan=plan,
                              previous_plans=format_previous_plans(previous_plans)))


def diversify_plan(client, model_cfg: dict, prompts: dict, metadata: dict,
                   plan: str, previous_plans: list[str]) -> str:
    """Extra pass when the refined plan is still too close to a previous one."""
    return _call_planner(client, model_cfg, prompts["diversify_plan"],
                         dict(metadata, plan=plan,
                              previous_plans=format_previous_plans(previous_plans)))


def revise_plan(client, model_cfg: dict, prompts: dict, metadata: dict,
                plan: str, failure_report: str) -> str:
    """Stage 5: the plan met reality and lost — revise it from the failure report."""
    return _call_planner(client, model_cfg, prompts["revise_plan"],
                         dict(metadata, plan=plan, failure_report=failure_report))


def build_failure_report(result: dict) -> str:
    """Compress an executed episode's failure into what the planner needs to see."""
    lines = [
        f"reward: {result['reward']} (needs 1.0)",
        f"terminated cleanly: {result['terminated']}"
        + ("" if result["terminated"] else " — ran out of steps or was cut off"),
        f"steps used: {result['n_steps']}",
    ]
    if result["violated_policies"]:
        lines.append("policy violations:")
        for v in result["violated_policies"]:
            lines.append(f"  - [{v.get('policy_category', '?')}] "
                         f"{v.get('description', '')}: {v.get('reason', '')}")
    else:
        lines.append("policy violations: none")
    tail = [s["action"] for s in result["steps"][-8:]]
    if tail:
        lines.append("last actions executed:")
        lines.extend(f"  {a}" for a in tail)
    if result.get("error"):
        lines.append(f"episode error: {result['error']}")
    return "\n".join(lines)
