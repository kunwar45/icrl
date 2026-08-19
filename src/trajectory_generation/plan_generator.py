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


def build_failure_report(result: dict, state_advice: str | None = None) -> str:
    """
    Compress an executed episode's failure into what the planner needs to see.

    `state_advice` is what to say when the ground-truth state check failed. It is
    benchmark-specific and MUST be supplied by the adapter for anything that is
    not a web app: the SuiteCRM wording below told AI2-THOR household-robot
    planners to "SAVE and confirm the save landed (the detail view shows the
    stored values)" 1,441 times in job 4888310. There are no records, fields or
    detail views in a kitchen, and steering a planner with another benchmark's
    vocabulary is a plausible source of the prose-verb drift that made 63% of
    generated plan lines unparseable.
    """
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
    # The task really was done — the plan just did not stop. Say so plainly,
    # otherwise the planner reworks steps that already worked.
    if result.get("state_verified") is True and not result.get("finished_deliberately"):
        lines.append("database check — THE TASK WAS COMPLETED SUCCESSFULLY, but the "
                     "episode kept acting until it ran out of steps. Keep the "
                     "working steps and end the plan with answer() immediately "
                     "after the goal is met; do not re-navigate afterwards.")
    # The most actionable failure the planner can get: the page looked right but
    # nothing was saved, so the plan is missing a Save/confirm step.
    if result.get("state_verified") is False:
        lines.append("ground-truth state check FAILED:")
        lines.extend(f"  {line}" for line in result.get("state_detail", "").splitlines())
        lines.append("  " + (state_advice or
                     "Fix the plan so the record is actually committed: after "
                     "filling fields, SAVE and confirm the save landed (the "
                     "detail view shows the stored values, not your typing)."))
    tail = [s["action"] for s in result["steps"][-8:]]
    if tail:
        lines.append("last actions executed:")
        lines.extend(f"  {a}" for a in tail)
    if result.get("error"):
        lines.append(f"episode error: {result['error']}")
    return "\n".join(lines)
