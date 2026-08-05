"""Terminal visualization for the adversarial ICRL GridWorld training loop.

Prints a self-contained report each eval interval showing:
  - Grid with constrained cells marked
  - Constraint feasibility probe (canonical safe vs unsafe path)
  - Training metrics table
  - Recent complete episodes with path strings
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from icrl.core.interfaces import BaseConstraint
from icrl.core.types import Trajectory, Transition
from icrl.envs.gridworld import CONSTRAINED_CELLS, IDX_TO_ACTION, GridWorldEnv

# ANSI colour helpers — degrade gracefully in non-TTY
_G = "\033[92m"   # green
_R = "\033[91m"   # red
_Y = "\033[93m"   # yellow
_C = "\033[96m"   # cyan
_B = "\033[1m"    # bold
_D = "\033[2m"    # dim
_X = "\033[0m"    # reset
_W = "\033[97m"   # white

_BAR = "─" * 62


def _col(text: str, code: str) -> str:
    import sys
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_X}"


# ── Grid rendering ─────────────────────────────────────────────────────────────

def _render_grid(
    safe_path: Optional[list[tuple[int, int]]] = None,
    unsafe_path: Optional[list[tuple[int, int]]] = None,
    size: int = 4,
) -> list[str]:
    grid = [["." for _ in range(size)] for _ in range(size)]

    for (r, c) in CONSTRAINED_CELLS:
        grid[r][c] = _col("!", _R)

    # Mark paths (interior cells only — not start/end)
    if safe_path:
        for (r, c) in safe_path[1:-1]:
            if grid[r][c] == ".":
                grid[r][c] = _col("*", _G)
    if unsafe_path:
        for (r, c) in unsafe_path[1:-1]:
            cell = grid[r][c]
            if cell == "." or cell == _col("!", _R):
                grid[r][c] = _col("#", _R)

    grid[0][0] = _col("O", _B)
    grid[3][0] = _col("E", _B)

    lines = ["  " + " ".join(_col(str(c), _D) for c in range(size))]
    for r in range(size):
        lines.append(_col(str(r), _D) + " " + " ".join(grid[r]))
    return lines


# ── Path extraction ────────────────────────────────────────────────────────────

def _traj_to_cell_path(traj: Trajectory) -> list[tuple[int, int]]:
    """Returns list of (row, col) visited, starting from first obs."""
    path = []
    for t in traj.transitions:
        obs = t.obs
        if hasattr(obs, "__len__") and len(obs) >= 2:
            path.append((int(obs[0]), int(obs[1])))
    if traj.transitions:
        last = traj.transitions[-1].next_obs
        if hasattr(last, "__len__") and len(last) >= 2:
            path.append((int(last[0]), int(last[1])))
    return path


def _format_path(traj: Trajectory) -> str:
    """Short human-readable path: O→D→(1,0)→D→(2,0)→D→E"""
    parts = ["O"]
    for i, t in enumerate(traj.transitions):
        action_name = IDX_TO_ACTION.get(int(t.action), str(t.action))[0].upper()
        r, c = int(t.next_obs[0]), int(t.next_obs[1])
        is_last = i == len(traj.transitions) - 1
        parts.append(action_name)
        parts.append("E" if is_last and t.done else f"({r},{c})")
    return "→".join(parts)


# ── Constraint probe ───────────────────────────────────────────────────────────

def _probe_constraint(
    constraint: BaseConstraint,
    safe_traj: Trajectory,
    unsafe_traj: Trajectory,
) -> tuple[float, float]:
    """Returns (safe_feasibility, unsafe_feasibility) with no_grad."""
    with torch.no_grad():
        sf = float(constraint.feasibility(safe_traj).item())
        uf = float(constraint.feasibility(unsafe_traj).item())
    return sf, uf


def _make_canonical_trajectories(env: GridWorldEnv) -> tuple[Trajectory, Trajectory]:
    """Build one canonical safe and one canonical unsafe trajectory."""
    from icrl.envs.gridworld import ACTION_TO_IDX
    from icrl.policies.rule_based import SafeRuleBasedPolicy, UnsafeRuleBasedPolicy

    def _run(policy_cls):
        policy = policy_cls()
        obs, _ = env.reset()
        transitions = []
        for _ in range(env.cfg.max_steps):
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            transitions.append(Transition(
                obs=obs.copy(), action=action, reward=reward,
                next_obs=next_obs.copy(), done=(terminated or truncated), info=info,
            ))
            obs = next_obs
            if terminated or truncated:
                break
        return Trajectory.from_transitions(transitions)

    safe_traj = _run(SafeRuleBasedPolicy)
    unsafe_traj = _run(UnsafeRuleBasedPolicy)
    return safe_traj, unsafe_traj


# ── Main report ────────────────────────────────────────────────────────────────

def print_training_report(
    iteration: int,
    total_iters: int,
    in_pretrain: bool,
    metrics: dict,
    recent_trajs: list[Trajectory],
    threshold: float,
    constraint: Optional[BaseConstraint] = None,
    canonical_safe: Optional[Trajectory] = None,
    canonical_unsafe: Optional[Trajectory] = None,
) -> None:
    phase = _col("PRETRAIN (unconstrained)", _Y) if in_pretrain else _col("ICRL active", _G)
    header = f"  {_col('GridWorld ICRL', _B)}  │  iter {iteration}/{total_iters}  │  {phase}"

    print("\n" + _col("━" * 62, _D))
    print(header)
    print(_col("━" * 62, _D))

    # Grid + constraint probe side by side
    safe_cells = _traj_to_cell_path(canonical_safe) if canonical_safe else None
    unsafe_cells = _traj_to_cell_path(canonical_unsafe) if canonical_unsafe else None
    grid_lines = _render_grid(safe_cells, unsafe_cells)

    probe_lines: list[str] = []
    if constraint is not None and canonical_safe and canonical_unsafe and not in_pretrain:
        sf, uf = _probe_constraint(constraint, canonical_safe, canonical_unsafe)
        gap = sf - uf
        safe_ok  = sf > 0.5
        unsafe_ok = uf < 0.5
        probe_lines = [
            _col("  Constraint probe", _B),
            f"  safe path  (via col 1)  feas={sf:.3f}  "
            + (_col("✓ SAFE",   _G) if safe_ok  else _col("✗ not yet", _Y)),
            f"  unsafe path (via col 0) feas={uf:.3f}  "
            + (_col("✓ UNSAFE", _G) if unsafe_ok else _col("✗ not yet", _Y)),
            f"  gap = {gap:+.3f}  "
            + (_col("(separation OK)", _G) if gap > 0.3 else _col("(learning...)", _Y)),
            "",
            _col("  Key", _D) + _col("  O", _B) + "=origin  " + _col("E", _B) + "=end  "
            + _col("!", _R) + "=forbidden  " + _col("*", _G) + "=safe  " + _col("#", _R) + "=unsafe",
        ]
    elif in_pretrain:
        probe_lines = [
            _col("  Constraint probe", _B),
            _col("  (not yet active — still in pretrain)", _D),
        ]
    else:
        probe_lines = [_col("  Constraint probe: no constraint set", _D)]

    # Merge grid + probe columns
    col_width = 14
    n_rows = max(len(grid_lines), len(probe_lines))
    print()
    for i in range(n_rows):
        gl = grid_lines[i] if i < len(grid_lines) else ""
        pl = probe_lines[i] if i < len(probe_lines) else ""
        # strip ANSI for padding calculation
        import re
        gl_plain = re.sub(r'\033\[[0-9;]*m', '', gl)
        pad = max(0, col_width - len(gl_plain))
        print(f"  {gl}{'  ' * pad}{pl}")

    # Metrics
    print()
    print(_col(f"  {_BAR}", _D))
    m = metrics
    r = m.get("mean_reward", 0.0)
    r_col = _col(f"{r:7.2f}", _G if r >= threshold - 0.1 else _Y)
    print(f"  mean_reward   {r_col}  {_col(f'(safe threshold {threshold:.1f})', _D)}")
    print(f"  unsafe_buffer {m.get('unsafe_buffer_size', 0):7d}")
    c_loss = m.get("constraint_loss", None)
    c_str  = f"{c_loss:.4f}" if c_loss is not None else _col("n/a", _D)
    print(f"  c_loss        {c_str:>7}")
    gap_val = m.get("feasibility_gap", None)
    gap_str = f"{gap_val:.4f}" if gap_val is not None else _col("n/a", _D)
    print(f"  feas_gap      {gap_str:>7}")
    lam = m.get("lambda", None)
    lam_str = f"{lam:.4f}" if lam is not None else _col("n/a", _D)
    print(f"  λ (Lagrange)  {lam_str:>7}")

    # Recent episodes
    complete = [t for t in recent_trajs if t.transitions and t.transitions[-1].done]
    if complete:
        print()
        print(_col(f"  {_BAR}", _D))
        print(_col("  Recent episodes", _B))
        for traj in complete[-5:]:
            r_val = traj.total_reward
            flagged = r_val > threshold + 0.01
            tag = _col(f"UNSAFE r={r_val:.1f} FLAGGED", _R) if flagged else _col(f"SAFE   r={r_val:.1f}", _G)
            path_str = _format_path(traj)
            print(f"  [{tag}]  {path_str}")

    print(_col("━" * 62, _D))
