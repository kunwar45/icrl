# ABOUTME: Trajectory-generation pipeline: synthesize an optimal plan, execute it with the
# ABOUTME: policy model in the real env, verify CuP by ground truth. Run via scripts/generate_trajectories.py.
"""
Synthetic expert-trajectory generation — the alternative to plain collection.

Staged like a synthetic-SFT pipeline, reworked for embodied trajectories:

  1. propose_plan   strong planner model sketches what an optimal, policy-
                    compliant trajectory looks like (the idea, not actions)
  2. refine_plan    critic pass over the plan against the policies
  3. execute        the policy model under test rolls the plan out in the REAL
                    benchmark env (plan-guided episode)
  4. verify         the benchmark's own evaluators judge the executed episode —
                    kept only on reward=1.0, zero violations, clean termination
  5. revise_plan    on failure, the planner sees the concrete failure report
                    and revises; loop back to 3

Verification is execution: a trajectory is never trusted because a model wrote
it, only because the ground-truth evaluators scored the real rollout. What gets
saved is the executed episode (real observations), with the plan attached as
provenance.

Benchmark-agnostic via the same BenchmarkAdapter seam as src/trajectory_collection.
"""
from src.trajectory_generation.generation_runner import run_generation

__all__ = ["run_generation"]
