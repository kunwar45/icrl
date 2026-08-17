# ABOUTME: Confirms the env honours end_on_score, so a text-presence evaluator cannot cut an episode short.
# ABOUTME: Run on the login node: PYTHONPATH=. python scratch/verify_end_on_score.py
import inspect
import os

os.environ.setdefault("SUITECRM", os.environ["WA_SUITECRM"])

from browsergym.stwebagentbench.task import GenericWebArenaTask
from src.trajectory_collection.stwebagentbench_adapter import STWebAgentBenchAdapter

params = inspect.signature(GenericWebArenaTask.__init__).parameters
print("task accepts end_on_score:", "end_on_score" in params)
print("default (evaluation fidelity):", params["end_on_score"].default)

adapter = STWebAgentBenchAdapter({"name": "stwebagentbench"})
# Task 244: a *different* seeded case already reads "Closed", so under the
# benchmark default its evaluator scores 1.0 on the first navigation and ends
# the episode before the agent can save anything.
env = adapter.make_env(244, max_steps=30, end_on_score=False)
env.reset()
task = env.unwrapped.task
print("generation env -> max_steps:", task.max_steps,
      "| end_on_score:", task.end_on_score)
env.close()

env_default = adapter.make_env(244)
env_default.reset()
print("default env    -> max_steps:", env_default.unwrapped.task.max_steps,
      "| end_on_score:", env_default.unwrapped.task.end_on_score)
env_default.close()
