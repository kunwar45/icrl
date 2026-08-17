# ABOUTME: Prints the action names the agent may emit, to check select_option is available
# ABOUTME: Run on killarney: PYTHONPATH=. python list_actions.py
import re

from src.environments.stwebagentbench_environment import build_action_set

description = build_action_set(multiaction=False).describe(
    with_long_description=False, with_examples=False)
names = sorted(set(re.findall(r"^\s*([a-z_]+)\(", description, re.M)))
print("actions:", names)
print("has select_option:", "select_option" in names)
