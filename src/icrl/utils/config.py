"""Lightweight YAML → typed-dataclass config loader.

Usage:
    cfg = load_config("configs/experiment/gridworld_smoke.yaml")
    # Then construct objects manually using cfg["trainer"], cfg["policy"], etc.

Returns a plain dict so run scripts stay explicit about what they build.
"""
from __future__ import annotations

import yaml


def load_config(yaml_path: str, overrides: dict | None = None) -> dict:
    """Load YAML and merge any programmatic overrides (nested dict)."""
    with open(yaml_path) as f:
        raw: dict = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return raw


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
