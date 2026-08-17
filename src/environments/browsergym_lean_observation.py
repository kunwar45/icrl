# ABOUTME: Trims BrowserGym's per-step observation to the fields an agent prompt actually reads.
# ABOUTME: Not run directly — installed on an env by stwebagentbench_adapter when episode.lean_observation is on.
"""
Why this exists: building one observation runs several extractions, and this
project reads almost none of them.

    _pre_extract                  mark DOM elements with bids        REQUIRED
    extract_merged_axtree         CDP Accessibility.getFullAXTree    REQUIRED
    extract_dom_snapshot          CDP DOMSnapshot.captureSnapshot    unused
    extract_dom_extra_properties  pure-Python walk over the WHOLE
                                  snapshot, per node                 unused
    extract_screenshot            full-viewport PNG through CDP      unused
    extract_focused_element_bid                                      unused
    read_webpage_content          (fork only) wait_for_load_state
                                  ('networkidle') + innerText        unused
    _post_extract                 cleanup                            REQUIRED

Our prompts consume `axtree_object`, `goal`, `url`, `chat_messages` and
`policies` (stwebagentbench_adapter.prompt_fields), and flatten the axtree with
`flatten_axtree_to_str(axtree)` — no `extra_properties` argument, so every
`with_visible` / `with_clickable` / `filter_visible_only` flag stays False.
Nothing in the benchmark fork or in `src/` reads `dom_object`,
`extra_element_properties` or `read_page` either (checked 2026-08-15). On a
SuiteCRM list view those unused steps dominate: the snapshot is a large CDP
round-trip, the properties walk traverses all of it in Python, and
`read_webpage_content` blocks on network idle every single step.

HOW, and why not the obvious way: the first version of this module replaced
`_get_obs` wholesale with a copy of BrowserGym's, minus the unused calls. That
broke immediately on the cluster, because ST-WebAgentBench does not use
BrowserGym's env — `stwebagentbench/browser_env/custom_env.py` defines its own
`_get_obs` with a DIFFERENT observation dict (no `goal_object`; it derives
`goal` from the chat, and adds `policies` and `read_page`). Re-deriving someone
else's observation shape is a standing invitation to drift.

So this patches the *extraction functions* in the namespace where the env's own
`_get_obs` looks them up, and leaves `_get_obs` itself alone. Whatever keys that
env builds, it still builds — they just carry cheap empty values for the parts
nobody reads. Works for BrowserGym's env and the fork's alike, and survives
either of them changing its observation dict.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

#: Set on a module whose extraction functions have been replaced.
_INSTALLED_FLAG = "_icrl_lean_observation_installed"

#: Extraction functions replaced with cheap stubs, by name in the env module's
#: namespace. `_pre_extract`, `extract_merged_axtree` and `_post_extract` are
#: deliberately absent: bids come from the marking pass and the action set
#: addresses elements by bid, so skipping either breaks grounding outright.
_STUBBED_EXTRACTORS = (
    "extract_screenshot",
    "extract_dom_snapshot",
    "extract_dom_extra_properties",
    "extract_focused_element_bid",
)


def _stub_screenshot(*_args, **_kwargs):
    """A 1x1 black image: the observation space wants an array, nobody looks."""
    import numpy as np
    return np.zeros((1, 1, 3), dtype=np.uint8)


def _stub_dom_snapshot(*_args, **_kwargs) -> dict:
    # Shaped like a real snapshot so a consumer that indexes it gets an empty
    # result rather than a KeyError.
    return {"documents": [], "strings": []}


def _stub_dom_extra_properties(*_args, **_kwargs) -> dict:
    return {}


def _stub_focused_element_bid(*_args, **_kwargs) -> str:
    return ""


def _stub_read_webpage_content(*_args, **_kwargs) -> str:
    return ""


_STUBS = {
    "extract_screenshot": _stub_screenshot,
    "extract_dom_snapshot": _stub_dom_snapshot,
    "extract_dom_extra_properties": _stub_dom_extra_properties,
    "extract_focused_element_bid": _stub_focused_element_bid,
}


def _env_observation_module(env):
    """The module whose namespace the env's `_get_obs` resolves names in."""
    get_obs = getattr(type(env), "_get_obs", None)
    if get_obs is None:
        return None
    module_name = getattr(get_obs, "__module__", None)
    return sys.modules.get(module_name) if module_name else None


def install_lean_observation(env) -> bool:
    """
    Make `env`'s observation builder skip the extractions nothing reads.

    Returns True when the lean path is in place. Returns False (with a warning)
    when the env does not expose a patchable `_get_obs` — a speed optimisation
    must never be the reason a run dies, and the full path is merely slower.

    Idempotent per module: several envs from one class patch it once.
    """
    inner = getattr(env, "unwrapped", env)

    if getattr(inner, "use_raw_page_output", False):
        # That mode already skips every extraction and returns the live page;
        # patching would silently change what the caller receives.
        return False

    module = _env_observation_module(inner)
    if module is None:
        logger.warning("%s exposes no patchable _get_obs — leaving the full "
                       "observation path in place", type(inner).__name__)
        return False

    # The per-step page-text read is a method on the fork's env class, not a
    # module-level function, so it is patched on the class it is defined on.
    if hasattr(type(inner), "read_webpage_content") and \
            not getattr(type(inner), _INSTALLED_FLAG, False):
        type(inner).read_webpage_content = _stub_read_webpage_content
        setattr(type(inner), _INSTALLED_FLAG, True)
        logger.debug("stubbed read_webpage_content on %s", type(inner).__name__)

    if getattr(module, _INSTALLED_FLAG, False):
        return True

    stubbed = []
    for name in _STUBBED_EXTRACTORS:
        if hasattr(module, name):
            setattr(module, name, _STUBS[name])
            stubbed.append(name)

    if not stubbed:
        # Nothing to trim is not a failure, but it does mean the assumption
        # behind this module no longer holds for this env — say so.
        logger.warning("%s resolves none of the expected extraction functions "
                       "— observation left untouched", module.__name__)
        return False

    setattr(module, _INSTALLED_FLAG, True)
    logger.info("lean observation installed in %s (stubbed: %s)",
                module.__name__, ", ".join(stubbed))
    return True
