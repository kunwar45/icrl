#!/usr/bin/env python3
"""
Local end-to-end verification of the demo collection pipeline.

Mocks SuiteCRM (simple HTTP server with login inputs) and vLLM (returns
noop() actions) to run the real Python pipeline code without GPUs or a
cluster. Tests every failure mode we have encountered.

Usage:
    cd /path/to/icrl
    python scripts/verify_pipeline.py

All tests must pass before submitting to SLURM.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/Users/kunwar/projects/ST-WebAgentBench/stwebagentbench")
sys.path.insert(0, "/Users/kunwar/projects/ST-WebAgentBench/browsergym/stwebagentbench/src")

PASS = "  ✓"
FAIL = "  ✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"{PASS if ok else FAIL} {name}" + (f": {detail}" if detail else ""))
    return ok


# ── Mock SuiteCRM HTTP server ─────────────────────────────────────────────────

# HTML that simulates what SuiteCRM Angular renders (inputs present immediately)
_LOGIN_HTML = b"""<!DOCTYPE html>
<html><head><title>SuiteCRM</title></head>
<body>
  <form id="login-form">
    <input type="text" name="user_name" id="user_name" placeholder="User Name" />
    <input type="password" name="user_password" id="user_password" placeholder="Password" />
    <input type="submit" name="button" value="Log In" />
  </form>
</body></html>"""

_DASHBOARD_HTML = b"""<!DOCTYPE html>
<html><head><title>SuiteCRM Dashboard</title></head>
<body><h1>Dashboard</h1><p>Logged in.</p></body></html>"""


class _CRMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        if self.path in ("/public", "/legacy/public", "/"):
            self.wfile.write(_LOGIN_HTML)
        else:
            self.wfile.write(_DASHBOARD_HTML)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_len)
        self.send_response(302)
        self.send_header("Location", "/dashboard")
        self.end_headers()

    def log_message(self, *args):
        pass


def _start_mock_crm(port: int = 18080) -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", port), _CRMHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


# ── Mock vLLM HTTP server ─────────────────────────────────────────────────────

_VLLM_ACTIONS = [
    'goto("http://127.0.0.1:18080/legacy/index.php?module=Accounts&action=index")',
    'noop()',
    'answer("Task complete.")',
]
_vllm_call_count = 0


class _VLLMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def do_POST(self):
        global _vllm_call_count
        content_len = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_len)
        action = _VLLM_ACTIONS[_vllm_call_count % len(_VLLM_ACTIONS)]
        _vllm_call_count += 1
        resp = json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [{"index": 0, "message": {"role": "assistant",
                         "content": f"```python\n{action}\n```"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args):
        pass


def _start_mock_vllm(port: int = 18081) -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", port), _VLLMHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_syntax():
    """Check syntax of all changed files."""
    import ast
    files = [
        "scripts/collect_safe_trajectories.py",
        "src/data/st_webagent.py",
        "/Users/kunwar/projects/ST-WebAgentBench/browsergym/stwebagentbench/src/browsergym/stwebagentbench/instance.py",
        "/Users/kunwar/projects/ST-WebAgentBench/stwebagentbench/browser_env/custom_env.py",
    ]
    for f in files:
        try:
            ast.parse(open(f).read())
            check(f"Syntax OK: {Path(f).name}", True)
        except SyntaxError as e:
            check(f"Syntax OK: {Path(f).name}", False, str(e))


def test_pw_extra_args_in_collect():
    """Verify collect_safe_trajectories.py passes pw_extra_args to gym.make."""
    src = open("scripts/collect_safe_trajectories.py").read()
    ok = "--no-sandbox" in src and "pw_extra_args" in src
    check("pw_extra_args in collect_safe_trajectories.py", ok)


def test_pw_extra_args_flows_to_browser(crm_url: str):
    """Verify pw_extra_args from gym.make reaches BrowserEnv.__init__."""
    import gymnasium
    import browsergym.stwebagentbench  # noqa

    os.environ["WA_SUITECRM"] = crm_url
    os.environ["SUITECRM"] = crm_url

    from stwebagentbench.browser_env.custom_env import BrowserEnv

    captured = []

    def mock_init(self, *a, **kw):
        captured.append(kw.get("pw_extra_args", []))
        raise RuntimeError("STOP_MOCK")

    with patch.object(BrowserEnv, "__init__", mock_init):
        try:
            gymnasium.make(
                "browsergym/STWebAgentBenchEnv.235",
                headless=True,
                pw_extra_args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
        except RuntimeError:
            pass
        except Exception as e:
            pass

    if captured:
        expected = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        ok = captured[0] == expected
        check("pw_extra_args flows to BrowserEnv.__init__", ok,
              f"got {captured[0]}" if not ok else "")
    else:
        check("pw_extra_args flows to BrowserEnv.__init__", False, "BrowserEnv.__init__ never called")


def test_playwright_with_slurm_flags(crm_url: str):
    """Launch Chromium with SLURM-safe flags and navigate to mock SuiteCRM."""
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                "", headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            page = ctx.new_page()
            page.goto(f"{crm_url}/public")
            page.wait_for_selector("input", state="visible", timeout=10000)
            check("Playwright: Chromium launches with SLURM flags", True)
            check("Playwright: wait_for_selector('input') finds inputs", True)

            page.locator("input").first.fill("user")
            page.locator("input[type='password']").first.fill("bitnami")
            check("Playwright: username + password fill works", True)

            try:
                page.get_by_text("Log in").click()
                check("Playwright: get_by_text('Log in') click works", True)
            except Exception:
                page.locator("input[type='submit']").first.click()
                check("Playwright: fallback submit click works", True)

            ctx.close()
    except Exception as e:
        check("Playwright: SLURM-flags launch", False, str(e))
        check("Playwright: wait_for_selector('input')", False, "browser didn't launch")
        check("Playwright: username + password fill", False, "browser didn't launch")
        check("Playwright: submit click", False, "browser didn't launch")


def test_suitecrm_instance_login(crm_url: str):
    """Run the actual instance.py ui_login code against mock SuiteCRM."""
    from playwright.sync_api import sync_playwright
    from browsergym.stwebagentbench.instance import WebArenaInstance

    os.environ["WA_SUITECRM"] = crm_url
    os.environ["SUITECRM"] = crm_url

    try:
        instance = WebArenaInstance()
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                "", headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            page = ctx.new_page()
            instance.ui_login("suitecrm", page)
            check("instance.py ui_login('suitecrm') succeeds on mock CRM", True)
            ctx.close()
    except Exception as e:
        check("instance.py ui_login('suitecrm') succeeds on mock CRM", False, str(e))


def test_action_parsing():
    """Test _extract_action handles all expected LLM output formats."""
    sys.path.insert(0, "scripts")
    from collect_safe_trajectories import _extract_action

    cases = [
        ('```python\nclick("123")\n```',       "click(\"123\")"),
        ('```\ngoto("http://x.com")\n```',      'goto("http://x.com")'),
        ('`fill("123", "hello")`',              'fill("123", "hello")'),
        ('I will click.\nclick("abc")',          'click("abc")'),
        ('noop()',                               'noop()'),
        ('send_msg_to_user("confirm?")',         'send_msg_to_user("confirm?")'),
        ('click("a", button="left")',            'click("a")'),  # kwargs stripped
        ('garbage text with no action',          None),
    ]

    all_ok = True
    for text, expected in cases:
        got = _extract_action(text)
        ok = got == expected
        if not ok:
            all_ok = False
            check(f"  action parsing: {text[:40]!r}", False, f"expected {expected!r}, got {got!r}")
    check("Action parsing: all formats handled correctly", all_ok)


def test_vllm_client(vllm_url: str):
    """Verify the vLLM OpenAI client can connect and get a response."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key="mock", base_url=vllm_url)
        resp = client.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=20,
        )
        content = resp.choices[0].message.content
        check("vLLM client: connects and gets response", bool(content), content[:50])
    except Exception as e:
        check("vLLM client: connects and gets response", False, str(e))


def test_trajectory_saving():
    """Verify save_trajectory writes a valid JSON file."""
    sys.path.insert(0, "scripts")
    from collect_safe_trajectories import save_trajectory

    result = {
        "task_id": 235, "model": "mock", "reward": 1.0, "cup": True,
        "n_steps": 2, "terminated": True, "violated_policies": [],
        "safety_report": [], "steps": [
            {"step_idx": 0, "observation": "axtree", "action": "noop()",
             "step_reward": 0.0, "url": "http://x.com"},
        ], "policies": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_trajectory(result, 235, 0, Path(tmpdir))
        ok = path.exists()
        if ok:
            data = json.loads(path.read_text())
            ok = ok and data["cup"] is True and data["task_id"] == 235
        check("Trajectory saving: JSON written correctly", ok)


def test_run_episode_no_crash(crm_url: str, vllm_url: str):
    """
    Run one real episode (task 235, 3 steps) against mock servers.
    Verifies the pipeline doesn't crash — episode may fail due to mock CRM
    not being real SuiteCRM, but the Python framework must not throw.
    """
    import gymnasium  # noqa
    import browsergym.stwebagentbench  # noqa

    os.environ["WA_SUITECRM"] = crm_url
    os.environ["SUITECRM"] = crm_url

    sys.path.insert(0, "scripts")
    from collect_safe_trajectories import run_episode, build_client
    from browsergym.core.action.highlevel import HighLevelActionSet

    try:
        client = build_client("vllm", "mock-model", vllm_url)

        def answer(message):
            """
            When the task is done, call this function with a summary.

            Examples:
                answer("I finished the task.")
                answer("I finished the task, the answer is 'value'")
            """
            pass

        action_set = HighLevelActionSet(
            custom_actions=[answer],
            subsets=["bid", "chat", "nav", "custom"],
            strict=False, multiaction=False, demo_mode="off",
        )

        result = run_episode(
            task_id=235,
            model="mock-model",
            client=client,
            action_set=action_set,
            max_steps=3,
            verbose=False,
        )
        # Episode may fail (mock CRM ≠ real SuiteCRM) but must not crash Python
        check("run_episode: completes without Python crash", True,
              f"reward={result['reward']:.1f} steps={result['n_steps']} "
              f"error={result.get('error', 'none')}")
    except Exception as e:
        check("run_episode: completes without Python crash", False, str(e))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== ICRL pipeline local verification ===\n")

    # Start mock servers
    crm_srv = _start_mock_crm(18080)
    vllm_srv = _start_mock_vllm(18081)
    time.sleep(0.3)  # let servers bind

    crm_url = "http://127.0.0.1:18080"
    vllm_url = "http://127.0.0.1:18081/v1"

    print("[ Code checks ]")
    test_syntax()
    test_pw_extra_args_in_collect()

    print("\n[ Playwright / browser ]")
    test_playwright_with_slurm_flags(crm_url)
    test_suitecrm_instance_login(crm_url)

    print("\n[ BrowserEnv kwargs ]")
    test_pw_extra_args_flows_to_browser(crm_url)

    print("\n[ LLM / action parsing ]")
    test_vllm_client(vllm_url)
    test_action_parsing()

    print("\n[ Trajectory saving ]")
    test_trajectory_saving()

    print("\n[ Full episode (mock CRM + mock vLLM) ]")
    test_run_episode_no_crash(crm_url, vllm_url)

    crm_srv.shutdown()
    vllm_srv.shutdown()

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("\nFailed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"  ✗ {name}: {detail}")
        sys.exit(1)
    else:
        print("All checks passed — safe to submit to SLURM.")
    print()


if __name__ == "__main__":
    main()
