# ABOUTME: Logs into SuiteCRM through the real login form and saves Playwright storage state to .auth/
# ABOUTME: Run on the login node: python scratch/generate_suitecrm_auth_state.py <base_url>
"""Log into SuiteCRM via the real UI and save Playwright storage state.

ST-WebAgentBench tasks declare `storage_state: ./.auth/suitecrm_state.json`
(cwd-relative), using the benchmark's standard demo account from
stwebagentbench/browser_env/env_config.py. This script performs the login the
same way the benchmark's agent would (through the login form) and saves the
authenticated state, so episodes start logged in.

Usage:  python generate_suitecrm_auth_state.py <base_url> <out_json>
"""
import sys
import time

from playwright.sync_api import sync_playwright

BASE, OUT = sys.argv[1], sys.argv[2]
# The benchmark's fixed demo credentials (ACCOUNTS["suitecrm"] in env_config.py).
USERNAME, PASSWORD = "user", "bitnami"

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context()
    page = ctx.new_page()
    page.goto(BASE, wait_until="networkidle", timeout=60_000)
    try:
        page.wait_for_selector("input[type='password']", timeout=45_000)
    except Exception:
        pass
    time.sleep(2)

    # SuiteCRM 8 Angular login form (formcontrolname) + fallbacks
    for sel_user, sel_pass, sel_btn in [
        ("input[formcontrolname='username']", "input[formcontrolname='password']",
         "button[type='submit']"),
        ("input[name='username']", "input[name='password']", "button[type='submit']"),
        ("input#username", "input#password", "button#login-button"),
        ("input[name='user_name']", "input[name='username_password']", "input#bigbutton"),
        ("input[type='text']", "input[type='password']", "button[type='submit']"),
    ]:
        if page.locator(sel_user).count():
            page.fill(sel_user, USERNAME)
            page.fill(sel_pass, PASSWORD)
            if page.locator(sel_btn).count():
                page.click(sel_btn)
            else:
                for alt in ["button:has-text('Login')", "button:has-text('LOG IN')",
                            "scrm-button button", "button.login-button"]:
                    if page.locator(alt).count():
                        page.click(alt)
                        break
                else:
                    page.locator(sel_pass).press("Enter")
            break
    else:
        print(f"LOGIN-FORM-NOT-FOUND url={page.url}")
        print(page.content()[:500])
        sys.exit(2)

    page.wait_for_load_state("networkidle", timeout=60_000)
    time.sleep(3)
    url = page.url
    content = page.content()
    logged_in = ("Login" not in page.title() and "auth" not in url) or "logout" in content.lower()
    print(f"post-login url: {url}")
    print(f"title: {page.title()}")
    if not logged_in:
        print("LOGIN-FAILED")
        sys.exit(1)

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ctx.storage_state(path=OUT)
    print(f"LOGIN-OK — storage state saved to {OUT}")
    browser.close()
