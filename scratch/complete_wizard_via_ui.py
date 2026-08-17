# ABOUTME: Completes the SuiteCRM first-login wizard through the app's own UI (legacy login + wizard flow).
# ABOUTME: Run on the login node after any DB reseed: PYTHONPATH=. python scratch/complete_wizard_via_ui.py
"""
The bench user's first login routes every SPA session to #/users/Wizard. The
wizard view is a legacy iframe, and the legacy sub-app has no session yet, so
the page renders blank and traps every agent episode on an empty screen.

This script does what a human user would: logs into the legacy sub-app with
the benchmark's standard credentials, opens the wizard, and clicks through it
so SuiteCRM records it as completed. After that, neither agent episodes nor
evaluator page-loads see the wizard again (until a DB reseed — rerun then).
"""
import os
import sys
import time

BASE = os.environ["WA_SUITECRM"].rstrip("/")
USERNAME = "user"      # benchmark account, from ST-WebAgentBench env_config.py
PASSWORD = "bitnami"   # public default, same source

from playwright.sync_api import sync_playwright

PW_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]


def main() -> int:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=PW_ARGS)
        page = browser.new_page()

        # 1. Legacy login (renders even when the SPA wizard is pending).
        page.goto(f"{BASE}/legacy/index.php?action=Login&module=Users")
        page.wait_for_selector("input[name='user_name']", timeout=30000)
        page.fill("input[name='user_name']", USERNAME)
        page.fill("input[name='username_password']", PASSWORD)
        page.click("input[type='submit'], #bigbutton")
        page.wait_for_load_state("networkidle", timeout=30000)
        print("after legacy login:", page.url)

        # 2. Walk the wizard's tab flow: welcome → personal info → locale →
        #    finish tab → Finish submit (ids observed on this deployment).
        page.goto(f"{BASE}/legacy/index.php?module=Users&action=Wizard")
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        for sel in ("#next_tab_personalinfo", "#next_tab_locale",
                    "#next_tab_finish", "input[type='submit'][value='Finish']"):
            loc = page.locator(sel).first
            try:
                loc.wait_for(state="visible", timeout=15000)
                loc.click()
                print("clicked:", sel)
                time.sleep(1.5)
            except Exception as e:
                print(f"could not click {sel}: {type(e).__name__}")
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        print("after wizard flow:", page.url)

        # 3. Verify: a FRESH SPA boot must reach a list route (this is what
        #    both agent episodes and the evaluator's new pages do).
        p2 = browser.new_page()
        p2.goto(f"{BASE}/")
        p2.wait_for_selector("input", state="visible", timeout=60000)
        p2.locator("input").first.fill(USERNAME)
        p2.locator("input[type='password']").first.fill(PASSWORD)
        try:
            p2.get_by_text("Log in").click()
        except Exception:
            p2.locator("button[type='submit'], button:has-text('Log')").first.click()
        time.sleep(5)
        p2.goto(f"{BASE}/#/contacts/index")
        deadline = time.time() + 30
        while time.time() < deadline:
            if "Halpert" in p2.content():
                print(f"fresh session renders contacts list (url: {p2.url})")
                print("WIZARD_COMPLETED")
                browser.close()
                return 0
            time.sleep(1.5)
        print(f"still trapped (url: {p2.url})")
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
