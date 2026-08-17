# ABOUTME: Dumps the legacy SuiteCRM wizard page's buttons/inputs/links to find its real controls.
# ABOUTME: Run on the login node: PYTHONPATH=. python scratch/dump_wizard_controls.py
import os
import time

BASE = os.environ["WA_SUITECRM"].rstrip("/")
from playwright.sync_api import sync_playwright

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = browser.new_page()
    page.goto(f"{BASE}/legacy/index.php?action=Login&module=Users")
    page.wait_for_selector("input[name='user_name']", timeout=30000)
    page.fill("input[name='user_name']", "user")
    page.fill("input[name='username_password']", "bitnami")
    page.click("input[type='submit'], #bigbutton")
    page.wait_for_load_state("networkidle", timeout=30000)
    page.goto(f"{BASE}/legacy/index.php?module=Users&action=Wizard")
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    time.sleep(2)
    print("url:", page.url)
    print("title:", page.title())
    for kind in ("button", "input[type='button']", "input[type='submit']", "a.button", "[id*=next]", "[id*=finish]", "[id*=skip]", "[id*=close]"):
        els = page.locator(kind)
        n = els.count()
        for i in range(min(n, 8)):
            e = els.nth(i)
            try:
                print(f"{kind}[{i}]: id={e.get_attribute('id')!r} value={e.get_attribute('value')!r} text={e.inner_text()[:40]!r} visible={e.is_visible()}")
            except Exception as ex:
                print(f"{kind}[{i}]: <{type(ex).__name__}>")
    body = page.inner_text("body")[:500].replace("\n", " | ")
    print("body:", body)
    browser.close()
