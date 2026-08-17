# ABOUTME: Tests whether a fresh-load, row-scoped DOM probe can verify PERSISTED record state.
# ABOUTME: Run on the login node: PYTHONPATH=. python scratch/probe_row_scoped_eval.py
"""
Why: `url:"last"` evals read the agent's own page, which contains its unsaved
typing, so they pass even when nothing was saved (tasks 244 and 252 both
"passed" while the DB shows no change). A fresh page load at an explicit URL
returns only server state; scoping to the target record's ROW removes
cross-row contamination (e.g. another case that is legitimately "Closed").
"""
import os
import time

os.environ.setdefault("SUITECRM", os.environ["WA_SUITECRM"])
BASE = os.environ["WA_SUITECRM"].rstrip("/")
USERNAME, PASSWORD = "user", "bitnami"

from playwright.sync_api import sync_playwright

# The row-scoped locator shape a rewritten eval would use.
ROW_PROBE = """() => {
  const needle = %r;
  const rows = Array.from(document.querySelectorAll('tr'));
  const row = rows.find(r => (r.innerText || '').includes(needle));
  return row ? row.innerText.replace(/\\s+/g, ' ').trim() : '(no row found)';
}"""

HEADERS = """() => Array.from(document.querySelectorAll('th'))
    .map(h => (h.innerText || '').replace(/\\s+/g,' ').trim()).filter(Boolean).join(' | ')"""


def load_fresh(browser, route):
    page = browser.new_page()
    page.goto(f"{BASE}/")
    page.wait_for_selector("input", state="visible", timeout=60000)
    page.locator("input").first.fill(USERNAME)
    page.locator("input[type='password']").first.fill(PASSWORD)
    try:
        page.get_by_text("Log in").click()
    except Exception:
        page.locator("button[type='submit'], button:has-text('Log')").first.click()
    time.sleep(5)
    page.goto(f"{BASE}/{route}")
    for _ in range(20):
        if page.locator("tr").count() > 1:
            break
        time.sleep(1.5)
    time.sleep(2)
    return page


with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True,
                                 args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
    for route, needle, want in (
        ("#/contacts/index", "Schrute", "Stark Industries (task 252 association)"),
        ("#/cases/index", "Email Integration Failure", "Closed (task 244 status)"),
    ):
        page = load_fresh(browser, route)
        print(f"\n=== {route}  (looking for: {want})")
        print("columns:", page.evaluate(HEADERS)[:220])
        row = page.evaluate(ROW_PROBE % needle)
        print("row    :", row[:220])
        page.close()
    browser.close()
