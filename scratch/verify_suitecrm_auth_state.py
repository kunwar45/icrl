# ABOUTME: Checks that a saved SuiteCRM Playwright storage state still grants an authenticated session
# ABOUTME: Run: python scratch/verify_suitecrm_auth_state.py <base_url> <state_json>
"""Verify a saved SuiteCRM storage state actually grants an authenticated session.

Opens #/contacts with the state; success = the contacts list renders and we are
not bounced to the #/Login route.

Usage:  python verify_suitecrm_auth_state.py <base_url> <state_json>
"""
import sys
import time

from playwright.sync_api import sync_playwright

BASE, STATE = sys.argv[1], sys.argv[2]

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(storage_state=STATE)
    page = ctx.new_page()
    page.goto(f"{BASE}/#/contacts", wait_until="networkidle", timeout=60_000)
    time.sleep(5)
    url = page.url
    body = page.content()
    on_login = "Login" in url or "input[type=password]" if False else ("#/Login" in url)
    has_form = page.locator("input[type='password']").count() > 0
    print(f"url: {url}")
    print(f"password-field-visible: {has_form}")
    for marker in ("Michael", "Contacts", "CONTACTS"):
        if marker in body:
            print(f"content-marker: {marker} FOUND")
            break
    else:
        print("content-marker: none found")
    if on_login or has_form:
        print("AUTH-STATE-INVALID")
        sys.exit(1)
    print("AUTH-STATE-VALID")
    browser.close()
