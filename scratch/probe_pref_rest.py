# ABOUTME: Probes REST write paths for the SuiteCRM user-preferences resource (wizard escape).
# ABOUTME: Run on the login node: PYTHONPATH=. python scratch/probe_pref_rest.py
import json
import os

os.environ.setdefault("SUITECRM", os.environ["WA_SUITECRM"])

from src.trajectory_collection.stwebagentbench_adapter import STWebAgentBenchAdapter

FETCH = """async ({url, method, body, ctype}) => {
  const xsrf = document.cookie.split("; ").find(c => c.startsWith("XSRF-TOKEN="));
  const token = xsrf ? decodeURIComponent(xsrf.split("=")[1]) : "";
  const resp = await fetch(url, {method, credentials: "same-origin",
    headers: {"Content-Type": ctype, "X-XSRF-TOKEN": token, "Accept": "application/ld+json"},
    body: body ? JSON.stringify(body) : undefined});
  const text = await resp.text();
  return resp.status + " :: " + text.slice(0, 300);
}"""

adapter = STWebAgentBenchAdapter({"name": "stwebagentbench"})
env = adapter.make_env(235)
env.reset()
page = env.unwrapped.page

attempts = [
    ("PUT ld+json", "/api/user-preferences/global", "PUT",
     {"items": {"ut": True}}, "application/ld+json"),
    ("PATCH merge-patch", "/api/user-preferences/global", "PATCH",
     {"items": {"ut": True}}, "application/merge-patch+json"),
]
for label, url, method, body, ctype in attempts:
    out = page.evaluate(FETCH, {"url": url, "method": method, "body": body, "ctype": ctype})
    print(f"[{label}] {out}")

out = page.evaluate(FETCH, {"url": "/api/graphql", "method": "POST",
                            "body": {"query": "{ userPreferences { edges { node { _id items } } } }"},
                            "ctype": "application/json"})
print("ut now:", '"ut":true' in out)
env.close()
