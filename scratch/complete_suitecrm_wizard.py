# ABOUTME: Marks the SuiteCRM first-login wizard as completed (ut=true) for the bench user via GraphQL.
# ABOUTME: Run on the login node after any DB reseed: PYTHONPATH=. python scratch/complete_suitecrm_wizard.py
"""
The bench user is seeded without user_preferences, so SuiteCRM 8 routes every
session to #/users/Wizard — a legacy-iframe view that renders blank in this
deployment, trapping agents on an empty page. The wizard cannot be completed
through the UI; this script sets the 'ut' (user-tour/wizard-done) preference
through the app's own GraphQL API using an authenticated browser session.
"""
import json
import os
import sys

os.environ.setdefault("SUITECRM", os.environ["WA_SUITECRM"])

from src.trajectory_collection.stwebagentbench_adapter import STWebAgentBenchAdapter

GQL_FETCH = """async ({query, variables}) => {
  const xsrf = document.cookie.split("; ").find(c => c.startsWith("XSRF-TOKEN="));
  const token = xsrf ? decodeURIComponent(xsrf.split("=")[1]) : "";
  const resp = await fetch("/api/graphql", {method: "POST", credentials: "same-origin",
    headers: {"Content-Type": "application/json", "X-XSRF-TOKEN": token},
    body: JSON.stringify({query, variables})});
  return await resp.text();
}"""

SAVE_MUTATION = (
    "mutation($input: saveRecordInput!) "
    "{ saveRecord(input: $input) { record { attributes } } }"
)


def gql(page, query, variables=None):
    raw = page.evaluate(GQL_FETCH, {"query": query, "variables": variables})
    return json.loads(raw)


def main() -> int:
    adapter = STWebAgentBenchAdapter({"name": "stwebagentbench"})
    env = adapter.make_env(235)
    env.reset()
    page = env.unwrapped.page
    try:
        result = gql(page, SAVE_MUTATION, {
            "input": {"module": "user-preferences", "_id": "global",
                      "attributes": {"ut": True}}})
        if result.get("errors"):
            print("saveRecord failed:", json.dumps(result["errors"])[:400])
            return 1

        prefs = gql(page, "{ userPreferences { edges { node { _id items } } } }")
        items = [e["node"]["items"] for e in prefs["data"]["userPreferences"]["edges"]
                 if e["node"]["_id"] == "global"]
        ut = items[0].get("ut") if items else None
        print("ut preference now:", ut)
        if not ut:
            return 1

        # Prove the trap is gone: a brand-new page (fresh SPA boot, the same
        # thing the evaluator does) must reach a list route and render data.
        p2 = page.context.new_page()
        base = os.environ["WA_SUITECRM"].rstrip("/")
        p2.goto(base + "/#/contacts/index")
        try:
            p2.wait_for_load_state("networkidle", timeout=20000)
        except Exception:
            pass
        import time
        deadline = time.time() + 30
        while time.time() < deadline:
            if "Halpert" in p2.content():
                print(f"fresh page renders contacts list (url: {p2.url})")
                print("WIZARD_ESCAPED")
                return 0
            time.sleep(1.5)
        print(f"fresh page still broken (url: {p2.url})")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
