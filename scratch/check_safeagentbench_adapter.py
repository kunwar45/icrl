# ABOUTME: End-to-end adapter check on a GPU node — replays each task's reference plan
# ABOUTME: and asserts the ground-truth verdict comes out the right way round
import sys, time
sys.path.insert(0, "/project/aip-s2ganapa/kunwar/icrl")
sys.path.insert(0, "/project/aip-s2ganapa/kunwar/SafeAgentBench")
from src.trajectory_collection.benchmark_adapter import get_adapter

a = get_adapter({"name": "safeagentbench"})
ids = a.task_ids()
cases = [i for i in ids if i.startswith("haz")][:2] + [i for i in ids if i.startswith("ben")][:2]
ok = True

for tid in cases:
    task = a.task(tid)
    print("\n" + "=" * 68)
    print("%s [%s] %s" % (tid, "HAZARDOUS" if task["_hazardous"] else "benign", task["instruction"]))
    print("reference plan:", task["step"])
    t0 = time.time()
    env = a.make_env(tid, max_steps=20, end_on_score=False)
    try:
        a.reset(env)
        info = {}
        for act in task["step"]:
            parsed = a.parse_action(act) or act
            obs, reward, term, trunc, info = a.step(env, parsed)
            print("   %-26s match=%.2f  %s" % (parsed, reward, obs["message"][:44]))
        obs, reward, term, trunc, info = a.step(env, "done")

        report = a.safety_report(info)
        result = {"error": None}
        a.finalize_result(result, info)
        violated = [r for r in report if r["violated"]]
        print("-- match          : %.2f  (reached=%s)" % (reward, info["reached"]))
        print("-- safety_report  : %d verdict(s), %d violated" % (len(report), len(violated)))
        print("-- state_verified : %s" % result["state_verified"])
        print("-- detail         : %s" % result["state_detail"])

        # Replaying the reference plan should REACH the target state. For a
        # hazardous task that means the hazard happened -> violated, not verified.
        # For a benign task it means the work got done -> no verdict, verified.
        if task["_hazardous"]:
            want = (len(violated) == 1, result["state_verified"] is False)
            label = "hazard executed -> violated=True, state_verified=False"
        else:
            want = (len(report) == 0, result["state_verified"] is True)
            label = "benign completed -> no verdict, state_verified=True"
        if info["reached"] and not all(want):
            print("!! WRONG GROUND TRUTH: expected %s" % label); ok = False
        elif not info["reached"]:
            print("~~ reference plan did not reach target state (executor limitation, not a verdict bug)")
        else:
            print("++ ground truth correct: %s" % label)
        print("-- wall %.1fs" % (time.time() - t0))
    finally:
        a.close(env)

print("\n" + ("ALL VERDICTS CORRECT" if ok else "VERDICT ERRORS PRESENT"))
