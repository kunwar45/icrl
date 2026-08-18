#!/usr/bin/env python3
# ABOUTME: Go/no-go probe for AI2-THOR headless rendering on an offline GPU compute node
# ABOUTME: Run via sbatch on a GPU node; prints THOR_OK / THOR_FAIL plus a diagnosis
os.environ.setdefault("HF_HUB_OFFLINE", "1")

print("python:", sys.version.split()[0], flush=True)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

try:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
except Exception:
    traceback.print_exc(); print("THOR_FAIL import"); sys.exit(1)

t0 = time.time()
try:
    c = Controller(platform=CloudRendering, scene="FloorPlan1",
                   width=300, height=300, quality="Low",
                   gpu_device=0, local_executable_path=None)
    print(f"controller started in {time.time()-t0:.1f}s", flush=True)
except Exception:
    traceback.print_exc(); print("THOR_FAIL controller_start"); sys.exit(1)

try:
    ev = c.step(action="MoveAhead")
    print("step ok:", ev.metadata["lastActionSuccess"],
          "| objects:", len(ev.metadata["objects"]),
          "| frame:", None if ev.frame is None else ev.frame.shape, flush=True)
    # a hazard-relevant interaction: is object state manipulable?
    knives = [o for o in ev.metadata["objects"] if "Knife" in o["objectType"]]
    print("knives in scene:", len(knives), flush=True)
    t1 = time.time()
    for _ in range(20):
        c.step(action="RotateRight")
    print(f"20 steps in {time.time()-t1:.2f}s -> {20/(time.time()-t1):.1f} steps/s", flush=True)
    c.stop()
    print("THOR_OK")
except Exception:
    traceback.print_exc(); print("THOR_FAIL step"); sys.exit(1)
