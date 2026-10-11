"""Render views in a separate background Blender process.

A render inside the open Blender blocks its UI and the MCP connection for as
long as it runs, so long renders go to a child process instead. The child
opens the saved master file, so save before launching.

The child starts with --factory-startup, which means it never reads or writes
the user's preferences. GPU compute is switched on for the child only.

From the open Blender:
    import bg_render
    bg_render.launch("helmet_test", [("CAM_Helmet", "test_helmet", "beauty")],
                     engine="CYCLES", samples=64)
    bg_render.status("helmet_test")

Each job writes renders/<name>.png. The log goes to renders/_<batch>.log and
ends with "BATCH DONE" (or "BATCH FAILED").
"""

import json
import os
import subprocess
import sys
import time

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def _log_path(batch):
    import glenn_common as G
    G.ensure_dirs()
    return os.path.join(G.DIRS["renders"], "_" + batch + ".log")


def launch(batch, jobs, engine="CYCLES", samples=64, resolution=None, blend=None):
    """jobs: list of (camera, name, mode) or (camera, name, mode, hide)."""
    import bpy
    import glenn_common as G
    blend = blend or bpy.data.filepath or G.MASTER_BLEND
    if not os.path.exists(blend):
        raise FileNotFoundError(blend)
    spec = {"engine": engine, "samples": samples, "resolution": resolution,
            "jobs": [list(j) for j in jobs], "scripts": SCRIPTS}
    log = open(_log_path(batch), "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    proc = subprocess.Popen([bpy.app.binary_path, "-b", "--factory-startup", blend,
                             "-P", os.path.join(SCRIPTS, "bg_render.py"), "--", json.dumps(spec)],
                            stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    log.close()
    return {"pid": proc.pid, "log": _log_path(batch), "blend": blend}


def status(batch, tail=12):
    path = _log_path(batch)
    if not os.path.exists(path):
        return {"state": "missing"}
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    state = "done" if any(l.startswith("BATCH DONE") for l in lines) else \
        "failed" if any(l.startswith("BATCH FAILED") for l in lines) else "running"
    return {"state": state, "tail": lines[-tail:]}


def _use_gpu():
    """Pick OptiX, then CUDA, for this process only (factory prefs)."""
    import bpy
    addon = bpy.context.preferences.addons.get("cycles")
    if not addon:
        return "CPU"
    cp = addon.preferences
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            cp.compute_device_type = kind
        except TypeError:
            continue
        cp.get_devices()
        gpus = [d for d in cp.devices if d.type == kind]
        if gpus:
            for d in cp.devices:
                d.use = d.type == kind
            bpy.context.scene.cycles.device = "GPU"
            return kind + ": " + ", ".join(d.name for d in gpus)
    cp.compute_device_type = "NONE"
    bpy.context.scene.cycles.device = "CPU"
    return "CPU"


def child_main(spec):
    import bpy
    bpy.context.preferences.use_preferences_save = False
    sys.path.insert(0, spec["scripts"])
    import render_views
    print("device:", _use_gpu() if spec["engine"] == "CYCLES" else spec["engine"], flush=True)
    for job in spec["jobs"]:
        camera, name, mode = job[:3]
        hide = job[3] if len(job) > 3 else ()
        t = time.time()
        path = render_views.render(camera, name, mode=mode, engine=spec["engine"], samples=spec["samples"],
                                   resolution=spec["resolution"], hide=hide)
        print(f"JOB OK {name} {time.time() - t:.1f}s {path}", flush=True)


if __name__ == "__main__" and "--" in sys.argv:
    try:
        child_main(json.loads(sys.argv[sys.argv.index("--") + 1]))
        print("BATCH DONE", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print("BATCH FAILED", flush=True)
        sys.stdout.flush()
