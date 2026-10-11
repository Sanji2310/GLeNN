"""Build the whole GLeNN driver into the master .blend.

Meshing the sculpt fields takes a minute or two of numpy work, which would
freeze the open Blender (and its MCP connection), so the build runs in a
separate background Blender process that opens the master file, rebuilds
every owned object and saves it. The open Blender then reloads the file.

Before the child starts, the current master is copied to
backup/GLeNN_Character_Master_<timestamp>.blend.

From the open Blender (save first; the child opens the file on disk):
    import build_character
    build_character.launch()
    build_character.status()        # "running", "done" or "failed"
    bpy.ops.wm.revert_mainfile()    # once done, load the rebuilt file

Or from a shell:
    blender -b --factory-startup project/GLeNN_Character_Master.blend \\
        -P scripts/build_character.py -- "{}"
"""

import json
import os
import shutil
import subprocess
import sys
import time

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

# Mesh resolution of each sculpt (voxel size in metres)
DEFAULTS = {"head": 0.0011, "hair": 0.0013, "suit": 0.003, "gloves": 0.0016, "boots": 0.0022}


def _log_path():
    import glenn_common as G
    G.ensure_dirs()
    return os.path.join(G.DIRS["reports"], "_build.log")


def backup_master():
    import glenn_common as G
    G.ensure_dirs()
    if not os.path.exists(G.MASTER_BLEND):
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(G.DIRS["backup"], "GLeNN_Character_Master_%s.blend" % stamp)
    shutil.copy2(G.MASTER_BLEND, dst)
    return dst


def launch(**voxels):
    import bpy
    import glenn_common as G
    if bpy.data.is_dirty and bpy.data.filepath == G.MASTER_BLEND:
        raise RuntimeError("Save or revert the master file first; the build replaces it.")
    backup = backup_master()
    spec = dict(DEFAULTS, **voxels)
    log = open(_log_path(), "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    args = [bpy.app.binary_path, "-b", "--factory-startup"]
    if os.path.exists(G.MASTER_BLEND):
        args.append(G.MASTER_BLEND)
    args += ["-P", os.path.join(SCRIPTS, "build_character.py"), "--", json.dumps(spec)]
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    log.close()
    return {"pid": proc.pid, "log": _log_path(), "backup": backup}


def status(tail=15):
    path = _log_path()
    if not os.path.exists(path):
        return {"state": "missing"}
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    state = "done" if any(l.startswith("BUILD DONE") for l in lines) else \
        "failed" if any(l.startswith("BUILD FAILED") for l in lines) else "running"
    return {"state": state, "tail": lines[-tail:]}


def build(spec):
    """Rebuild everything in the current file. Returns per-object face counts."""
    import bpy
    import build_body
    import build_hair
    import build_head
    import build_helmet
    import glenn_common as G
    import setup_scene
    t0 = time.time()
    steps = [("scene", setup_scene.run),
             ("head", lambda: build_head.run(voxel=spec["head"])),
             ("hair", lambda: build_hair.run(voxel=spec["hair"])),
             ("outfit", lambda: build_body.run(spec["suit"], spec["gloves"], spec["boots"])),
             ("helmet", build_helmet.run)]
    for name, fn in steps:
        t = time.time()
        fn()
        print("STEP OK %s %.1fs" % (name, time.time() - t), flush=True)
    faces = {}
    for coll in ("GLN_Character", "GLN_Helmet"):
        c = bpy.data.collections.get(coll)
        for o in (c.all_objects if c else []):
            if o.type == "MESH":
                faces[o.name] = len(o.data.polygons)
    print("FACES " + json.dumps(faces), flush=True)
    print("TOTAL %.1fs" % (time.time() - t0), flush=True)
    return faces


def child_main(spec):
    import bpy
    import glenn_common as G
    bpy.context.preferences.use_preferences_save = False
    build(spec)
    bpy.ops.wm.save_as_mainfile(filepath=G.MASTER_BLEND, relative_remap=True)
    print("SAVED " + G.MASTER_BLEND, flush=True)


if __name__ == "__main__" and "--" in sys.argv:
    try:
        child_main(dict(DEFAULTS, **json.loads(sys.argv[sys.argv.index("--") + 1] or "{}")))
        print("BUILD DONE", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print("BUILD FAILED", flush=True)
        sys.stdout.flush()
