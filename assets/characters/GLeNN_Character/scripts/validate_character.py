"""Measure and check the GLeNN character. Nothing here changes the scene.

Reports, per object and per group:
  - base and evaluated vertex / face / triangle counts
  - n-gons, triangles, non-manifold edges, loose vertices, zero-area faces
  - UV layers, materials, unapplied scale, missing names
and writes reports/validation_report.md plus a JSON copy.

Run inside Blender 5.2.2:
    import validate_character; validate_character.run()
"""

import json
import os
from datetime import datetime

import bmesh
import bpy

import glenn_common as G

GROUPS = ("GLN_Body", "GLN_Head", "GLN_Hair", "GLN_Outfit", "GLN_Helmet")


def inspect(obj, depsgraph):
    me = obj.data
    info = {
        "object": obj.name,
        "verts": len(me.vertices),
        "faces": len(me.polygons),
        "tris": sum(len(p.vertices) - 2 for p in me.polygons),
        "ngons": sum(1 for p in me.polygons if len(p.vertices) > 4),
        "tri_faces": sum(1 for p in me.polygons if len(p.vertices) == 3),
        "uv_layers": [uv.name for uv in me.uv_layers],
        "materials": [s.material.name if s.material else None for s in obj.material_slots],
        "scale_applied": all(abs(s - 1.0) < 1e-4 for s in obj.scale),
        "modifiers": [m.type for m in obj.modifiers],
    }
    bm = bmesh.new()
    bm.from_mesh(me)
    info["non_manifold_edges"] = sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
    info["boundary_edges"] = sum(1 for e in bm.edges if e.is_boundary)
    info["loose_verts"] = sum(1 for v in bm.verts if not v.link_edges)
    info["zero_area_faces"] = sum(1 for f in bm.faces if f.calc_area() < 1e-10)
    bm.free()
    ev = obj.evaluated_get(depsgraph)
    em = ev.to_mesh()
    info["eval_verts"] = len(em.vertices)
    info["eval_tris"] = sum(len(p.vertices) - 2 for p in em.polygons)
    ev.to_mesh_clear()
    return info


def run(write=True):
    dg = bpy.context.evaluated_depsgraph_get()
    rows, totals = [], {}
    for group in GROUPS:
        coll = bpy.data.collections.get(group)
        if not coll:
            continue
        for obj in sorted(coll.all_objects, key=lambda o: o.name):
            if obj.type != "MESH":
                continue
            info = inspect(obj, dg)
            info["group"] = group
            rows.append(info)
            t = totals.setdefault(group, {"tris": 0, "eval_tris": 0, "verts": 0})
            t["tris"] += info["tris"]
            t["eval_tris"] += info["eval_tris"]
            t["verts"] += info["verts"]
    issues = []
    for r in rows:
        for key in ("non_manifold_edges", "loose_verts", "zero_area_faces"):
            if r[key]:
                issues.append(f"{r['object']}: {r[key]} {key.replace('_', ' ')}")
        if not r["uv_layers"]:
            issues.append(f"{r['object']}: no UV map")
        if not r["materials"] or None in r["materials"]:
            issues.append(f"{r['object']}: missing material")
        if not r["scale_applied"]:
            issues.append(f"{r['object']}: scale not applied")
    report = {"time": datetime.now().isoformat(timespec="seconds"), "blender": bpy.app.version_string,
              "file": bpy.data.filepath, "objects": rows, "totals": totals, "issues": issues}
    if write:
        G.ensure_dirs()
        with open(os.path.join(G.DIRS["reports"], "validation_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        lines = [f"# Validation report", "", f"{report['time']}, Blender {report['blender']}", "",
                 f"File: `{report['file']}`", "", "## Triangle budget by group", "",
                 "| Group | Base tris | Evaluated tris | Base verts |", "| --- | ---: | ---: | ---: |"]
        for g, t in totals.items():
            lines.append(f"| {g} | {t['tris']:,} | {t['eval_tris']:,} | {t['verts']:,} |")
        lines += ["", "## Objects", "",
                  "| Object | Tris | Eval tris | N-gons | Non-manifold | UVs | Materials |",
                  "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
        for r in rows:
            lines.append(f"| {r['object']} | {r['tris']:,} | {r['eval_tris']:,} | {r['ngons']} | "
                         f"{r['non_manifold_edges']} | {', '.join(r['uv_layers']) or 'none'} | "
                         f"{', '.join(m or 'none' for m in r['materials'])} |")
        lines += ["", "## Issues", ""] + ([f"- {i}" for i in issues] or ["- None found"])
        with open(os.path.join(G.DIRS["reports"], "validation_report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return {"totals": totals, "issues": issues, "objects": len(rows)}
