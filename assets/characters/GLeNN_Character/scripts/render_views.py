"""Presentation and inspection renders for the GLeNN character.

Beauty views use the real materials. Inspection views are kept separate:
  clay       every character object gets a neutral clay material
  wireframe  clay plus the real polygon edges (a Wireframe modifier on a
             temporary copy, so quads show as quads, not triangles)

Run inside Blender 5.2.2:
    import render_views
    render_views.render("CAM_Hero", "final_hero")
    render_views.render("CAM_Face", "wireframe_face", mode="wireframe")
"""

import os

import bpy

import glenn_common as G
import setup_scene

CHARACTER_COLLECTIONS = ("GLN_Character", "GLN_Helmet")


def _character_objects():
    objs = []
    for name in CHARACTER_COLLECTIONS:
        coll = bpy.data.collections.get(name)
        if coll:
            objs += [o for o in coll.all_objects if o.type in ("MESH", "CURVE", "FONT")]
    return objs


def _clay():
    return G.principled("MAT_InspectClay", G.PALETTE["clay"], roughness=0.6)


def _wire_mat():
    return G.principled("MAT_InspectWire", (0.004, 0.012, 0.035), roughness=0.9)


def _override(objs, mat):
    saved = []
    for o in objs:
        for slot in o.material_slots:
            saved.append((slot, slot.link, slot.material))
            slot.link = "OBJECT"
            slot.material = mat
    return saved


def _restore(saved):
    for slot, link, mat in saved:
        slot.material = None
        slot.link = link
        if link == "OBJECT":
            slot.material = mat


def _wire_copies(objs, thickness=0.0005):
    """Temporary wireframe overlays built from each object's base mesh
    (subdivision off), so the real modelled topology is what shows."""
    made = []
    wire = _wire_mat()
    for o in objs:
        if o.type != "MESH" or o.hide_render:
            continue
        dup = o.copy()
        dup.data = o.data
        dup.name = o.name + "_WIRE"
        dup["gln_owner"] = "inspect"
        for m in list(dup.modifiers):
            if m.type == "SUBSURF":
                dup.modifiers.remove(m)
        w = dup.modifiers.new("Wire", "WIREFRAME")
        w.thickness = thickness
        w.use_relative_offset = False
        w.use_even_offset = True
        w.material_offset = 0
        bpy.context.scene.collection.objects.link(dup)
        for slot in dup.material_slots:
            slot.link = "OBJECT"
            slot.material = wire
        made.append(dup)
    return made


def render(camera, name, mode="beauty", engine="BLENDER_EEVEE", samples=None, resolution=None,
           hide=()):
    """Render one view to renders/<name>.png. mode: beauty | clay | wireframe."""
    scene = bpy.context.scene
    cam = bpy.data.objects[camera]
    scene.camera = cam
    res = resolution or setup_scene.CAMERAS.get(camera, (None,) * 4 + ((1600, 1600),))[4]
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.engine = engine
    if engine == "BLENDER_EEVEE" and samples:
        scene.eevee.taa_render_samples = samples
    if engine == "CYCLES" and samples:
        scene.cycles.samples = samples
    hidden = []
    for oname in hide:
        o = bpy.data.objects.get(oname)
        if o and not o.hide_render:
            o.hide_render = True
            hidden.append(o)
    objs = _character_objects()
    saved, wires = [], []
    try:
        if mode in ("clay", "wireframe"):
            saved = _override(objs, _clay())
        if mode == "wireframe":
            wires = _wire_copies(objs)
        G.ensure_dirs()
        path = os.path.join(G.DIRS["renders"], name + ".png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
    finally:
        _restore(saved)
        for w in wires:
            bpy.data.objects.remove(w, do_unlink=True)
        for o in hidden:
            o.hide_render = False
    return path
