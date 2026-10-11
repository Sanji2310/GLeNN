"""Brows and hair for the racer, built from Blender primitives and curves.

Brows: Add > Mesh > Grid strip laid over the brow ridge, kept on the skin
with a Shrinkwrap modifier and given thickness with Solidify, mirrored
across the head.

    import groom
    groom.brows()
"""

import bmesh
import bpy
from mathutils import Vector

import sculpt_kit as K

HEAD = "CHR_Head"

# Brow strip, inner end to tail: (x, top z, bottom z). Thick and level at the
# inner end, a low angular peak two thirds along, thinning to the tail.
BROW = [(0.0085, 1.7030, 1.6960), (0.0190, 1.7062, 1.6990), (0.0300, 1.7092, 1.7025),
        (0.0410, 1.7098, 1.7045), (0.0500, 1.7068, 1.7035), (0.0565, 1.7030, 1.7018)]


def _collection(name):
    vl = bpy.context.view_layer
    vl.active_layer_collection = vl.layer_collection.children[name]


def brows(name="CHR_Brows", head=HEAD):
    _collection("CHR_Hair")
    K.mode("OBJECT")
    K.op(bpy.ops.mesh.primitive_grid_add, x_subdivisions=len(BROW) - 1, y_subdivisions=1, size=0.05,
         location=(0.0, 0.0, 0.0))
    ob = bpy.context.view_layer.objects.active
    ob.name = ob.data.name = name
    h = bpy.data.objects[head]
    K.mode("EDIT")
    bm = bmesh.from_edit_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    cols = sorted({round(v.co.x, 5) for v in bm.verts})
    moves = []
    for v in bm.verts:
        x, top, bottom = BROW[cols.index(round(v.co.x, 5))]
        z = top if v.co.y > 0 else bottom
        hit = K.surface(head, (x, -0.3, z), (0, 1, 0))
        moves.append((v.index, (x, hit[1] if hit else -0.09, z)))
    K.drag(name, moves)
    K.mode("OBJECT")
    sw = ob.modifiers.new("Shrinkwrap", "SHRINKWRAP")
    sw.target = h
    sw.wrap_method = "NEAREST_SURFACEPOINT"
    sw.offset = 0.0008
    so = ob.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = 0.0016
    so.offset = 1.0
    mi = ob.modifiers.new("Mirror", "MIRROR")
    mi.mirror_object = h
    K.op(bpy.ops.object.shade_smooth)
    return ob


# ---------------------------------------------------------------- hair
# Undercut: the buzzed sides and back are a shell copied off the scalp
# (Duplicate, Separate) with Solidify. The long top is a set of clumps, each
# a Bezier curve bevelled with a flattened profile and tapered by its point
# radii, as in curve-based stylized hair.

def _hairline(p):
    """True above the hairline of the buzzed undercut (world position)."""
    ax = abs(p.x)
    if p.y < -0.045:
        return p.z > 1.757 - 0.38 * ax
    if p.y < -0.035:
        return p.z > (1.732 if ax > 0.05 else 1.745)
    if p.y < -0.008 and ax > 0.058:
        return p.z > 1.655          # short sideburn in front of the ear
    if p.y < -0.012:
        return p.z > 1.732
    if p.y < 0.040 and ax > 0.055:
        return p.z > 1.690          # just over the ear
    return p.z > 1.598 if p.y > 0.040 else p.z > 1.70


HAIRLINE_SMOOTH = 3


def scalp(name="CHR_HairCap", head=HEAD):
    _collection("CHR_Hair")
    h = K.focus(head, mode="EDIT")
    K.sel(head, _hairline, kind="FACE")
    K.op(bpy.ops.mesh.duplicate)
    K.op(bpy.ops.mesh.separate, type="SELECTED")
    K.mode("OBJECT")
    ob = next(o for o in bpy.context.selected_objects if o != h)
    ob.name = ob.data.name = name
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    bpy.data.collections["CHR_Hair"].objects.link(ob)
    # the hairline comes out in steps (whole faces): select its edge loop and
    # Smooth Vertices so it runs clean (the midline ends stay put)
    K.focus(name, mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    K.op(bpy.ops.mesh.select_all, action="DESELECT")
    bm = bmesh.from_edit_mesh(ob.data)
    for v in bm.verts:
        v.select_set(v.is_boundary and abs(v.co.x) > 1e-4)
    bmesh.update_edit_mesh(ob.data)
    K.op(bpy.ops.mesh.vertices_smooth, factor=0.6, repeat=HAIRLINE_SMOOTH)
    K.mode("OBJECT")
    so = ob.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = 0.0016
    so.offset = 1.0
    # Solidify last, after Mirror and Subdivision Surface, so the shell follows the smooth head
    bpy.context.view_layer.objects.active = ob
    with K.override():
        bpy.ops.object.modifier_move_to_index(modifier="Solidify", index=len(ob.modifiers) - 1)
    return ob


# Clumps: (x, y, height above the scalp, radius) from root to tip. The side
# part is on his right (-X); the top sweeps up, back and over to his left,
# two locks fall forward on the right of the forehead.
CLUMPS = [
    # front quiff
    [(-0.030, -0.064, 0.002, 0.010), (-0.012, -0.086, 0.020, 0.016), (0.015, -0.076, 0.030, 0.017), (0.040, -0.048, 0.026, 0.012), (0.060, -0.012, 0.012, 0.002)],
    [(-0.010, -0.068, 0.002, 0.011), (0.010, -0.084, 0.018, 0.015), (0.035, -0.068, 0.024, 0.014), (0.058, -0.038, 0.016, 0.009), (0.072, -0.004, 0.006, 0.002)],
    [(0.012, -0.066, 0.002, 0.010), (0.032, -0.076, 0.014, 0.013), (0.055, -0.056, 0.016, 0.012), (0.070, -0.028, 0.010, 0.007), (0.079, 0.002, 0.004, 0.002)],
    # top, swept back
    [(-0.030, -0.036, 0.002, 0.012), (-0.010, -0.030, 0.022, 0.017), (0.020, 0.000, 0.026, 0.017), (0.045, 0.038, 0.018, 0.011), (0.055, 0.074, 0.006, 0.002)],
    [(-0.028, 0.000, 0.002, 0.012), (0.000, 0.010, 0.020, 0.016), (0.030, 0.040, 0.020, 0.015), (0.050, 0.070, 0.012, 0.009), (0.055, 0.096, 0.004, 0.002)],
    [(-0.024, 0.034, 0.002, 0.011), (0.000, 0.050, 0.016, 0.015), (0.025, 0.074, 0.014, 0.013), (0.040, 0.094, 0.008, 0.008), (0.045, 0.110, 0.003, 0.002)],
    [(0.010, -0.040, 0.002, 0.011), (0.035, -0.030, 0.020, 0.015), (0.058, -0.004, 0.016, 0.012), (0.071, 0.030, 0.008, 0.007), (0.074, 0.058, 0.003, 0.002)],
    [(0.000, 0.060, 0.002, 0.010), (0.020, 0.088, 0.012, 0.013), (0.030, 0.104, 0.008, 0.010), (0.030, 0.113, 0.003, 0.002)],
    [(0.040, 0.030, 0.002, 0.010), (0.060, 0.050, 0.010, 0.012), (0.072, 0.072, 0.006, 0.008), (0.075, 0.088, 0.002, 0.002)],
    # part side, short, swept back over the right
    [(-0.038, -0.054, 0.002, 0.008), (-0.054, -0.048, 0.010, 0.011), (-0.064, -0.020, 0.008, 0.008), (-0.068, 0.018, 0.003, 0.002)],
    [(-0.038, -0.010, 0.002, 0.008), (-0.054, 0.010, 0.010, 0.011), (-0.062, 0.045, 0.007, 0.007), (-0.060, 0.075, 0.002, 0.002)],
    # forelocks over the right of the forehead: (x, None, z, radius, gap) points
    # sit `gap` in front of the forehead at height z
    [(-0.030, -0.066, 0.004, 0.006), (-0.042, None, 1.772, 0.008, 0.014), (-0.048, None, 1.748, 0.007, 0.010),
     (-0.046, None, 1.724, 0.005, 0.006), (-0.040, None, 1.708, 0.0015, 0.004)],
    [(-0.018, -0.070, 0.004, 0.005), (-0.024, None, 1.772, 0.007, 0.014), (-0.026, None, 1.750, 0.006, 0.009),
     (-0.020, None, 1.730, 0.0015, 0.005)],
]


def _scalp_z(head, x, y):
    hit = K.surface(head, (x, y, 2.2), (0, 0, -1))
    return hit[2] if hit else 1.70


def _profile(name="CRV_HairProfile"):
    """Flattened, slightly pointed cross-section for the clumps."""
    cu = bpy.data.curves.get(name) or bpy.data.curves.new(name, "CURVE")
    cu.splines.clear()
    sp = cu.splines.new("BEZIER")
    pts = [(1.0, 0.0), (0.0, 0.42), (-1.0, 0.0), (0.0, -0.42)]
    sp.bezier_points.add(len(pts) - 1)
    for bp, (x, y) in zip(sp.bezier_points, pts):
        bp.co = (x, y, 0.0)
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    for i in (0, 2):     # crisp side edges
        sp.bezier_points[i].handle_left_type = sp.bezier_points[i].handle_right_type = "VECTOR"
    sp.use_cyclic_u = True
    ob = bpy.data.objects.get(name) or bpy.data.objects.new(name, cu)
    if not ob.users_collection:
        bpy.data.collections["GUIDES"].objects.link(ob)
    ob.hide_render = True
    ob.hide_set(True)
    return ob


def hair(name="CHR_HairTop", head=HEAD):
    _collection("CHR_Hair")
    scalp()
    profile = _profile()
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.twist_mode = "Z_UP"
    cu.bevel_mode = "OBJECT"
    cu.bevel_object = profile
    cu.use_fill_caps = True
    cu.resolution_u = 8
    for clump in CLUMPS:
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(clump) - 1)
        for bp, pt in zip(sp.bezier_points, clump):
            x, y, h, r = pt[:4]
            if y is None:
                hit = K.surface(head, (x, -0.3, h), (0, 1, 0))
                bp.co = (x, (hit[1] if hit else -0.09) - pt[4], h)
            else:
                bp.co = (x, y, _scalp_z(head, x, y) + h)
            bp.radius = r
            bp.handle_left_type = bp.handle_right_type = "AUTO"
    ob = bpy.data.objects.new(name, cu)
    bpy.data.collections["CHR_Hair"].objects.link(ob)
    return ob
