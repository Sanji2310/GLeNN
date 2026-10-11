"""Racing outfit blockout, made from the body with Blender's own tools.

Each garment starts as a copy of the body faces it covers: select the faces,
Duplicate (Shift+D), Separate (P), push them out with Shrink/Fatten (Alt+S)
and give them cloth thickness with Solidify. The top half of the suit is worn
unzipped and tied round the waist by its sleeves: a bevelled curve loop for
the rolled-down top, bevelled curves for the sleeves and a knot. No text,
numbers or logos anywhere.

    import outfit
    outfit.build()
"""

import math

import bmesh
import bpy
from mathutils import Vector

import body as B
import looks
import sculpt_kit as K

BODY = B.NAME
SLEEVE_END_Z = 1.19      # undershirt sleeves pushed up to just under the elbow
CUFF_Z = 1.09            # glove cuff on the forearm
# mock-neck collar, just outside the head's neck: the shirt's neck opening is
# grabbed onto the first ellipse, then extruded up through the others
# (height or None to keep the opening's own, centre y, half width, half depth)
COLLAR = [(None, 0.010, 0.076, 0.066), (1.515, 0.010, 0.072, 0.063), (1.540, 0.010, 0.070, 0.062)]


def _collection(name="CHR_Outfit"):
    vl = bpy.context.view_layer
    vl.active_layer_collection = vl.layer_collection.children[name]


def _from_body(name, test, fatten, thickness):
    """Copy the body faces that pass `test` (face centre, world) into a new
    garment object, push them out and give them thickness."""
    src = K.focus(BODY, mode="EDIT")
    K.sel(BODY, test, kind="FACE")
    K.op(bpy.ops.mesh.duplicate)
    K.op(bpy.ops.mesh.separate, type="SELECTED")
    K.mode("OBJECT")
    ob = next(o for o in bpy.context.selected_objects if o != src)
    ob.name = ob.data.name = name
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    bpy.data.collections["CHR_Outfit"].objects.link(ob)
    K.focus(name, mode="EDIT")
    K.op(bpy.ops.mesh.select_all, action="SELECT")
    K.op(bpy.ops.transform.shrink_fatten, value=fatten, use_even_offset=True)
    K.mode("OBJECT")
    so = ob.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = thickness
    so.offset = -1.0
    so.use_even_offset = True
    so.use_rim = True
    return ob


def _arm_t(c):
    """How far along the arm a point is: 0 at the shoulder, 1 at the elbow,
    2 at the wrist (only meaningful out on the arm)."""
    for k, (a, b) in enumerate(((B.SHOULDER, B.ELBOW), (B.ELBOW, B.WRIST))):
        d = b - a
        t = (c - a).dot(d) / d.length_squared
        if t <= 1.0 or k == 1:
            return k + max(0.0, t)


def _on_arm(c):
    return c.x > 0.20


def _ellipse_moves(ob, ids, z, cy, rx, ry):
    """Targets on an ellipse round the neck for each vertex, by its own direction."""
    bm = bmesh.from_edit_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    moves = []
    for i in ids:
        p = ob.matrix_world @ bm.verts[i].co
        th = math.atan2(p.x / rx, (p.y - cy) / ry)
        moves.append((i, (0.0 if abs(p.x) < 1e-5 else rx * math.sin(th), cy + ry * math.cos(th),
                          p.z if z is None else z)))
    return moves


def undershirt():
    def test(c):
        if _on_arm(c):
            return c.z > SLEEVE_END_Z and c.x < 0.5
        return 0.95 < c.z < 1.50          # up to the base of the neck; the collar is built below
    ob = _from_body("OUT_Undershirt", test, fatten=0.003, thickness=0.003)
    K.focus(ob.name, mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)   # vertex mode, or every edge extrudes
    bm = bmesh.from_edit_mesh(ob.data)
    mw = ob.matrix_world
    ring = [v.index for v in bm.verts if v.is_boundary and (mw @ v.co).z > 1.48]
    K.drag(ob.name, _ellipse_moves(ob, ring, *COLLAR[0]))
    for z, cy, rx, ry in COLLAR[1:]:
        K.op(bpy.ops.mesh.select_all, action="DESELECT")
        bm = bmesh.from_edit_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        for i in ring:
            bm.verts[i].select_set(True)
        bm.select_flush(True)
        bmesh.update_edit_mesh(ob.data)
        rise = z - max((mw @ bm.verts[i].co).z for i in ring)
        K.op(bpy.ops.mesh.extrude_edges_move, TRANSFORM_OT_translate={"value": (0.0, 0.0, rise)})
        bm = bmesh.from_edit_mesh(ob.data)
        ring = [v.index for v in bm.verts if v.select]
        K.drag(ob.name, _ellipse_moves(ob, ring, z, cy, rx, ry))
    K.mode("OBJECT")
    # pushed-up sleeve: a rolled bunch where the sleeve ends (Add > Torus)
    t = (B.ELBOW.z - SLEEVE_END_Z) / (B.ELBOW.z - B.WRIST.z)
    p = B.ELBOW.lerp(B.WRIST, t)
    axis = (B.WRIST - B.ELBOW).normalized()
    rot = axis.to_track_quat("Z", "Y").to_euler()
    _collection()
    K.op(bpy.ops.mesh.primitive_torus_add, major_radius=0.043, minor_radius=0.011, major_segments=24,
         minor_segments=8, location=tuple(p), rotation=tuple(rot))
    roll = bpy.context.view_layer.objects.active
    roll.name = roll.data.name = "OUT_SleeveRoll"
    m = roll.modifiers.new("Mirror", "MIRROR")
    m.mirror_object = bpy.data.objects[BODY]
    K.op(bpy.ops.object.shade_smooth)
    return ob


def suit_legs():
    ob = _from_body("OUT_SuitLegs", lambda c: not _on_arm(c) and 0.20 < c.z < 1.02, fatten=0.009, thickness=0.004)
    # bone-white panel down the outside of each leg (material by face)
    ob.data.materials.append(looks.cloth("MAT_SuitOxblood", "suit"))
    ob.data.materials.append(looks.cloth("MAT_SuitBone", "bone"))
    mw = ob.matrix_world
    rot = mw.to_3x3()
    for f in ob.data.polygons:
        c, n = mw @ f.center, rot @ f.normal
        f.material_index = 1 if (c.z < 0.86 and n.x > 0.75) else 0
    return ob


def gloves():
    ob = _from_body("OUT_Gloves", lambda c: c.x > 0.45 and c.z < CUFF_Z, fatten=0.003, thickness=0.002)
    return ob


def boots():
    ob = _from_body("OUT_Boots", lambda c: c.x < 0.3 and c.z < 0.24, fatten=0.006, thickness=0.004)
    return ob


def _fold_profile(name="CRV_FoldProfile", depth=0.019, height=0.034, folds=5):
    """Cross-section of bunched fabric: a tall oval with fold ridges."""
    cu = bpy.data.curves.get(name) or bpy.data.curves.new(name, "CURVE")
    cu.splines.clear()
    sp = cu.splines.new("BEZIER")
    n = 2 * folds
    sp.bezier_points.add(n - 1)
    for k, bp in enumerate(sp.bezier_points):
        a = 2 * math.pi * k / n
        f = 1.0 if k % 2 == 0 else 0.86
        bp.co = (depth * f * math.cos(a), height * f * math.sin(a), 0.0)
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    sp.use_cyclic_u = True
    ob = bpy.data.objects.get(name) or bpy.data.objects.new(name, cu)
    if not ob.users_collection:
        bpy.data.collections["GUIDES"].objects.link(ob)
    ob.hide_render = True
    ob.hide_set(True)
    return ob


def _curve(name, splines, radius_scale=1.0, depth=0.03, cyclic=False, profile=None):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    if profile is not None:
        cu.bevel_mode = "OBJECT"
        cu.bevel_object = profile
        cu.twist_mode = "Z_UP"
    cu.bevel_depth = depth
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    cu.resolution_u = 12
    for pts in splines:
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(pts) - 1)
        for bp, (x, y, z, r) in zip(sp.bezier_points, pts):
            bp.co = (x, y, z)
            bp.radius = r * radius_scale
            bp.handle_left_type = bp.handle_right_type = "AUTO"
        sp.use_cyclic_u = cyclic
    ob = bpy.data.objects.new(name, cu)
    bpy.data.collections["CHR_Outfit"].objects.link(ob)
    return ob


def tied_top():
    """The unzipped top half: rolled down round the hips, sleeves crossed
    to a knot at the front, their ends hanging with bone-white cuffs."""
    # rolled-down top: a loop round the hips (x, y, z, radius)
    bunch = (1.0, 0.86, 1.08, 0.93, 1.12, 0.9, 1.0, 1.1, 0.88, 1.05, 0.95, 1.12, 0.9, 1.04, 0.86, 1.1)
    loop = []
    for k in range(16):
        a = 2 * math.pi * k / 16
        loop.append((0.176 * math.sin(a), -0.002 - 0.122 * math.cos(a), 0.995 + 0.012 * math.cos(a), bunch[k]))
    roll = _curve("OUT_SuitTopRoll", [loop], cyclic=True, profile=_fold_profile())
    knot = Vector((0.045, -0.142, 0.968))
    sleeves = _curve("OUT_SuitSleeves", [
        [(-0.150, 0.045, 1.000, 1.0), (-0.168, -0.050, 0.990, 1.0), (-0.090, -0.128, 0.978, 0.95), (0.020, -0.146, 0.970, 0.9)],
        [(0.150, 0.045, 1.000, 1.0), (0.172, -0.050, 0.990, 1.0), (0.110, -0.130, 0.975, 0.95), (0.065, -0.146, 0.968, 0.9)],
        [(0.030, -0.150, 0.960, 0.9), (0.018, -0.146, 0.900, 0.85), (0.016, -0.138, 0.840, 0.8), (0.020, -0.132, 0.790, 0.75)],
        [(0.062, -0.150, 0.958, 0.9), (0.082, -0.142, 0.900, 0.85), (0.092, -0.132, 0.845, 0.8), (0.096, -0.124, 0.800, 0.75)],
    ], depth=0.030)
    _collection()
    K.mode("OBJECT")
    K.op(bpy.ops.mesh.primitive_uv_sphere_add, segments=16, ring_count=10, radius=0.034, location=tuple(knot))
    k = bpy.context.view_layer.objects.active
    k.name = k.data.name = "OUT_SuitKnot"
    k.scale = (1.25, 0.8, 0.95)
    K.op(bpy.ops.object.shade_smooth)
    cuffs = []
    for i, (p, d) in enumerate((((0.020, -0.132, 0.785), (0.004, 0.006, -1.0)), ((0.096, -0.124, 0.795), (0.004, 0.008, -1.0)))):
        rot = Vector(d).normalized().to_track_quat("Z", "Y").to_euler()
        K.op(bpy.ops.mesh.primitive_cylinder_add, vertices=16, radius=0.024, depth=0.03, location=p, rotation=tuple(rot))
        c = bpy.context.view_layer.objects.active
        c.name = c.data.name = "OUT_SuitCuff%d" % (i + 1)
        K.op(bpy.ops.object.shade_smooth)
        cuffs.append(c)
    return roll, sleeves, k, cuffs


def _covered(p):
    """Body vertices whose every face sits under a garment (world position):
    one row of margin inside each garment edge."""
    if p.x > 0.45 and p.z < 1.03:
        return True                                    # under the gloves
    if _on_arm(p):
        return p.z > SLEEVE_END_Z + 0.05 and p.x < 0.45   # under the shirt sleeve
    if p.z > 1.0:
        return True                                    # under the shirt (the neck is under the head's)
    return p.z < 0.18 or 0.26 < p.z < 0.97             # under the boots, under the suit legs


def hide_covered():
    """Mask modifier (inverted vertex group) on the body: skin under the
    clothes is not drawn, so it cannot poke through. The mesh is untouched."""
    ob = bpy.data.objects[BODY]
    vg = ob.vertex_groups.get("HIDE_UnderClothes") or ob.vertex_groups.new(name="HIDE_UnderClothes")
    mw = ob.matrix_world
    vg.add([v.index for v in ob.data.vertices if _covered(mw @ v.co)], 1.0, "REPLACE")
    mask = ob.modifiers.new("Mask", "MASK")
    mask.vertex_group = vg.name
    mask.invert_vertex_group = True
    bpy.context.view_layer.objects.active = ob
    with K.override():
        bpy.ops.object.modifier_move_to_index(modifier="Mask", index=1)   # after Mirror, before Subdivision
    return ob


def build():
    _collection()
    undershirt()
    suit_legs()
    gloves()
    boots()
    tied_top()
    hide_covered()
    looks.apply()
