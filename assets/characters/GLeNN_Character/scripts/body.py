"""Body base mesh for the racer, box-modeled.

The steps are a modeler's: Add > Cube, Subdivide (3 cuts), Subdivide the
side edges once more for torso loops, delete the -X half, Mirror (Clipping)
and Subdivision Surface. Each horizontal loop of the torso is grabbed onto a
drawn cross-section. Arms are extruded (E) out of four faces at the side of
the chest, legs out of the outer four faces underneath, the neck out of the
two inner faces on top; after each extrusion its new ring is grabbed round
the limb, so there is a loop at every joint for the rig.

A-pose, 1.80 m, Z up, facing -Y.

    import body
    body.build()
"""

import math

import bmesh
import bpy
from mathutils import Vector

import sculpt_kit as K

NAME = "CHR_Body"

# Torso loops, bottom to top: (z, 9 points from the front midline round the
# side to the back midline), x >= 0.
TORSO = [
    (0.860, [(0, -0.075), (0.060, -0.072), (0.110, -0.060), (0.145, -0.030), (0.152, 0.0), (0.148, 0.035),
             (0.120, 0.072), (0.060, 0.092), (0, 0.094)]),
    (0.925, [(0, -0.088), (0.070, -0.084), (0.125, -0.065), (0.160, -0.030), (0.168, 0.005), (0.162, 0.040),
             (0.130, 0.085), (0.065, 0.108), (0, 0.104)]),
    (0.990, [(0, -0.095), (0.070, -0.090), (0.120, -0.070), (0.152, -0.035), (0.158, 0.0), (0.152, 0.035),
             (0.125, 0.070), (0.065, 0.092), (0, 0.088)]),
    (1.060, [(0, -0.095), (0.060, -0.092), (0.105, -0.075), (0.132, -0.040), (0.138, 0.0), (0.134, 0.035),
             (0.110, 0.065), (0.060, 0.080), (0, 0.074)]),
    (1.140, [(0, -0.098), (0.060, -0.095), (0.108, -0.078), (0.136, -0.040), (0.142, 0.0), (0.138, 0.038),
             (0.112, 0.070), (0.060, 0.085), (0, 0.080)]),
    (1.220, [(0, -0.105), (0.062, -0.103), (0.116, -0.085), (0.148, -0.045), (0.156, 0.0), (0.153, 0.042),
             (0.127, 0.078), (0.067, 0.095), (0, 0.092)]),
    (1.300, [(0, -0.112), (0.068, -0.113), (0.126, -0.093), (0.160, -0.050), (0.168, 0.0), (0.165, 0.045),
             (0.137, 0.086), (0.073, 0.104), (0, 0.100)]),
    (1.380, [(0, -0.100), (0.074, -0.103), (0.138, -0.086), (0.176, -0.045), (0.184, 0.0), (0.180, 0.045),
             (0.148, 0.083), (0.078, 0.099), (0, 0.097)]),
    (1.460, [(0, -0.060), (0.085, -0.062), (0.160, -0.048), (0.198, -0.026), (0.204, 0.0), (0.199, 0.026),
             (0.160, 0.052), (0.085, 0.066), (0, 0.065)]),
]
# inside of the top (shoulders, neck base) and of the bottom (crotch), by
# cube grid position (x, y) in -1..1 units
TOP_INNER = {(0.0, -0.5): (0.0, -0.035, 1.495), (0.0, 0.0): (0.0, 0.012, 1.500), (0.0, 0.5): (0.0, 0.055, 1.495),
             (0.5, -0.5): (0.075, -0.035, 1.485), (0.5, 0.0): (0.080, 0.010, 1.490), (0.5, 0.5): (0.075, 0.050, 1.485)}
BOTTOM_INNER = {(0.0, -0.5): (0.0, -0.040, 0.845), (0.0, 0.0): (0.0, 0.0, 0.835), (0.0, 0.5): (0.0, 0.045, 0.845),
                (0.5, -0.5): (0.075, -0.045, 0.850), (0.5, 0.0): (0.072, 0.0, 0.842), (0.5, 0.5): (0.075, 0.050, 0.850)}

# limbs: joints, then rings as (segment, t along it, radius across, radius front-back, y offset)
SHOULDER, ELBOW, WRIST = Vector((0.192, 0.006, 1.412)), Vector((0.392, 0.014, 1.232)), Vector((0.552, 0.0, 1.045))
KNUCKLES, FINGERTIPS = Vector((0.616, -0.006, 0.957)), Vector((0.652, -0.010, 0.890))
ARM = [((SHOULDER, ELBOW), 0.12, 0.060, 0.060, 0.0), ((SHOULDER, ELBOW), 0.45, 0.049, 0.048, 0.0),
       ((SHOULDER, ELBOW), 0.80, 0.040, 0.040, 0.0), ((SHOULDER, ELBOW), 1.00, 0.036, 0.037, 0.0),
       ((ELBOW, WRIST), 0.25, 0.040, 0.037, 0.0), ((ELBOW, WRIST), 0.60, 0.031, 0.031, 0.0),
       ((ELBOW, WRIST), 1.00, 0.021, 0.028, 0.0),
       ((WRIST, KNUCKLES), 0.40, 0.020, 0.040, 0.0), ((WRIST, KNUCKLES), 1.00, 0.017, 0.044, 0.0),
       ((KNUCKLES, FINGERTIPS), 0.55, 0.014, 0.041, 0.0), ((KNUCKLES, FINGERTIPS), 1.00, 0.010, 0.033, 0.0)]
# thumb: extruded from the front of the palm by the wrist (palms face the thighs in the A-pose)
THUMB_BASE = WRIST.lerp(KNUCKLES, 0.2) + Vector((0.0, -0.036, 0.0))
THUMB_ROOT, THUMB_TIP = Vector((0.565, -0.040, 1.027)), Vector((0.596, -0.060, 0.957))
THUMB = [((THUMB_ROOT, THUMB_TIP), 0.30, 0.016, 0.014, 0.0), ((THUMB_ROOT, THUMB_TIP), 0.65, 0.013, 0.011, 0.0),
         ((THUMB_ROOT, THUMB_TIP), 1.00, 0.009, 0.008, 0.0)]
HIP, KNEE, ANKLE = Vector((0.092, 0.0, 0.860)), Vector((0.104, -0.010, 0.500)), Vector((0.110, 0.022, 0.085))
LEG = [((HIP, KNEE), 0.17, 0.083, 0.085, 0.0), ((HIP, KNEE), 0.45, 0.073, 0.074, 0.0),
       ((HIP, KNEE), 0.80, 0.060, 0.058, 0.0), ((HIP, KNEE), 1.00, 0.053, 0.055, 0.0),
       ((KNEE, ANKLE), 0.20, 0.052, 0.058, 0.010), ((KNEE, ANKLE), 0.45, 0.049, 0.054, 0.008),
       ((KNEE, ANKLE), 0.75, 0.040, 0.040, 0.0), ((KNEE, ANKLE), 1.00, 0.034, 0.036, 0.0)]
# sole of the foot, heel to toe: (y, half width, z)
SOLE = [(0.068, 0.030, 0.004), (0.010, 0.038, 0.002), (-0.060, 0.044, 0.002), (-0.130, 0.048, 0.003),
        (-0.195, 0.034, 0.010)]
FOOT_X = 0.114
NECK = (1.535, 0.012, 0.054, 0.049)   # z, centre y, half width, half depth (stays inside the head's neck)


def _bm(ob):
    bm = bmesh.from_edit_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return bm


def _select_faces(ob, idx):
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    bm = _bm(ob)
    for f in bm.faces:
        f.select_set(False)
    for i in idx:
        bm.faces[i].select_set(True)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(ob.data)


def _outline(ob, faces):
    fs = set(faces)
    out = set()
    for f in fs:
        for e in f.edges:
            if sum(1 for lf in e.link_faces if lf in fs) == 1:
                out.update(v.index for v in e.verts)
    return out


def _frame(axis):
    """Across (in the XZ plane) and front-back (Y) directions for a limb axis."""
    a = axis.normalized()
    v = Vector((0.0, 1.0, 0.0))
    u = v.cross(a).normalized()
    v = a.cross(u).normalized()
    return a, u, v


def _ring_moves(ob, ids, center, axis, ru, rv):
    mw = ob.matrix_world
    bm = _bm(ob)
    a, u, v = _frame(axis)
    moves = []
    for i in ids:
        d = mw @ bm.verts[i].co - center
        d -= a * d.dot(a)
        th = math.atan2(d.dot(v), d.dot(u))
        moves.append((i, tuple(center + u * (ru * math.cos(th)) + v * (rv * math.sin(th)))))
    return moves


def _extrude_limb(ob, faces, rings):
    """Extrude the face region ring by ring along the limb, grabbing each
    new ring round its joint section."""
    name = ob.name
    _select_faces(ob, faces)
    mw = ob.matrix_world
    for (p0, p1), t, ru, rv, dy in rings:
        center = p0.lerp(p1, t) + Vector((0.0, dy, 0.0))
        bm = _bm(ob)
        sel = [f for f in bm.faces if f.select]
        now = sum((mw @ f.calc_center_median() for f in sel), Vector()) / len(sel)
        K.op(bpy.ops.mesh.extrude_region_move, TRANSFORM_OT_translate={"value": tuple(center - now)})
        bm = _bm(ob)
        sel = [f for f in bm.faces if f.select]
        K.drag(name, _ring_moves(ob, _outline(ob, sel), center, p1 - p0, ru, rv))
        _select_faces(ob, [f.index for f in sel])
    return [f.index for f in _bm(ob).faces if f.select]


def build(name=NAME, collection="CHR_Body"):
    vl = bpy.context.view_layer
    vl.active_layer_collection = vl.layer_collection.children[collection]
    K.mode("OBJECT")
    K.op(bpy.ops.mesh.primitive_cube_add, size=2.0, location=(0.0, 0.0, 0.0))
    ob = vl.objects.active
    ob.name = ob.data.name = name
    K.mode("EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    K.op(bpy.ops.mesh.select_all, action="SELECT")
    K.op(bpy.ops.mesh.subdivide, number_cuts=3)
    # one more loop between each pair round the sides: select the vertical side edges, Subdivide
    K.sel(name, lambda co: True, kind="EDGE")
    bm = _bm(ob)
    for e in bm.edges:
        d = e.verts[1].co - e.verts[0].co
        e.select_set(abs(d.z) > 1e-4 and abs(d.x) < 1e-4 and abs(d.y) < 1e-4)
    bmesh.update_edit_mesh(ob.data)
    K.op(bpy.ops.mesh.subdivide, number_cuts=1)
    K.sel(name, lambda co: co.x < -1e-5)
    K.op(bpy.ops.mesh.delete, type="VERT")
    # Mirror with Clipping before any extrusion, so Extrude leaves the mirror plane open
    m = ob.modifiers.new("Mirror", "MIRROR")
    m.use_clip = m.use_mirror_merge = True
    # remember every vertex by its cube grid position, then grab it onto the body
    bm = _bm(ob)
    unit = {v.index: v.co.copy() for v in bm.verts}
    zs = sorted({round(c.z, 4) for c in unit.values()})
    moves = []
    for i, c in unit.items():
        ring = zs.index(round(c.z, 4))
        on_side = abs(abs(c.x) - 1) < 1e-4 or abs(abs(c.y) - 1) < 1e-4
        key = (round(c.x * 2) / 2, round(c.y * 2) / 2)
        if not on_side and ring == len(zs) - 1:
            moves.append((i, TOP_INNER[key]))
        elif not on_side and ring == 0:
            moves.append((i, BOTTOM_INNER[key]))
        else:
            z, pts = TORSO[ring]
            # perimeter order from the front midline: (0,-1) (0.5,-1) (1,-1) (1,-0.5) ... (0,1)
            if abs(c.y + 1) < 1e-4:
                k = int(round(c.x * 2))
            elif abs(c.y - 1) < 1e-4:
                k = 8 - int(round(c.x * 2))
            else:
                k = 4 + int(round(c.y * 2))
            moves.append((i, (pts[k][0], pts[k][1], z)))
    K.drag(name, moves)
    # regions for the neck, arm and leg, found by their cube grid positions
    bm = _bm(ob)

    def region(test):
        return [f.index for f in bm.faces if test(sum((unit[v.index] for v in f.verts), Vector()) / len(f.verts))]
    top, bottom = zs[-1], zs[0]
    neck = region(lambda c: abs(c.z - top) < 1e-4 and c.x < 0.5 and abs(c.y) < 0.5)
    arm = region(lambda c: abs(c.x - 1) < 1e-4 and abs(c.y) < 0.5 and c.z > zs[-3])
    leg = region(lambda c: abs(c.z - bottom) < 1e-4 and c.x > 0.5)
    counts = {"neck": len(neck), "arm": len(arm), "leg": len(leg)}
    # operators renumber faces, so keep each region by its face centres
    mw = ob.matrix_world
    centers = {key: [(mw @ bm.faces[i].calc_center_median()).copy() for i in idx]
               for key, idx in (("arm", arm), ("leg", leg))}

    def refind(key):
        bm = _bm(ob)
        return [f.index for f in bm.faces
                if any(((mw @ f.calc_center_median()) - c).length < 1e-5 for c in centers[key])]
    # Neck: extrude up and round it
    _select_faces(ob, neck)
    K.op(bpy.ops.mesh.extrude_region_move, TRANSFORM_OT_translate={"value": (0.0, 0.0, NECK[0] - 1.498)})
    bm = _bm(ob)
    sel = [f for f in bm.faces if f.select]
    ring = _outline(ob, sel)
    z, cy, rx, ry = NECK
    moves = []
    for i in ring:
        p = ob.matrix_world @ bm.verts[i].co
        th = math.atan2(p.x / rx, (p.y - cy) / ry)
        moves.append((i, (0.0 if abs(p.x) < 1e-5 else rx * math.sin(th), cy + ry * math.cos(th), z)))
    K.drag(name, moves)
    _select_faces(ob, [f.index for f in sel])
    K.op(bpy.ops.mesh.delete, type="FACE")        # open: the head's neck covers it
    # Arm and hand
    _extrude_limb(ob, refind("arm"), ARM)
    # Thumb: extrude the palm face nearest the thumb base
    bm = _bm(ob)
    base = min(bm.faces, key=lambda f: ((mw @ f.calc_center_median()) - THUMB_BASE).length)
    _extrude_limb(ob, [base.index], THUMB)
    # Leg
    end = _extrude_limb(ob, refind("leg"), LEG)
    # Foot: extrude the ankle cap down to the floor, then grab its outline onto the sole
    _select_faces(ob, end)
    K.op(bpy.ops.mesh.extrude_region_move, TRANSFORM_OT_translate={"value": (0.0, 0.0, -0.075)})
    bm = _bm(ob)
    cap = [f for f in bm.faces if f.select]
    pts = {i: ob.matrix_world @ bm.verts[i].co for i in {v.index for f in cap for v in f.verts}}
    order = sorted(pts, key=lambda i: -pts[i].y)          # heel first, in pairs across the foot
    moves = []
    for n in range(0, len(order) - 1, 2):
        a, b = sorted(order[n:n + 2], key=lambda i: pts[i].x)
        y, hw, z = SOLE[min(len(SOLE) - 1, round((n // 2) * (len(SOLE) - 1) / max(1, len(order) // 2 - 1)))]
        moves += [(a, (FOOT_X - hw, y, z)), (b, (FOOT_X + hw, y, z))]
    K.drag(name, moves)
    K.mode("OBJECT")
    sub = ob.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = sub.render_levels = 2
    K.op(bpy.ops.object.shade_smooth)
    return counts
