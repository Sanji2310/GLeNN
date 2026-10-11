"""Face features on the head cage, cut in with Blender's modeling tools.

Each step is a short sequence of the operators a modeler would use:
Select faces, Inset (I), Delete Faces (X), Grab (G) vertices into place,
Extrude (E), Scale (S), Loop Cut (Ctrl+R). Targets are world-space
positions in metres; the face looks along -Y and the Mirror modifier
supplies the -X half.

    import head_face
    head_face.eye()
"""

import math

import bmesh
import bpy
from mathutils import Vector

import sculpt_kit as K

HEAD = "CHR_Head"
EYE_CENTER = Vector((0.033, -0.071, 1.684))
EYE_RADIUS = 0.0125

# Eye rings, by direction around the eye in the front view (degrees,
# 0 = outer corner, 90 = top, 180 = inner corner). The opening sits on the
# eyeball and its corners tuck in past the eyeball's silhouette; the lid
# fold is the upper-lid crease and lower-lid edge; the rim is the orbit
# (brow above, cheek below, nose side, temple side).
EYE_OPENING = {
    180: (0.0195, -0.0735, 1.6810), 135: (0.0255, -0.0807, 1.6870), 90: (0.0335, -0.0818, 1.6893),
    45: (0.0415, -0.0790, 1.6885), 0: (0.0475, -0.0690, 1.6855), -45: (0.0420, -0.0792, 1.6800),
    -90: (0.0345, -0.0819, 1.6775), -135: (0.0260, -0.0802, 1.6785),
}
EYE_FOLD = {
    180: (0.0165, -0.0800, 1.6835), 135: (0.0240, -0.0840, 1.6935), 90: (0.0340, -0.0846, 1.6962),
    45: (0.0435, -0.0818, 1.6950), 0: (0.0510, -0.0715, 1.6875), -45: (0.0440, -0.0814, 1.6760),
    -90: (0.0345, -0.0842, 1.6730), -135: (0.0250, -0.0830, 1.6745),
}
EYE_RIM = {
    180: (0.0140, -0.0865, 1.6830), 135: (0.0170, -0.0905, 1.7040), 90: (0.0340, -0.0945, 1.7085),
    45: (0.0515, -0.0880, 1.7055), 0: (0.0570, -0.0715, 1.6880), -45: (0.0505, -0.0780, 1.6670),
    -90: (0.0345, -0.0855, 1.6630), -135: (0.0180, -0.0870, 1.6650),
}
LID_CREASE = 0.8   # Shift+E on the lid edge keeps it crisp under Subdivision Surface


def _bm(ob):
    bm = bmesh.from_edit_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return bm


def _select_faces(ob, faces):
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    bm = _bm(ob)
    for f in bm.faces:
        f.select_set(False)
    for f in faces:
        f.select_set(True)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(ob.data)


def _boundary_verts(faces):
    fs = set(faces)
    out = set()
    for f in fs:
        for e in f.edges:
            if sum(1 for lf in e.link_faces if lf in fs) == 1:
                out.update(v.index for v in e.verts)
    return out


def _by_angle(ob, ids, center):
    """Label the 8 vertices of a ring round a 2 x 2 patch by direction from
    its centre (front view): 0 outer, 45, 90 top, ... 180 inner, ... -45.
    They are taken in angular order, so a stretched ring still gets all 8."""
    mw = ob.matrix_world
    bm = _bm(ob)
    ang = {i: math.degrees(math.atan2((mw @ bm.verts[i].co).z - center.z, (mw @ bm.verts[i].co).x - center.x))
           for i in ids}
    order = sorted(ids, key=lambda i: ang[i] % 360.0)
    start = min(range(len(order)), key=lambda n: abs(ang[order[n]]))
    order = order[start:] + order[:start]
    return dict(zip((0, 45, 90, 135, 180, -135, -90, -45), order))


def _positions(ob, ids):
    mw = ob.matrix_world
    bm = _bm(ob)
    return {k: (mw @ bm.verts[i].co).copy() for k, i in ids.items()}


def _indices(name, points):
    return {k: K.nearest(name, p) for k, p in points.items()}


def eye(name=HEAD):
    """Socket and eyelids: inset the four faces around the eye twice, open
    the middle, grab the three rings into orbit rim, lid fold and almond
    opening, crease the lid edge, then extrude it back into the socket."""
    ob = K.focus(name, mode="EDIT")
    mw = ob.matrix_world
    vc = K.nearest(name, (0.033, -0.082, 1.677))
    bm = _bm(ob)
    center = mw @ bm.verts[vc].co
    patch = list(bm.verts[vc].link_faces)
    rim = _positions(ob, _by_angle(ob, _boundary_verts(patch), center))   # operators renumber vertices
    _select_faces(ob, patch)
    K.op(bpy.ops.mesh.inset, thickness=0.0035, depth=0.0, use_even_offset=True)
    bm = _bm(ob)
    fold = _positions(ob, _by_angle(ob, _boundary_verts([f for f in bm.faces if f.select]), center))
    K.op(bpy.ops.mesh.inset, thickness=0.0035, depth=0.0, use_even_offset=True)
    bm = _bm(ob)
    opening = _positions(ob, _by_angle(ob, _boundary_verts([f for f in bm.faces if f.select]), center))
    K.op(bpy.ops.mesh.delete, type="FACE")
    rim, fold, opening = _indices(name, rim), _indices(name, fold), _indices(name, opening)
    K.drag(name, [(rim[k], EYE_RIM[k]) for k in rim])
    K.drag(name, [(fold[k], EYE_FOLD[k]) for k in fold])
    K.drag(name, [(opening[k], EYE_OPENING[k]) for k in opening])
    _select_verts(ob, opening.values())
    K.op(bpy.ops.transform.edge_crease, value=LID_CREASE)
    K.op(bpy.ops.mesh.extrude_edges_move, TRANSFORM_OT_translate={"value": (0.0, 0.0035, 0.0)})
    K.op(bpy.ops.transform.resize, value=(0.86, 0.86, 0.86))
    K.mode("OBJECT")
    return {"rim": len(rim), "fold": len(fold), "opening": len(opening)}


def _select_verts(ob, ids):
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bm = _bm(ob)
    for v in bm.verts:
        v.select_set(False)
    for i in ids:
        bm.verts[i].select_set(True)
    bm.select_flush(True)
    bmesh.update_edit_mesh(ob.data)


def eyeball(name="CHR_Eyes", head=HEAD):
    """UV Sphere with its pole facing forward, mirrored across the head."""
    vl = bpy.context.view_layer
    vl.active_layer_collection = vl.layer_collection.children["CHR_Head"]
    K.mode("OBJECT")
    K.op(bpy.ops.mesh.primitive_uv_sphere_add, segments=32, ring_count=16, radius=EYE_RADIUS,
         location=tuple(EYE_CENTER), rotation=(math.radians(90), 0.0, 0.0))
    ob = vl.objects.active
    ob.name = ob.data.name = name
    m = ob.modifiers.new("Mirror", "MIRROR")
    m.mirror_object = bpy.data.objects[head]
    K.op(bpy.ops.object.shade_smooth)
    return ob


# Nose: the two midline faces between the eyes and the upper lip are
# extruded forward, then each new vertex is grabbed onto the nose.
# Keys are (row, column): rows top/mid/bot, column 0 on the midline, 1 beside it.
NOSE_TIP = {
    ("top", 0): (0.0, -0.1000, 1.6690), ("top", 1): (0.0070, -0.0955, 1.6690),
    ("mid", 0): (0.0, -0.1240, 1.6380), ("mid", 1): (0.0118, -0.1075, 1.6400),
    ("bot", 0): (0.0, -0.1085, 1.6270), ("bot", 1): (0.0178, -0.0960, 1.6285),
}
NOSE_BASE = {
    ("top", 0): (0.0, -0.0950, 1.6840), ("mid", 1): (0.0228, -0.0878, 1.6470),
    ("bot", 0): (0.0, -0.1000, 1.6230), ("bot", 1): (0.0220, -0.0905, 1.6240),
}


def _rows(ob, ids):
    """Label vertices (top/mid/bot, column): column 0 on the midline, 1 beside
    it, rows by height within each column."""
    mw = ob.matrix_world
    bm = _bm(ob)
    pts = {i: mw @ bm.verts[i].co for i in ids}
    out = {}
    for col in (0, 1):
        ids_c = sorted((i for i, p in pts.items() if (abs(p.x) < 1e-5) == (col == 0)), key=lambda i: -pts[i].z)
        for row, i in zip(("top", "mid", "bot"), ids_c):
            out[(row, col)] = i
    return out


def nose(name=HEAD):
    ob = K.focus(name, mode="EDIT")
    mw = ob.matrix_world
    bm = _bm(ob)
    faces = [f for f in bm.faces
             if any(abs((mw @ v.co).x) < 1e-5 for v in f.verts)
             and 1.62 < (mw @ f.calc_center_median()).z < 1.68 and (mw @ f.calc_center_median()).y < -0.05]
    base = _positions(ob, _rows(ob, {v.index for f in faces for v in f.verts}))
    _select_faces(ob, faces)
    K.op(bpy.ops.mesh.extrude_region_move, TRANSFORM_OT_translate={"value": (0.0, -0.006, 0.0)})
    bm = _bm(ob)
    tip = _rows(ob, {v.index for v in bm.verts if v.select})
    base = _indices(name, base)
    K.drag(name, [(tip[k], NOSE_TIP[k]) for k in tip])
    K.drag(name, [(base[k], NOSE_BASE[k]) for k in NOSE_BASE if k in base])
    K.mode("OBJECT")
    return {"faces": len(faces), "tip": sorted(tip), "base": sorted(base)}


# Mouth: the 2 x 2 faces beside the midline between the nose and the chin
# are inset twice (no inset along the mirror edge), the middle is opened
# into the mouth slit, and the three rings are grabbed into place. Each ring
# runs from the top of the midline round the corner to the bottom of it.
MOUTH_OUTER = [(0.0, -0.1000, 1.6230), (0.0220, -0.0905, 1.6240), (0.0345, -0.0845, 1.6215), (0.0370, -0.0805, 1.6050),
               (0.0330, -0.0820, 1.5920), (0.0165, -0.0900, 1.5880), (0.0, -0.0925, 1.5870)]
MOUTH_LIP = [(0.0, -0.1022, 1.6140), (0.0110, -0.0995, 1.6150), (0.0225, -0.0910, 1.6090), (0.0275, -0.0860, 1.6050),
             (0.0225, -0.0905, 1.6000), (0.0115, -0.0978, 1.5955), (0.0, -0.0995, 1.5945)]
MOUTH_LINE = [(0.0, -0.0978, 1.6058), (0.0115, -0.0952, 1.6060), (0.0205, -0.0898, 1.6055), (0.0245, -0.0866, 1.6050),
              (0.0205, -0.0896, 1.6046), (0.0115, -0.0948, 1.6044), (0.0, -0.0972, 1.6044)]
MOUTH_CENTER = Vector((0.0, -0.09, 1.606))
MOUTH_CREASE = 0.7


def _ring_order(ob, ids, center=MOUTH_CENTER):
    """Ring vertices ordered from the top of the midline, round the corner,
    to the bottom of the midline (front view). Of the vertices on the
    midline only the top and bottom ones belong to the ring."""
    mw = ob.matrix_world
    bm = _bm(ob)
    pts = {i: mw @ bm.verts[i].co for i in ids}
    mid = sorted((i for i in ids if abs(pts[i].x) < 1e-5), key=lambda i: pts[i].z)
    keep = [i for i in ids if abs(pts[i].x) >= 1e-5] + ([mid[0], mid[-1]] if mid else [])
    return sorted(keep, key=lambda i: -math.atan2(pts[i].z - center.z, pts[i].x - center.x))


def _region_boundary(faces):
    """Boundary vertices of a face region, including the open mirror edge."""
    fs = set(faces)
    out = set()
    for f in fs:
        for e in f.edges:
            if sum(1 for lf in e.link_faces if lf in fs) == 1 or e.is_boundary:
                out.update(v.index for v in e.verts)
    return out


def mouth(name=HEAD):
    ob = K.focus(name, mode="EDIT")
    mw = ob.matrix_world
    bm = _bm(ob)
    patch = [f for f in bm.faces if (lambda c: c.x < 0.03 and 1.596 < c.z < 1.622 and c.y < -0.07)(mw @ f.calc_center_median())]
    outer = [(mw @ bm.verts[i].co).copy() for i in _ring_order(ob, _region_boundary(patch))]
    _select_faces(ob, patch)
    K.op(bpy.ops.mesh.inset, thickness=0.004, depth=0.0, use_even_offset=True, use_boundary=False)
    bm = _bm(ob)
    lip = [(mw @ bm.verts[i].co).copy() for i in _ring_order(ob, _region_boundary([f for f in bm.faces if f.select]))]
    K.op(bpy.ops.mesh.inset, thickness=0.004, depth=0.0, use_even_offset=True, use_boundary=False)
    bm = _bm(ob)
    sel = [f for f in bm.faces if f.select]
    line = [(mw @ bm.verts[i].co).copy() for i in _ring_order(ob, _region_boundary(sel))]
    counts = (len(patch), len(outer), len(lip), len(line))
    K.op(bpy.ops.mesh.delete, type="FACE")
    # the middle row of the opened region is gone; what is left of the line
    # ring is its top and bottom edges plus the corner
    line = [p for p in line if any((p - q).length < 1e-6 for q in [mw @ v.co for v in _bm(ob).verts])]
    for ring, target in ((outer, MOUTH_OUTER), (lip, MOUTH_LIP), (line, MOUTH_LINE)):
        ids = [K.nearest(name, p) for p in ring]
        if len(ids) == len(target):
            K.drag(name, list(zip(ids, target)))
    ids = [K.nearest(name, p) for p in line]
    _select_verts(ob, ids)
    K.op(bpy.ops.transform.edge_crease, value=MOUTH_CREASE)
    K.op(bpy.ops.mesh.extrude_edges_move, TRANSFORM_OT_translate={"value": (0.0, 0.008, 0.0)})
    K.op(bpy.ops.transform.resize, value=(0.85, 1.0, 0.6))
    K.mode("OBJECT")
    return {"patch, outer, lip, line": counts, "line kept": len(line)}


NECK_TOP = (0.010, 0.048, 0.046)      # centre y, half width, half depth of the ring tucked under the jaw
NECK_RINGS = [   # (height, centre y, half width, half depth) of each extruded ring
    (1.525, 0.012, 0.060, 0.052),
    (1.440, 0.008, 0.064, 0.055),
]


def _smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def _fit_ring(name, ids, z, cy, rx, ry):
    """Grab a ring of vertices onto an ellipse, each along its own direction
    from the ring's centre line. z is a height or a function of the point."""
    ob = bpy.data.objects[name]
    mw = ob.matrix_world
    bm = _bm(ob)
    moves = []
    for i in ids:
        p = mw @ bm.verts[i].co
        if abs(p.x) < 1e-5:
            q = Vector((0.0, cy + (ry if p.y > cy else -ry), 0.0))
        else:
            a = math.atan2(p.x / rx, (p.y - cy) / ry)
            q = Vector((rx * math.sin(a), cy + ry * math.cos(a), 0.0))
        q.z = z(q) if callable(z) else z
        moves.append((i, tuple(q)))
    K.drag(name, moves)


def _neck_top_z(q):
    """The neck meets the head low under the chin and higher at the nape."""
    return 1.553 + 0.022 * _smooth((q.y + 0.03) / 0.08)


def _outline(ob, faces):
    """The outline of a face region without its edges on the mirror plane,
    so a region touching the midline gives an open ring of vertices."""
    mw = ob.matrix_world
    fs = set(faces)
    out = set()
    for f in fs:
        for e in f.edges:
            if all(abs((mw @ v.co).x) < 1e-5 for v in e.verts):
                continue
            if sum(1 for lf in e.link_faces if lf in fs) == 1:
                out.update(v.index for v in e.verts)
    return out


def neck(name=HEAD):
    """Under the head (behind the chin): Inset once so the outer ring stays as
    the jaw line, lift the jaw line towards the ears, then extrude the inner
    region down into the neck in two steps and round each ring."""
    ob = K.focus(name, mode="EDIT")
    mw = ob.matrix_world
    rot = mw.to_3x3()
    bm = _bm(ob)
    faces = [f for f in bm.faces if (rot @ f.normal).z < -0.5
             and (mw @ f.calc_center_median()).y > -0.045 and (mw @ f.calc_center_median()).z < 1.572]
    jaw = [(mw @ bm.verts[i].co).copy() for i in _outline(ob, faces)]
    _select_faces(ob, faces)
    K.op(bpy.ops.mesh.inset, thickness=0.009, depth=0.0, use_even_offset=True, use_boundary=False)
    bm = _bm(ob)
    sel = [f.index for f in bm.faces if f.select]
    # jaw line: rises from under the chin to the jaw angle below the ear
    jaw_ids = [K.nearest(name, p) for p in jaw]
    moves = []
    for i, p in zip(jaw_ids, jaw):
        side = _smooth((abs(p.x) - 0.012) / 0.025)
        nape = _smooth((p.y - 0.02) / 0.025)
        lift = 0.030 * _smooth((p.y + 0.035) / 0.065) * side * (1.0 - 0.6 * nape) + 0.010 * nape * (1 - side)
        moves.append((i, (p.x, p.y, p.z + lift)))
    K.drag(name, moves)
    bm = _bm(ob)
    top_ring = list(_outline(ob, [bm.faces[i] for i in sel]))
    _fit_ring(name, top_ring, _neck_top_z, *NECK_TOP)
    _select_faces(ob, [_bm(ob).faces[i] for i in sel])
    top = 1.56
    for z, cy, rx, ry in NECK_RINGS:
        K.op(bpy.ops.mesh.extrude_region_move, TRANSFORM_OT_translate={"value": (0.0, 0.0, z - top)})
        top = z
        bm = _bm(ob)
        sel = [f.index for f in bm.faces if f.select]
        _fit_ring(name, list(_outline(ob, [bm.faces[i] for i in sel])), z, cy, rx, ry)
        _select_faces(ob, [_bm(ob).faces[i] for i in sel])
    # the end cap goes: the neck runs on under the collar
    K.op(bpy.ops.mesh.delete, type="FACE")
    K.mode("OBJECT")
    return {"faces": len(faces), "jaw": len(jaw)}


# Ear: four faces on the side of the head, inset for the ear's base, extruded
# out, the new rim grabbed into the ear outline (side view, by direction:
# 0 = back, 90 = top, 180 = front at the tragus, -90 = lobe), then the bowl
# inset and pushed in.
EAR_RIM = {
    0: (0.0875, 0.0310, 1.6680), 45: (0.0860, 0.0265, 1.6870), 90: (0.0820, 0.0120, 1.6960),
    135: (0.0785, -0.0020, 1.6860), 180: (0.0770, -0.0060, 1.6600), -135: (0.0785, -0.0020, 1.6400),
    -90: (0.0790, 0.0080, 1.6270), -45: (0.0850, 0.0240, 1.6450),
}


def _by_angle_side(ob, ids, center):
    mw = ob.matrix_world
    bm = _bm(ob)
    ang = {i: math.degrees(math.atan2((mw @ bm.verts[i].co).z - center.z, (mw @ bm.verts[i].co).y - center.y))
           for i in ids}
    order = sorted(ids, key=lambda i: ang[i] % 360.0)
    start = min(range(len(order)), key=lambda n: abs(ang[order[n]]))
    order = order[start:] + order[:start]
    return dict(zip((0, 45, 90, 135, 180, -135, -90, -45), order))


def ear(name=HEAD):
    ob = K.focus(name, mode="EDIT")
    mw = ob.matrix_world
    vc = K.nearest(name, (0.075, 0.012, 1.662))
    bm = _bm(ob)
    center = (mw @ bm.verts[vc].co).copy()
    patch = list(bm.verts[vc].link_faces)
    _select_faces(ob, patch)
    K.op(bpy.ops.mesh.inset, thickness=0.002, depth=0.0, use_even_offset=True)
    K.op(bpy.ops.mesh.extrude_region_move, TRANSFORM_OT_translate={"value": (0.006, 0.0, 0.0)})
    bm = _bm(ob)
    sel = [f for f in bm.faces if f.select]
    rim = _by_angle_side(ob, _boundary_verts(sel), center)
    K.drag(name, [(rim[k], EAR_RIM[k]) for k in rim])
    # the bowl (concha): inset the ear's face and push it in
    K.op(bpy.ops.mesh.inset, thickness=0.0035, depth=0.0, use_even_offset=True)
    K.op(bpy.ops.transform.translate, value=(-0.0075, 0.002, 0.0))
    K.op(bpy.ops.transform.resize, value=(1.0, 0.85, 0.9))
    K.mode("OBJECT")
    return {"patch": len(patch), "rim": len(rim)}
