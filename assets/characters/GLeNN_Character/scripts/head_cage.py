"""Box-modeled head cage for the racer, driven through Blender's own tools.

The steps are the ones a modeler does by hand:
    1. Add > Mesh > Cube, Subdivide (7 cuts), Transform > To Sphere.
    2. Scale to head size, Apply Scale.
    3. Delete the -X half, add Mirror (Clipping) and Subdivision Surface.
    4. Shape: drag each cage vertex (G) so that the head matches a set of
       horizontal cross-sections drawn over the top view, from the
       under-jaw up to the crown. Each vertex keeps its height and slides
       along its own direction from the head's centre line.

Features (eyes, nose, mouth, ears, neck) are added afterwards with Inset,
Extrude and Loop Cut, and refined with Sculpt Mode brushes.

    import head_cage
    head_cage.build()        # new cage in collection CHR_Head
    head_cage.fit()          # drag the cage onto SECTIONS
"""

import math

import bmesh
import bpy
from mathutils import Vector

import sculpt_kit as K

CENTER = Vector((0.0, 0.008, 1.677))   # head centre (m); the face looks along -Y
HALF = Vector((0.076, 0.103, 0.118))   # half width, depth and height of the start shape
YC = CENTER.y
CROWN = 1.7955

# Half cross-sections (x >= 0), front midline to back midline, as (x, y).
SECTIONS = {
    1.553: [(0, -0.078), (0.012, -0.075), (0.022, -0.064), (0.030, -0.045), (0.035, -0.020), (0.038, 0.005),
            (0.038, 0.025), (0.032, 0.040), (0.018, 0.048), (0, 0.050)],
    1.565: [(0, -0.097), (0.012, -0.0935), (0.022, -0.083), (0.030, -0.068), (0.037, -0.050), (0.042, -0.028),
            (0.045, -0.005), (0.046, 0.015), (0.044, 0.035), (0.035, 0.050), (0.02, 0.056), (0, 0.058)],
    1.574: [(0, -0.1035), (0.010, -0.102), (0.020, -0.095), (0.028, -0.084), (0.037, -0.067), (0.045, -0.046),
            (0.051, -0.022), (0.055, 0.0), (0.056, 0.02), (0.051, 0.040), (0.035, 0.056), (0, 0.062)],
    1.590: [(0, -0.093), (0.015, -0.091), (0.026, -0.085), (0.037, -0.073), (0.047, -0.056), (0.055, -0.036),
            (0.061, -0.014), (0.064, 0.008), (0.065, 0.026), (0.060, 0.046), (0.040, 0.062), (0, 0.066)],
    1.606: [(0, -0.102), (0.015, -0.099), (0.028, -0.091), (0.038, -0.079), (0.048, -0.062), (0.056, -0.040),
            (0.062, -0.016), (0.065, 0.006), (0.066, 0.026), (0.061, 0.047), (0.046, 0.064), (0.024, 0.072),
            (0, 0.074)],
    1.626: [(0, -0.104), (0.012, -0.101), (0.025, -0.093), (0.040, -0.082), (0.050, -0.066), (0.058, -0.046),
            (0.064, -0.021), (0.067, 0.002), (0.066, 0.026), (0.061, 0.050), (0.049, 0.070), (0.027, 0.083),
            (0, 0.086)],
    1.650: [(0, -0.104), (0.012, -0.099), (0.025, -0.090), (0.040, -0.082), (0.057, -0.069), (0.068, -0.050),
            (0.072, -0.026), (0.072, 0.0), (0.070, 0.030), (0.065, 0.060), (0.052, 0.086), (0.028, 0.099),
            (0, 0.102)],
    1.680: [(0, -0.095), (0.012, -0.089), (0.025, -0.083), (0.035, -0.081), (0.048, -0.076), (0.062, -0.062),
            (0.070, -0.040), (0.074, -0.012), (0.075, 0.02), (0.073, 0.052), (0.065, 0.082), (0.048, 0.102),
            (0.024, 0.111), (0, 0.113)],
    1.704: [(0, -0.100), (0.020, -0.099), (0.035, -0.095), (0.050, -0.086), (0.062, -0.068), (0.071, -0.040),
            (0.077, -0.005), (0.077, 0.035), (0.070, 0.072), (0.051, 0.098), (0.025, 0.110), (0, 0.113)],
    1.730: [(0, -0.094), (0.020, -0.093), (0.036, -0.088), (0.050, -0.078), (0.062, -0.060), (0.070, -0.034),
            (0.074, 0.0), (0.073, 0.035), (0.066, 0.068), (0.049, 0.092), (0.024, 0.104), (0, 0.107)],
    1.749: [(0, -0.086), (0.020, -0.085), (0.036, -0.079), (0.049, -0.068), (0.058, -0.050), (0.064, -0.025),
            (0.066, 0.005), (0.064, 0.040), (0.056, 0.068), (0.042, 0.086), (0.021, 0.094), (0, 0.096)],
    1.774: [(0, -0.063), (0.018, -0.061), (0.032, -0.054), (0.042, -0.042), (0.048, -0.025), (0.050, 0.0),
            (0.049, 0.03), (0.043, 0.052), (0.031, 0.067), (0.016, 0.074), (0, 0.075)],
    1.793: [(0, -0.020), (0.012, -0.016), (0.018, -0.006), (0.020, 0.008), (0.018, 0.022), (0.012, 0.032),
            (0, 0.036)],
}


def build(name="CHR_Head", collection="CHR_Head"):
    vl = bpy.context.view_layer
    vl.active_layer_collection = vl.layer_collection.children[collection]
    K.mode("OBJECT")
    K.op(bpy.ops.mesh.primitive_cube_add, size=2.0, location=tuple(CENTER))
    ob = vl.objects.active
    ob.name = ob.data.name = name
    K.mode("EDIT")
    K.op(bpy.ops.mesh.select_all, action="SELECT")
    K.op(bpy.ops.mesh.subdivide, number_cuts=7)
    K.op(bpy.ops.transform.tosphere, value=1.0)
    K.mode("OBJECT")
    r = sum(v.co.length for v in ob.data.vertices) / len(ob.data.vertices)   # To Sphere keeps the mean radius
    ob.scale = (HALF.x / r, HALF.y / r, HALF.z / r)
    K.op(bpy.ops.object.transform_apply, location=False, rotation=False, scale=True)
    K.mode("EDIT")
    K.sel(name, lambda co: co.x < -1e-5)
    K.op(bpy.ops.mesh.delete, type="VERT")
    K.mode("OBJECT")
    mirror = ob.modifiers.new("Mirror", "MIRROR")
    mirror.use_clip = mirror.use_mirror_merge = True
    sub = ob.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = sub.render_levels = 2
    K.op(bpy.ops.object.shade_smooth)
    return ob


def _ray(poly, th):
    """Distance from the centre line to the section outline at angle th
    (0 = straight ahead, pi = straight back)."""
    dx, dy = math.sin(th), -math.cos(th)
    best = None
    for (x0, y0), (x1, y1) in zip(poly, poly[1:]):
        ex, ey = x1 - x0, y1 - y0
        den = -dx * ey + dy * ex
        if abs(den) < 1e-12:
            continue
        rx, ry = x0, y0 - YC
        t = (-rx * ey + ry * ex) / den
        s = (dx * ry - dy * rx) / den
        if -1e-6 <= s <= 1 + 1e-6 and t > 0:
            best = t if best is None else min(best, t)
    return best


def _radius(th, z):
    zs = sorted(SECTIONS)
    if z <= zs[0]:
        return _ray(SECTIONS[zs[0]], th)
    if z >= zs[-1]:
        return _ray(SECTIONS[zs[-1]], th) * max(0.0, (CROWN - z) / (CROWN - zs[-1]))
    for z0, z1 in zip(zs, zs[1:]):
        if z0 <= z <= z1:
            f = (z - z0) / (z1 - z0)
            return _ray(SECTIONS[z0], th) * (1 - f) + _ray(SECTIONS[z1], th) * f


def fit(name="CHR_Head"):
    """Drag every cage vertex onto the sections (G, one vertex at a time)."""
    ob = bpy.data.objects[name]
    mw = ob.matrix_world
    K.mode("EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bm = bmesh.from_edit_mesh(ob.data)
    moves = []
    for v in bm.verts:
        w = mw @ v.co
        if abs(w.x) < 1e-6 and abs(w.y - YC) < 1e-6:
            continue
        th = math.atan2(abs(w.x), -(w.y - YC))
        r = _radius(th, w.z)
        tgt = Vector((0.0 if abs(w.x) < 1e-6 else math.sin(th) * r, YC - math.cos(th) * r, w.z))
        moves.append((v.index, tgt - w))
    for i, d in moves:
        if d.length < 1e-5:
            continue
        bm = bmesh.from_edit_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        for v in bm.verts:
            v.select_set(False)
        bm.verts[i].select_set(True)
        bmesh.update_edit_mesh(ob.data)
        K.op(bpy.ops.transform.translate, value=tuple(d), use_proportional_edit=False)
    K.mode("OBJECT")
    return {"moved": len(moves), "max_move": round(max(d.length for _, d in moves), 4)}
