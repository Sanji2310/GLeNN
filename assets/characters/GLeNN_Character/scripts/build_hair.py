"""Stylised sculpted hair built as separate mesh masses on the scalp.

The hair has three layers:
  HAIR_Cap     a thin shell over the scalp that follows the hairline. It is
               short on the faded sides and back and fuller on top.
  HAIR_Clumps  tapered clump meshes that sweep up and back into a textured
               quiff. Every clump is a quad tube with UVs that run root to tip
               (u across, v along), so the material can grade colour and
               place an anisotropic highlight.
  HAIR_Fringe  a few smaller loose clumps at the front for personality.

Clumps are described in scalp coordinates: azimuth phi (0 = front, + toward
the character's left), elevation theta (90 = crown) around the head centre,
and height above the scalp. Each path point is projected onto the head mesh
with a ray cast, so the hair hugs the skull and never cuts into it.

Run inside Blender 5.2.2 after the head exists:
    import build_hair; build_hair.run(head_obj, center)
"""

import math

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import glenn_common as G
from shader_dsl import NB

OWNER = "hair"


def direction(phi, theta):
    p, t = math.radians(phi), math.radians(theta)
    return Vector((math.sin(p) * math.cos(t), -math.cos(p) * math.cos(t), math.sin(t)))


class Scalp:
    """Ray-cast projector onto the head surface (object space of the head)."""

    def __init__(self, head_obj, center):
        dg = bpy.context.evaluated_depsgraph_get()
        self.bvh = BVHTree.FromObject(head_obj, dg)
        self.center = Vector(center)

    def point(self, phi, theta, height=0.0):
        d = direction(phi, theta)
        origin = self.center + d * 0.5
        hit, normal, _i, _dist = self.bvh.ray_cast(origin, -d)
        if hit is None:
            hit, normal = self.center + d * 0.1, d
        return hit + normal * height, normal


def catmull(points, samples):
    """Sample a Catmull-Rom spline through points."""
    pts = [points[0] + (points[0] - points[1])] + list(points) + [points[-1] + (points[-1] - points[-2])]
    out = []
    segs = len(points) - 1
    for s in range(samples):
        t = s / (samples - 1) * segs
        i = min(int(t), segs - 1)
        u = t - i
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u))
    return out


def clump_mesh(bm, uv, path, normals, width, thickness, section=8, twist=0.0, flat_tip=False):
    """Append a tapered tube along path. normals give the 'up' of the flat
    cross-section (usually the scalp normal) at each sample."""
    n = len(path)
    rings = []
    for i, p in enumerate(path):
        t = i / (n - 1)
        tan = (path[min(i + 1, n - 1)] - path[max(i - 1, 0)]).normalized()
        up = (normals[i] - tan * normals[i].dot(tan)).normalized()
        side = tan.cross(up).normalized()
        # width profile: full at the root, swelling slightly, then tapering
        w = width * (0.85 + 0.35 * math.sin(math.pi * min(t * 1.3, 1.0))) * (1.0 - t) ** 0.85
        h = thickness * (1.0 - t) ** 0.7
        if flat_tip:
            w, h = max(w, width * 0.12), max(h, thickness * 0.15)
        ang0 = math.radians(twist) * t
        ring = []
        for k in range(section):
            a = 2 * math.pi * k / section + ang0
            # lens-shaped section: flat underside, rounded top
            ca, sa = math.cos(a), math.sin(a)
            lift = h * (sa if sa > 0 else 0.35 * sa)
            ring.append(bm.verts.new(p + side * (w * ca) + up * lift))
        rings.append(ring)
    tip = bm.verts.new(path[-1] + (path[-1] - path[-2]).normalized() * 0.002)
    for i in range(n - 1):
        for k in range(section):
            k2 = (k + 1) % section
            f = bm.faces.new((rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]))
            v0, v1 = i / (n - 1), (i + 1) / (n - 1)
            for loop, co in zip(f.loops, ((k / section, v0), ((k + 1) / section, v0),
                                          ((k + 1) / section, v1), (k / section, v1))):
                loop[uv].uv = co
    for k in range(section):
        k2 = (k + 1) % section
        f = bm.faces.new((rings[-1][k], rings[-1][k2], tip))
        for loop, co in zip(f.loops, ((k / section, 1.0), ((k + 1) / section, 1.0), ((k + 0.5) / section, 1.0))):
            loop[uv].uv = co
    # close the root (it sits under the cap, but keeps the mesh watertight)
    f = bm.faces.new(list(reversed(rings[0])))
    for loop in f.loops:
        loop[uv].uv = (0.5, 0.0)


def build_clumps(name, scalp, specs, samples=14):
    """specs: list of dicts with keys
         path: [(phi, theta, height), ...] scalp-space control points
         width, thickness, twist (deg)"""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    for spec in specs:
        ctrl, ctrl_n = [], []
        for phi, theta, h in spec["path"]:
            p, nrm = scalp.point(phi, theta, h)
            ctrl.append(p)
            ctrl_n.append(nrm)
        path = catmull(ctrl, samples)
        # normals: interpolate control normals along the path
        normals = []
        for s in range(samples):
            t = s / (samples - 1) * (len(ctrl_n) - 1)
            i = min(int(t), len(ctrl_n) - 2)
            u = t - i
            normals.append((ctrl_n[i] * (1 - u) + ctrl_n[i + 1] * u).normalized())
        clump_mesh(bm, uv, path, normals, spec["width"], spec["thickness"],
                   twist=spec.get("twist", 0.0), flat_tip=spec.get("flat_tip", False))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 2
    return obj


def build_cap(name, scalp, hairline, thickness, cols=64, rows=22):
    """Shell over the scalp. hairline(phi) gives the lowest elevation the
    hair reaches at each azimuth; thickness(phi, theta) gives the height."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    grid = []
    for r in range(rows + 1):
        row = []
        for c in range(cols):
            phi = -180.0 + 360.0 * c / cols
            lo = hairline(phi)
            theta = lo + (88.0 - lo) * (r / rows) ** 0.9
            h = thickness(phi, theta) * min(1.0, r / 2.0 + 0.15)
            p, _n = scalp.point(phi, theta, h)
            row.append(bm.verts.new(p))
        grid.append(row)
    for r in range(rows):
        for c in range(cols):
            c2 = (c + 1) % cols
            f = bm.faces.new((grid[r][c], grid[r][c2], grid[r + 1][c2], grid[r + 1][c]))
            for loop, (cc, rr) in zip(f.loops, ((c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1))):
                loop[uv].uv = (cc / cols, rr / rows)
    # close the crown with a fan around a centre vertex
    top, _n = scalp.point(0.0, 89.5, thickness(0.0, 90.0))
    tv = bm.verts.new(top)
    for c in range(cols):
        f = bm.faces.new((grid[rows][c], grid[rows][(c + 1) % cols], tv))
        for loop, co in zip(f.loops, ((c / cols, 1.0), ((c + 1) / cols, 1.0), ((c + 0.5) / cols, 1.0))):
            loop[uv].uv = co
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    sol = obj.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.0015
    sol.offset = -1.0
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 2
    return obj


def hair_material():
    """Dark brown hair: darker roots, warmer mid-lengths and a soft
    anisotropic sheen along each clump."""
    mat = bpy.data.materials.get("MAT_Hair") or bpy.data.materials.new("MAT_Hair")
    nb = NB(mat)
    nb.clear()
    u, v, _ = nb.xyz(nb.texcoord("UV"))
    grad = nb.smoothstep(0.0, 0.9, v)
    base = nb.mix(grad, (0.020, 0.011, 0.007), (0.075, 0.040, 0.022))
    streak = nb.noise(nb.texcoord("Object"), 140.0, detail=2.0)
    base = nb.mix(nb.smoothstep(0.45, 0.75, streak) * 0.35, base, (0.11, 0.065, 0.038))
    bsdf = nb.principled(base=base, rough=0.42, aniso=0.6, spec=0.45, coat=0.0)
    nb.output(bsdf)
    return mat
