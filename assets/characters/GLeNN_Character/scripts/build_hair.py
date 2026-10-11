"""The GLeNN driver's hair and the strand tools shared with the brows.

HAIR_Sculpt is meshed from design_hair.py: a faded cap plus tapered clumps
that rise into a short swept-back quiff. Each vertex stores GLN_HairMask:
  R  height above the scalp (0 at the roots, 1 at the outer surface)
  G  a random value per clump, for colour variation between hair masses
  B  how close the hair lies to the skin (the faded sides show skin through)

catmull() and clump_mesh() build small strand meshes along a curve; the
eyebrows use them.

Run inside Blender 5.2.2:
    import build_hair; build_hair.run()
"""

import math

import bmesh
import bpy
import numpy as np

import design_hair as HR
import glenn_common as G
import sdf as S
from shader_dsl import NB

OWNER = "hair"


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


def _hair_masks(verts):
    q = (np.asarray(verts, dtype=np.float32) - np.array([0, 0, HR.H.EYE_Z], dtype=np.float32)) \
        / np.float32(HR.H.D.scale)
    above = HR.scalp(q)
    height = S.smoothstep(0.0, 0.020, above)
    near = 1.0 - S.smoothstep(0.0008, 0.0035, above)
    ids = np.zeros(len(q), dtype=np.float32)
    best = np.full(len(q), 1e3, dtype=np.float32)
    rng = np.random.default_rng(61)
    for c in HR.clumps():
        d = c(q)
        closer = d < best
        best[closer] = d[closer]
        ids[closer] = rng.random()
    ids[best > 0.002] = 0.5
    return np.stack([height, ids, near, np.ones_like(ids)], axis=1)


def _hair_flow(verts, normals):
    """Comb direction per vertex: front to back on top, back and down on
    the sides. Used as the tangent for the anisotropic highlight."""
    q = (np.asarray(verts, dtype=np.float32) - np.array([0, 0, HR.H.EYE_Z], dtype=np.float32))
    d = q - HR.C0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    side = 1.0 - S.smoothstep(0.3, 0.8, d[:, 2])
    flow = np.stack([np.zeros(len(q)), np.ones(len(q)), -0.6 * side], axis=1).astype(np.float32)
    n = np.asarray(normals, dtype=np.float32)
    flow -= (flow * n).sum(axis=1, keepdims=True) * n
    flow /= np.maximum(np.linalg.norm(flow, axis=1, keepdims=True), 1e-6)
    return flow


def sculpt_material():
    mat = bpy.data.materials.get("MAT_HairSculpt") or bpy.data.materials.new("MAT_HairSculpt")
    nb = NB(mat)
    nb.clear()
    m = nb.node("ShaderNodeAttribute", attribute_name="GLN_HairMask", attribute_type="GEOMETRY")
    height, cid, near = nb.xyz(m.outputs["Vector"])
    obj = nb.texcoord("Object")
    flow = nb.node("ShaderNodeAttribute", attribute_name="GLN_HairFlow", attribute_type="GEOMETRY")
    streak = nb.noise(nb.vmath("MULTIPLY", obj, (700.0, 90.0, 700.0)), 1.0, detail=4.0)
    col = nb.mix(nb.smoothstep(0.05, 0.9, height), (0.010, 0.006, 0.004), (0.040, 0.023, 0.014))
    col = nb.mix(cid, nb.mix(0.5, col, (0.020, 0.012, 0.008)), nb.mix(0.5, col, (0.060, 0.034, 0.020)))
    col = nb.mix(nb.smoothstep(0.4, 0.8, streak) * 0.4, col, (0.075, 0.045, 0.026))
    col = nb.mix(near * 0.45, col, (0.16, 0.095, 0.065))
    normal = nb.bump(streak, strength=0.35, distance=0.0003)
    bsdf = nb.principled(base=col, rough=0.52, spec=0.25, aniso=0.45, tangent=flow.outputs["Vector"],
                         normal=normal)
    nb.output(bsdf)
    return mat


def sculpted(voxel=0.0012):
    verts, quads = S.mesh_field(HR.field, *HR.bounds(), voxel)
    me = S.to_blender_mesh("HAIR_Sculpt", verts, quads)
    attr = me.color_attributes.new("GLN_HairMask", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", _hair_masks(verts).ravel())
    nrm = np.empty(len(verts) * 3, dtype=np.float32)
    me.vertices.foreach_get("normal", nrm)
    flow = me.attributes.new("GLN_HairFlow", "FLOAT_VECTOR", "POINT")
    flow.data.foreach_set("vector", _hair_flow(verts, nrm.reshape(-1, 3)).ravel())
    obj = bpy.data.objects.new("HAIR_Sculpt", me)
    G.link(obj, "GLN_Hair", OWNER)
    G.set_material(obj, sculpt_material())
    return obj


def run(voxel=0.0012):
    G.ensure_collections()
    G.clear_owned(OWNER)
    return {"hair": sculpted(voxel).name}
