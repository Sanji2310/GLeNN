"""Build the GLeNN driver's head in Blender from the sculpted design field.

Creates, in the GLN_Head collection:
  HEAD_Skin     the sculpted head and neck (quad mesh from design_head.py),
                with a GLN_Masks colour attribute: R lips, G warmth, B beard
                shadow, used by the skin material
  HEAD_Eye_L/R  eyeballs with a real corneal bulge and a procedural iris
  HEAD_Brows    sculpted eyebrow masses; the left brow has a notch

The skin here is the design sculpt. Game topology is projected onto the same
field at the retopology stage.

Run inside Blender 5.2.2:
    import build_head; build_head.run()
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import build_hair
import design_head as H
import glenn_common as G
import sdf as S
from shader_dsl import NB

OWNER = "head"
COLL = "GLN_Head"
F32 = np.float32


def _local(p):
    """World points to the head design frame (eye line origin, unscaled)."""
    return (np.asarray(p, dtype=F32) - np.array([0, 0, H.EYE_Z], dtype=F32)) / F32(H.D.scale)


# -- skin ----------------------------------------------------------------------------

def _masks(verts):
    q = S.mirror_x(_local(verts))
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    lips = S.smoothstep(0.0010, -0.0006, H._lips(q))
    warm = np.zeros(len(q), dtype=F32)
    for c, r, a in (((0.046, -0.068, -0.032), 0.024, 0.8), ((0.0, -0.112, -0.040), 0.014, 0.6),
                    ((0.075, 0.016, -0.020), 0.030, 0.45)):
        d = np.sqrt(((q - np.array(c, dtype=F32)) ** 2).sum(axis=1))
        warm = np.maximum(warm, F32(a) * (1.0 - S.smoothstep(0.0, r, d)))
    line = -0.050 + 0.40 * (y + 0.050)
    beard = S.smoothstep(0.004, -0.006, z - line) * (1.0 - S.smoothstep(-0.130, -0.150, z))
    moustache = 1.0 - S.smoothstep(0.6, 1.0, np.sqrt((x / 0.021) ** 2 + ((y + 0.098) / 0.02) ** 2
                                                     + ((z + 0.064) / 0.0065) ** 2))
    beard = np.maximum(beard, moustache) * (1.0 - lips)
    return np.stack([lips, warm, beard, np.ones_like(lips)], axis=1)


def skin(voxel=0.0011):
    lo, hi = H.bounds()
    verts, quads = S.mesh_field(H.field, lo, hi, voxel)
    me = S.to_blender_mesh("HEAD_Skin", verts, quads)
    attr = me.color_attributes.new("GLN_Masks", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", _masks(verts).ravel())
    obj = bpy.data.objects.new("HEAD_Skin", me)
    G.link(obj, COLL, OWNER)
    G.set_material(obj, skin_material())
    return obj


def skin_material():
    mat = bpy.data.materials.get("MAT_Skin") or bpy.data.materials.new("MAT_Skin")
    nb = NB(mat)
    nb.clear()
    m = nb.node("ShaderNodeAttribute", attribute_name="GLN_Masks", attribute_type="GEOMETRY")
    lips, warm, beard = nb.xyz(m.outputs["Vector"])
    obj = nb.texcoord("Object")
    mottle = nb.noise(obj, 60.0, detail=3.0)
    base = nb.mix(nb.smoothstep(0.35, 0.75, mottle) * 0.25, G.PALETTE["skin"], (0.30, 0.165, 0.105))
    base = nb.mix(warm * 0.50, base, (0.36, 0.145, 0.095))
    base = nb.mix(beard * 0.30, base, (0.20, 0.155, 0.135))
    base = nb.mix(lips * 0.80, base, (0.27, 0.105, 0.075))
    pores = nb.noise(obj, 900.0, detail=1.0)
    rough = nb.mixf(lips, 0.50, 0.36) + (pores - 0.5) * 0.08
    normal = nb.bump(pores, strength=0.08, distance=0.0002)
    bsdf = nb.principled(base=base, rough=rough, sss=0.35, sss_radius=(1.0, 0.35, 0.18),
                         sss_scale=0.004, spec=0.45, normal=normal)
    nb.output(bsdf)
    return mat


# -- eyes ----------------------------------------------------------------------------

IRIS_R = 0.0066
PUPIL_R = 0.0026


def eyeball(name, center):
    """Sphere with the cornea bulging in front of the iris. Local -Y looks
    forward, so the iris shader can work in object space."""
    r = H.D.eye_r * H.D.scale
    ri = IRIS_R * H.D.scale
    rc = r * 0.65
    cz = -r * math.cos(math.asin(ri / r)) + math.sqrt(rc * rc - ri * ri)   # cornea centre on the axis
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=r, calc_uvs=True)
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(-90), 3, "X"))
    for v in bm.verts:
        p = v.co
        if p.y < 0:
            rho = math.hypot(p.x, p.z)
            if rho < ri:
                # cornea sphere surface at this radius
                yc = -cz - math.sqrt(max(rc * rc - rho * rho, 0.0))
                v.co.y = min(p.y, yc)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    obj = bpy.data.objects.new(name, me)
    obj.location = center
    G.link(obj, COLL, OWNER)
    G.set_material(obj, eye_material())
    return obj


def eye_material():
    mat = bpy.data.materials.get("MAT_Eye") or bpy.data.materials.new("MAT_Eye")
    nb = NB(mat)
    nb.clear()
    s = H.D.scale
    ri, rp = IRIS_R * s, PUPIL_R * s
    x, y, z = nb.xyz(nb.texcoord("Object"))
    rho = nb.sqrt(x * x + z * z)
    front = nb.smoothstep(0.0005, -0.0005, y)
    iris = nb.smoothstep(ri + 0.00025, ri - 0.00025, rho) * front
    pupil = nb.smoothstep(rp + 0.00018, rp - 0.00018, rho) * front
    ang = nb.atan2(z, x)
    fibres = nb.noise(nb.vec(ang * 6.0, rho * 900.0, 0.0), 1.0, detail=4.0, rough=0.6)
    t = rho / ri
    inner = nb.mix(nb.smoothstep(0.30, 0.95, t), (0.36, 0.20, 0.045), (0.11, 0.085, 0.030))
    iris_col = nb.mix(nb.smoothstep(0.3, 0.8, fibres) * 0.45, inner, (0.52, 0.32, 0.08))
    iris_col = nb.mix(nb.smoothstep(0.78, 1.0, t), iris_col, (0.04, 0.03, 0.02))
    sclera = nb.mix(nb.smoothstep(0.006, 0.013, rho) * 0.35, (0.80, 0.77, 0.72), (0.78, 0.52, 0.46))
    col = nb.mix(iris, sclera, iris_col)
    col = nb.mix(pupil, col, (0.005, 0.005, 0.006))
    bsdf = nb.principled(base=col, rough=0.35, coat=1.0, coat_rough=0.02, spec=0.5,
                         sss=0.15, sss_radius=(1.0, 0.4, 0.3), sss_scale=0.002)
    nb.output(bsdf)
    return mat


# -- brows ---------------------------------------------------------------------------

BROW_PATH = [(0.0115, -0.0905, 0.0120), (0.0215, -0.0895, 0.0148), (0.0330, -0.0865, 0.0168),
             (0.0450, -0.0795, 0.0168), (0.0555, -0.0700, 0.0142)]


def _on_skin(points_local):
    """Project local points onto the skin; returns world points and normals."""
    s = H.D.scale
    p = np.asarray(points_local, dtype=F32) * F32(s) + np.array([0, 0, H.EYE_Z], dtype=F32)
    p = S.project(H.field, p, iterations=6, eps=2e-4)
    n = S.gradient(H.field, p, 2e-4)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return p, n


def _strip(bm, uv, ctrl, width, thick, lift=0.0002, samples=12, flat_tip=True):
    p, n = _on_skin(ctrl)
    path = build_hair.catmull([Vector(v) for v in p], samples)
    nrm = build_hair.catmull([Vector(v) for v in n], samples)
    nrm = [v.normalized() for v in nrm]
    path = [a + b * lift for a, b in zip(path, nrm)]
    build_hair.clump_mesh(bm, uv, path, nrm, width, thick, section=8, flat_tip=flat_tip)


def brows():
    """Each brow is one main mass plus a few short upswept clumps at its
    head. The left brow (+X) is split by a small notch two thirds along."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    for side in (1, -1):
        path = [(side * x, y, z) for x, y, z in BROW_PATH]
        if side == 1:
            # notch: two masses with a gap between 0.038 and 0.041
            _strip(bm, uv, path[:3] + [(0.0378, -0.0848, 0.0170)], 0.0034, 0.0010)
            _strip(bm, uv, [(0.0412, -0.0825, 0.0170)] + path[3:], 0.0026, 0.0008, flat_tip=False)
        else:
            _strip(bm, uv, path, 0.0034, 0.0010, flat_tip=False)
        for i, (dx, dz) in enumerate(((0.0000, 0.0), (0.0040, 0.0006), (0.0080, 0.0010))):
            x0 = side * (0.0120 + dx)
            _strip(bm, uv, [(x0, -0.0905, 0.0108 + dz), (x0 + side * 0.0018, -0.0902, 0.0150 + dz),
                            (x0 + side * 0.0050, -0.0895, 0.0178 + dz)], 0.0018, 0.0006, samples=8, flat_tip=False)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("HEAD_Brows")
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    obj = bpy.data.objects.new("HEAD_Brows", me)
    G.link(obj, COLL, OWNER)
    G.set_material(obj, build_hair.hair_material())
    return obj


def run(voxel=0.0011):
    G.ensure_collections()
    G.clear_owned(OWNER)
    out = {"skin": skin(voxel)}
    for side, c in zip("LR", H.eye_centers()):
        out["eye_" + side] = eyeball("HEAD_Eye_" + side, c)
    out["brows"] = brows()
    return {k: v.name for k, v in out.items()}
