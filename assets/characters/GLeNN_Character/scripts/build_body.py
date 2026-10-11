"""Build the GLeNN driver's racing suit, gloves and boots in Blender from
the sculpted design fields (design_body.py).

Creates, in the GLN_Outfit collection:
  SUIT_Body    one-piece suit (the body is never seen bare). GLN_Livery
               colour attribute: R signed distance to the cream panels,
               G signed distance to the amber areas (metres, positive inside).
               The material draws crisp panel edges and amber pinstripes
               from these, and projects the number and team name from
               textures/suit_decals.png.
  SUIT_Gloves  gloves. GLN_Glove: R palm grip, G knuckle strip, B wrist strap.
  SUIT_Boots   boots. GLN_Boot: R sole, G heel tab, B ankle strap.

These are design sculpts. Game topology is projected onto the same fields at
the retopology stage.

Run inside Blender 5.2.2:
    import build_body; build_body.run()
"""

import os

import bpy
import numpy as np

import design_body as B
import glenn_common as G
import sdf as S
from shader_dsl import NB

OWNER = "outfit"
COLL = "GLN_Outfit"
F32 = np.float32
DECALS = "suit_decals.png"


def _mesh(name, field, part, voxel, attr, values):
    lo, hi = B.bounds(part)
    verts, quads = S.mesh_field(field, lo, hi, voxel)
    me = S.to_blender_mesh(name, verts, quads)
    a = me.color_attributes.new(attr, "FLOAT_COLOR", "POINT")
    a.data.foreach_set("color", values(verts).ravel())
    obj = bpy.data.objects.new(name, me)
    G.link(obj, COLL, OWNER)
    return obj


def _rgba(r, g, b):
    r = np.asarray(r, dtype=F32)
    return np.stack([r, np.asarray(g, dtype=F32), np.asarray(b, dtype=F32), np.ones_like(r)], axis=1)


# -- masks -------------------------------------------------------------------------------

def suit_masks(verts):
    q = S.mirror_x(np.asarray(verts, dtype=F32))
    return _rgba(B.cream_panels(q), B.amber_panels(q), np.zeros(len(q)))


def glove_masks(verts):
    q = S.mirror_x(np.asarray(verts, dtype=F32))
    f, pn, sd = B._hand_frame()
    R = np.stack([sd, pn, f], axis=1).astype(F32)
    h = S.local(q, B.J.wrist(), R)
    hy, hz = h[:, 1], h[:, 2]
    grip = np.minimum(hy - F32(0.004), hz - F32(0.012))                 # palm side of hand and fingers
    knuckle = np.minimum(F32(0.014) - np.abs(hz - F32(0.092)), -hy - F32(0.002))
    strap = F32(0.011) - np.abs(hz + F32(0.034))
    return _rgba(grip, knuckle, strap)


def boot_masks(verts):
    q = S.mirror_x(np.asarray(verts, dtype=F32))
    x, y, z = q[:, 0] - F32(B.J.ankle[0]), q[:, 1], q[:, 2]
    sole = F32(0.017) - z
    tab = np.minimum(np.minimum(F32(0.011) - np.abs(x), y - F32(0.072)), z - F32(0.024))
    strap = F32(0.011) - np.abs(z - F32(0.105))
    return _rgba(sole, tab, strap)


# -- materials ---------------------------------------------------------------------------

def _edge(nb, sd, width=0.0005):
    return nb.smoothstep(-width, width, sd)


def _fabric(nb, scale=2600.0):
    """Fine twill weave as a bump height and a roughness variation."""
    obj = nb.texcoord("Object")
    weave = nb.wave(obj, scale, kind="BANDS", direction="DIAGONAL", profile="SIN")
    fuzz = nb.noise(obj, 400.0, detail=2.0)
    return weave * 0.6 + fuzz * 0.4


def _decal_image():
    path = os.path.join(G.DIRS["textures"], DECALS)
    img = bpy.data.images.get(DECALS)
    if img is None and os.path.exists(path):
        img = bpy.data.images.load(path)
    if img is not None:
        img.colorspace_settings.name = "Non-Color"
    return img


def _decal(nb, img, x, z, cx, cz, w, h, cell, flip=False):
    """Sample one atlas cell projected along Y onto the suit. Returns
    (fill, outline) masks. cell = (u0, v0, du, dv) in the atlas."""
    u = ((cx - x) if flip else (x - cx)) / w + 0.5
    v = (z - cz) / h + 0.5
    u0, v0, du, dv = cell
    uv = nb.vec(nb.clamp01(u) * du + u0, nb.clamp01(v) * dv + v0, 0.0)
    n = nb.node("ShaderNodeTexImage", interpolation="Cubic", extension="CLIP")
    n.image = img
    nb.feed(n.inputs["Vector"], uv)
    r, g, _ = nb.xyz(n.outputs["Color"])
    return r, g


CELL_NUMBER = (0.0, 0.0, 0.5, 1.0)
CELL_STACKED = (0.5, 0.5, 0.5, 0.5)
CELL_LINE = (0.5, 0.0, 0.5, 0.5)


def suit_material():
    P = G.PALETTE
    mat = bpy.data.materials.get("MAT_Suit") or bpy.data.materials.new("MAT_Suit")
    nb = NB(mat)
    nb.clear()
    cream_sd, amber_sd, _ = nb.xyz(nb.node("ShaderNodeAttribute", attribute_name="GLN_Livery",
                                           attribute_type="GEOMETRY").outputs["Vector"])
    cream = _edge(nb, cream_sd)
    pin = _edge(nb, cream_sd + 0.0077) * (1.0 - _edge(nb, cream_sd + 0.0043))
    amber = nb.mx(_edge(nb, amber_sd), pin)
    x, y, z = nb.xyz(nb.texcoord("Object"))
    front = nb.smoothstep(-0.02, -0.05, y)
    back = nb.smoothstep(0.02, 0.05, y)
    col = nb.layer(P["indigo"], [(cream, P["cream"]), (amber, P["amber"])])
    printed = None
    img = _decal_image()
    if img is not None:
        n_fill, n_line = _decal(nb, img, x, z, -0.074, 1.232, 0.088, 0.088, CELL_NUMBER)
        w_fill, _ = _decal(nb, img, x, z, 0.074, 1.240, 0.100, 0.050, CELL_STACKED)
        b_fill, b_line = _decal(nb, img, x, z, 0.0, 1.190, 0.170, 0.170, CELL_NUMBER, flip=True)
        y_fill, _ = _decal(nb, img, x, z, 0.0, 1.383, 0.200, 0.100, CELL_LINE, flip=True)
        fill = nb.mx(nb.mx(n_fill, w_fill) * front, b_fill * back)
        line = nb.mx(n_line * front, b_line * back)
        col = nb.layer(col, [(line, P["amber"]), (fill, P["cream"]), (y_fill * back, P["indigo"])])
        printed = nb.mx(nb.mx(fill, line), y_fill * back)
    weave = _fabric(nb)
    rough = 0.66 + (weave - 0.5) * 0.10
    if printed is not None:
        rough = nb.mixf(printed, rough, 0.42)
    bsdf = nb.principled(base=col, rough=rough, sheen=0.35, sheen_rough=0.45,
                         sheen_tint=(0.85, 0.85, 1.0), spec=0.35,
                         normal=nb.bump(weave, strength=0.12, distance=0.0002))
    nb.output(bsdf)
    return mat


def glove_material():
    P = G.PALETTE
    mat = bpy.data.materials.get("MAT_Gloves") or bpy.data.materials.new("MAT_Gloves")
    nb = NB(mat)
    nb.clear()
    grip, knuckle, strap = nb.xyz(nb.node("ShaderNodeAttribute", attribute_name="GLN_Glove",
                                          attribute_type="GEOMETRY").outputs["Vector"])
    g, k, s = _edge(nb, grip), _edge(nb, knuckle), _edge(nb, strap)
    col = nb.layer(P["indigo"], [(k, P["cream"]), (g, P["charcoal"]), (s, P["amber"])])
    suede = nb.noise(nb.texcoord("Object"), 1800.0, detail=3.0)
    rough = nb.mixf(g, 0.55, 0.85) + (suede - 0.5) * 0.1
    bsdf = nb.principled(base=col, rough=rough, sheen=nb.mixf(g, 0.2, 0.6), sheen_rough=0.5, spec=0.35,
                         normal=nb.bump(_fabric(nb, 2200.0), strength=0.10, distance=0.0002))
    nb.output(bsdf)
    return mat


def boot_material():
    P = G.PALETTE
    mat = bpy.data.materials.get("MAT_Boots") or bpy.data.materials.new("MAT_Boots")
    nb = NB(mat)
    nb.clear()
    sole, tab, strap = nb.xyz(nb.node("ShaderNodeAttribute", attribute_name="GLN_Boot",
                                       attribute_type="GEOMETRY").outputs["Vector"])
    so, ta, st = _edge(nb, sole), _edge(nb, tab), _edge(nb, strap)
    col = nb.layer(P["indigo"], [(st, P["amber"]), (ta, P["amber"]), (so, P["cream"])])
    grain = nb.noise(nb.texcoord("Object"), 900.0, detail=3.0)
    rough = nb.mixf(so, 0.40, 0.70) + (grain - 0.5) * 0.08
    bsdf = nb.principled(base=col, rough=rough, spec=0.45,
                         normal=nb.bump(grain, strength=0.06, distance=0.0002))
    nb.output(bsdf)
    return mat


def run(voxel=0.003, glove_voxel=0.0015, boot_voxel=0.002):
    G.ensure_collections()
    G.clear_owned(OWNER)
    suit = _mesh("SUIT_Body", B.suit, "suit", voxel, "GLN_Livery", suit_masks)
    G.set_material(suit, suit_material())
    gloves = _mesh("SUIT_Gloves", B.gloves, "gloves", glove_voxel, "GLN_Glove", glove_masks)
    G.set_material(gloves, glove_material())
    boots = _mesh("SUIT_Boots", B.boots, "boots", boot_voxel, "GLN_Boot", boot_masks)
    G.set_material(boots, boot_material())
    out = {}
    for o in (suit, gloves, boots):
        o.data.shade_smooth()
        out[o.name] = len(o.data.polygons)
    return out
