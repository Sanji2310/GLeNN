"""Build the GLeNN driver character, studio and presentation renders.

Run headless with Blender:
    blender -b -P build_glenn_driver.py -- --quality final --renders renders --blend glenn_driver.blend
or with the `bpy` Python module:
    python build_glenn_driver.py --quality preview

Inside a running Blender (Text Editor or an MCP bridge) call
`build(quality="final", ...)` after adding this folder to sys.path.
"""

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import bpy
from mathutils import Matrix, Vector, Euler

import sdf_core as S
import shapes_head as H
import shapes_body as B
import shapes_helmet as HL
import materials as M
import blender_util as U

QUALITY = {
    # mesh resolution (voxel size in metres) per part, render samples and size
    "draft": dict(suit=0.0055, glove=0.0022, boot=0.0035, head=0.0018, hair=0.0018, brow=0.0008,
                  shell=0.0024, visor=0.0016, hw=0.0020, samples=12, scale=0.35),
    "preview": dict(suit=0.0040, glove=0.0016, boot=0.0025, head=0.0013, hair=0.0013, brow=0.0006,
                    shell=0.0016, visor=0.0011, hw=0.0014, samples=32, scale=0.5),
    "final": dict(suit=0.0026, glove=0.0011, boot=0.0018, head=0.00085, hair=0.0009, brow=0.00045,
                  shell=0.0011, visor=0.0008, hw=0.0010, samples=64, scale=1.0),
    # density used for the committed .blend (keeps the file small enough for git)
    "blend": dict(suit=0.0045, glove=0.0018, boot=0.0028, head=0.0013, hair=0.0014, brow=0.0006,
                  shell=0.0019, visor=0.0013, hw=0.0016, samples=64, scale=1.0),
}


def log(*a):
    print("[glenn]", *a, flush=True)


# --------------------------------------------------------------------------
# poses
# --------------------------------------------------------------------------

def _hero_base():
    r = {}
    r["pelvis"] = (0.0, 0.0, -6.0)
    r["spine"] = (3.0, 0.0, -3.0)
    r["chest"] = (4.0, -1.5, -6.0)          # subtle forward lean, shoulders angled
    r["neck"] = (2.0, 0.0, 5.0)
    r["head"] = (1.5, 1.5, 10.0)            # turned towards (and just past) the lens
    r["clav_L"] = (0.0, 0.0, 1.5)
    r["clav_R"] = (0.0, 0.0, -1.0)
    r["thigh_L"] = (-2.0, -1.0, 4.0)
    r["thigh_R"] = (2.0, 1.0, -2.0)
    r["shin_L"] = (3.0, 0.0, 0.0)
    return r


# helmet carried in front of the waist in both hands, visor towards the camera
HERO_HELMET = dict(center=(0.035, -0.300, 1.075), yaw=14.0, pitch=-6.0, roll=0.0)
HOLD_CONTACT = (0.1185, 0.004, -0.040)       # helmet-local palm contact on each side


def _helmet_rotation(yaw, pitch, roll):
    return S.rot_z(yaw) @ S.rot_x(pitch) @ S.rot_y(roll)


def hero_pose():
    """Hero pose and the matching helmet transform (helmet-local -> world)."""
    h = HERO_HELMET
    Rh = _helmet_rotation(h["yaw"], h["pitch"], h["roll"])
    C = np.asarray(h["center"])
    rot = _hero_base()
    for side in (1, -1):
        cl = np.array([side * HOLD_CONTACT[0], HOLD_CONTACT[1], HOLD_CONTACT[2]])
        cl = S.project(HL.shell_sdf, cl[None], 0.0004, iters=12)[0]
        g = S.grad(HL.shell_sdf, cl[None], 0.0005)[0]
        n_w = B.norm(Rh @ g)                       # outward normal = back of the hand
        contact = C + Rh @ cl
        u = Rh @ B.norm(np.array([side * 0.05, -0.55, -0.83]))   # fingers forward and down
        u = B.norm(u - n_w * (u @ n_w))
        v = np.cross(u, n_w) * side
        axes = np.stack([u, v, n_w], axis=1)
        wrist = contact - u * 0.056 + n_w * 0.021
        pole = B.norm(np.array([side * 0.7, 0.55, -0.45]))
        rot = B.solve_arm(rot, side, wrist, axes, pole)
    return B.Pose(rot), _helmet_matrix(C, Rh)


def _helmet_matrix(C, R):
    M4 = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M4[i][j] = R[i, j]
        M4[i][3] = C[i]
    return M4


def neutral_pose():
    return B.Pose({"neck": (2.5, 0.0, 0.0), "head": (1.0, 0.0, 0.0)})


# --------------------------------------------------------------------------
# geometry builders
# --------------------------------------------------------------------------

def _mesh(name, f, lo, hi, h, col, mat, rest_fn=None, matrix=None, relax=1):
    t = time.time()
    V, F = S.mesh_sdf(f, np.asarray(lo), np.asarray(hi), h, relax_iters=relax)
    attrs = {"rest": rest_fn(V)} if rest_fn else None
    ob = U.new_mesh_object(name, V, F, collection=col, attrs=attrs)
    ob.data.materials.append(mat)
    if matrix is not None:
        ob.matrix_world = matrix
    log(f"{name}: {len(V)} verts {len(F)} quads in {time.time() - t:.1f}s")
    return ob


def head_matrix(pose):
    T = Matrix.Translation(Vector(B.HEAD_ORIGIN))
    Sc = Matrix.Scale(B.HEAD_SCALE, 4)
    return Matrix(pose.matrix("head").tolist()) @ T @ Sc


_HEAD_CACHE = {}


def build_head(col, q, pose, mats, tag, gaze=(0.0, 0.0)):
    root = bpy.data.objects.new(f"Head_{tag}", None)
    col.objects.link(root)
    root.matrix_world = head_matrix(pose)
    root.empty_display_size = 0.12
    key = q["head"]
    if key not in _HEAD_CACHE:
        _HEAD_CACHE[key] = {}
    cache = _HEAD_CACHE[key]
    parts = [
        ("Skin", lambda P: H.head_sdf(P), (-0.095, -0.135, -0.25), (0.095, 0.125, 0.135), q["head"], mats["skin"]),
        ("Hair", H.hair_sdf, (-0.095, -0.125, -0.07), (0.095, 0.125, 0.14), q["hair"], mats["hair"]),
        ("Brows", H.brows_sdf, (-0.068, -0.115, -0.008), (0.068, -0.056, 0.042), q["brow"], mats["brow"]),
    ]
    for pname, f, lo, hi, h, mat in parts:
        name = f"{pname}_{tag}"
        if pname in cache:
            ob = bpy.data.objects.new(name, cache[pname])     # shared mesh data
            col.objects.link(ob)
        else:
            ob = _mesh(name, f, lo, hi, h, col, mat)
            cache[pname] = ob.data
        ob.parent = root
    for side in (1, -1):
        me = bpy.data.meshes.get("GLN_EyeMesh")
        if me is None:
            bm_ob = bpy.ops.mesh.primitive_uv_sphere_add(radius=H.EYE_R, segments=64, ring_count=32)
            tmp = bpy.context.object
            me = tmp.data
            me.name = "GLN_EyeMesh"
            me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
            bpy.data.objects.remove(tmp)
            me.materials.append(mats["eye"])
        e = bpy.data.objects.new(f"Eye_{'L' if side > 0 else 'R'}_{tag}", me)
        col.objects.link(e)
        e.parent = root
        e.location = Vector(H.eye_center(side))
        # gaze: yaw towards +X is a rotation about Z
        e.rotation_euler = Euler((math.radians(gaze[1]), 0.0, math.radians(gaze[0])), "XYZ")
    return root


def build_body(col, q, pose, mats, tag, curl_L, curl_R):
    lo = (-0.40, -0.40, 0.06)
    hi = (0.40, 0.30, 1.54)
    _mesh(f"Suit_{tag}", lambda P: B.suit_sdf(P, pose), lo, hi, q["suit"], col, mats["suit"],
          rest_fn=lambda V: B.suit_rest_coords(V, pose))
    for side, curl in ((1, curl_L), (-1, curl_R)):
        sname = "L" if side > 0 else "R"
        Wr = pose.J[f"hand_{sname}"]
        glo = Wr - 0.24
        ghi = Wr + 0.24
        _mesh(f"Glove_{sname}_{tag}", lambda P, s=side, c=curl: B.glove_sdf(P, pose, s, c), glo, ghi, q["glove"], col,
              mats["glove"], rest_fn=lambda V, s=side, c=curl: B.glove_rest_coords(V, pose, s, c))
        A = pose.J[f"foot_{sname}"]
        _mesh(f"Boot_{sname}_{tag}", lambda P, s=side: B.boot_sdf(P, pose, s), A - np.array([0.10, 0.25, 0.095]),
              A + np.array([0.10, 0.12, 0.10]), q["boot"], col, mats["boot"],
              rest_fn=lambda V, s=side: B.boot_rest_coords(V, pose, s))


_HELMET_CACHE = {}


def build_helmet(col, q, mats, tag, matrix):
    root = bpy.data.objects.new(f"Helmet_{tag}", None)
    col.objects.link(root)
    root.matrix_world = matrix
    root.empty_display_size = 0.15
    lo, hi = HL.BOUNDS
    for pname, f, h, mat in (("Shell", HL.shell_sdf, q["shell"], mats["helmet"]),
                             ("Visor", HL.visor_sdf, q["visor"], mats["visor"]),
                             ("Hardware", HL.hardware_sdf, q["hw"], mats["hw"])):
        name = f"Helmet{pname}_{tag}"
        if pname in _HELMET_CACHE:
            ob = bpy.data.objects.new(name, _HELMET_CACHE[pname])   # shared mesh data
            col.objects.link(ob)
        else:
            ob = _mesh(name, f, lo, hi, h, col, mat)
            _HELMET_CACHE[pname] = ob.data
        ob.parent = root
    helmet_decals(col, root, mats, tag)
    return root


# --------------------------------------------------------------------------
# decals: GLeNN wordmark and driver number, projected onto the surfaces
# --------------------------------------------------------------------------

def _text_mesh(name, body, size, col, extrude=0.0, shear=0.18, spacing=1.06, offset=0.0, align="CENTER"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.shear = shear
    cu.space_character = spacing
    cu.offset = offset
    cu.extrude = extrude
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.resolution_u = 4
    ob = bpy.data.objects.new(name, cu)
    col.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    out = bpy.data.objects.new(name, me)
    col.objects.link(out)
    return out


def _conform(ob, target, offset=0.0004, levels=3):
    sub = ob.modifiers.new("dense", "SUBSURF")
    sub.subdivision_type = "SIMPLE"
    sub.levels = levels
    sub.render_levels = levels
    sw = ob.modifiers.new("project", "SHRINKWRAP")
    sw.target = target
    sw.wrap_method = "NEAREST_SURFACEPOINT"
    sw.wrap_mode = "OUTSIDE_SURFACE"
    sw.offset = offset


def _place(ob, loc, normal, up=(0, 0, 1), parent=None):
    n = Vector(normal).normalized()
    upv = Vector(up)
    x = upv.cross(n).normalized() if abs(upv.dot(n)) < 0.99 else Vector((1, 0, 0))
    y = n.cross(x).normalized()
    M4 = Matrix(((x[0], y[0], n[0], loc[0]), (x[1], y[1], n[1], loc[1]), (x[2], y[2], n[2], loc[2]), (0, 0, 0, 1)))
    if parent is not None:
        ob.parent = parent
        ob.matrix_parent_inverse = Matrix.Identity(4)
        ob.matrix_basis = M4
    else:
        ob.matrix_world = M4


def helmet_decals(col, root, mats, tag):
    shell = [c for c in root.children if c.name.startswith("HelmetShell")][0]
    items = []
    for s in (1, -1):
        # wordmark on each side behind the eye-port, number on the chin bar sides
        t = _text_mesh(f"HelmetWord_{'L' if s > 0 else 'R'}_{tag}", "GLeNN", 0.030, col, offset=0.0006)
        _place(t, (s * 0.112, 0.088, -0.004), (s, 0.35, 0.05), up=(0, 0, 1), parent=root)   # on the graphite blade
        items.append((t, mats["decal_white"]))
        n = _text_mesh(f"HelmetNum_{'L' if s > 0 else 'R'}_{tag}", "07", 0.028, col, offset=0.0010, shear=0.2)
        _place(n, (s * 0.082, -0.106, -0.070), (s * 0.7, -0.7, -0.05), parent=root)
        items.append((n, mats["decal_white"]))
    back = _text_mesh(f"HelmetBack_{tag}", "GLeNN", 0.034, col, offset=0.0006)
    _place(back, (0.0, 0.168, 0.030), (0.0, 1.0, 0.1), parent=root)
    items.append((back, mats["decal_blue"]))
    top = _text_mesh(f"HelmetTopNum_{tag}", "07", 0.040, col, offset=0.0012, shear=0.2)
    _place(top, (0.0, 0.070, 0.152), (0.0, 0.25, 1.0), up=(0, -1, 0), parent=root)
    items.append((top, mats["decal_graphite"]))
    for ob, mat in items:
        ob.data.materials.clear()
        ob.data.materials.append(mat)
        _conform(ob, shell, offset=0.00035)


def suit_decals(col, pose, mats, suit, tag):
    """Chest wordmark, back wordmark and belt marks (positions in rest space,
    carried by the torso bones)."""
    def put(name, text, size, loc, normal, bone, up=(0, 0, 1), mat="decal_white", offset=0.0008, shear=0.18):
        ob = _text_mesh(f"{name}_{tag}", text, size, col, offset=offset, shear=shear)
        R = pose.R[bone]
        jb = B.JOINTS[bone][1]
        p = pose.J[bone] + R @ (np.asarray(loc) - jb)
        n = R @ np.asarray(normal)
        u = R @ np.asarray(up)
        _place(ob, p, n, up=u)
        ob.data.materials.clear()
        ob.data.materials.append(mats[mat])
        _conform(ob, suit, offset=0.0006)
        return ob

    put("ChestWord", "GLeNN", 0.026, (-0.072, -0.105, 1.355), (-0.15, -1.0, 0.25), "chest")
    put("BackWord", "GLeNN", 0.044, (0.044, 0.108, 1.372), (0.0, 1.0, 0.25), "chest")   # clear of the back blade
    put("BeltWordL", "GLeNN", 0.020, (0.060, -0.098, 1.060), (0.45, -1.0, 0.0), "spine")
    put("BeltWordR", "GLeNN", 0.020, (-0.060, -0.098, 1.060), (-0.45, -1.0, 0.0), "spine")
    put("ChestNum", "07", 0.030, (0.068, -0.110, 1.330), (0.2, -1.0, 0.25), "chest", mat="decal_blue", shear=0.2)


# --------------------------------------------------------------------------
# studio
# --------------------------------------------------------------------------

def build_studio(col, mats):
    # curved cyclorama, near-black
    bpy.ops.mesh.primitive_plane_add(size=1)
    cyc = bpy.context.object
    cyc.name = "Studio_Cyc"
    for c in cyc.users_collection:
        c.objects.unlink(cyc)
    col.objects.link(cyc)
    me = cyc.data
    import bmesh
    bm = bmesh.new()
    W, D, Hh, R = 14.0, 7.0, 7.0, 1.6
    rows = []
    for i in range(24):
        a = i / 23.0 * (math.pi / 2)
        y = 2.5 + R * math.sin(a)
        z = R - R * math.cos(a)
        rows.append((y, z))
    prof = [(-D, 0.0)] + rows + [(2.5 + R, Hh)]
    verts = []
    for y, z in prof:
        verts.append([bm.verts.new((-W / 2, y, z)), bm.verts.new((W / 2, y, z))])
    for a, b in zip(verts[:-1], verts[1:]):
        bm.faces.new((a[0], a[1], b[1], b[0]))
    bm.to_mesh(me)
    bm.free()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.materials.clear()
    me.materials.append(mats["backdrop"])
    # lights: large key softbox, cool and warm rims, top, low fill
    L = []
    L.append(U.add_area_light("Key", (-1.9, -2.6, 2.6), (0.0, 0.0, 1.30), 50, 1.4, size_y=1.9, color=(1.0, 0.95, 0.88), spread=70, collection=col))
    L.append(U.add_area_light("RimCool", (2.1, 1.9, 2.0), (0.0, 0.0, 1.25), 46, 0.35, size_y=2.4, color=(0.55, 0.72, 1.0), spread=45, collection=col))
    L.append(U.add_area_light("RimWarm", (-2.2, 1.7, 1.9), (0.0, 0.0, 1.25), 30, 0.35, size_y=2.2, color=(1.0, 0.86, 0.72), spread=45, collection=col))
    L.append(U.add_area_light("Top", (0.2, 0.3, 3.6), (0.0, 0.0, 1.2), 24, 1.6, size_y=1.6, color=(1.0, 0.98, 0.95), spread=60, collection=col))
    L.append(U.add_area_light("Fill", (2.4, -2.6, 1.2), (0.0, 0.0, 1.1), 7, 2.0, size_y=2.0, color=(0.85, 0.9, 1.0), collection=col))
    L.append(U.add_area_light("FloorKick", (0.0, -2.2, 0.15), (0.0, 0.0, 0.6), 3.5, 2.0, size_y=0.4, color=(0.8, 0.85, 1.0), collection=col))
    # flat floor for the orthographic turnaround (the cyclorama would block the rear camera)
    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0.0, 0.0, 0.0))
    fl = bpy.context.object
    fl.name = "Studio_OrthoFloor"
    for c in fl.users_collection:
        c.objects.unlink(fl)
    col.objects.link(fl)
    fl.data.materials.append(mats["backdrop"])
    fl.hide_render = True
    return cyc, L


def build_world():
    w = bpy.data.worlds.new("GLN_World")
    bpy.context.scene.world = w
    try:
        w.use_nodes = True
    except AttributeError:
        pass
    nt = w.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.004, 0.0045, 0.0055, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    return w


# --------------------------------------------------------------------------
# view setup
# --------------------------------------------------------------------------

VIEWS = [
    # name, collection, camera, (resolution), override
    ("01_front_ortho", "neutral"),
    ("02_side_ortho", "neutral"),
    ("03_rear_ortho", "neutral"),
    ("04_three_quarter", "neutral"),
    ("05_hero_three_quarter", "hero"),
    ("06_face_closeup", "neutral"),
    ("07_helmet_closeup", "display"),
    ("08_wireframe", "neutral"),
    ("09_clay_sculpt", "neutral"),
    ("10_final_material", "neutral"),
]


def build_cameras(col):
    cams = {}
    oz = 0.885
    cams["01_front_ortho"] = (U.add_camera("Cam_FrontOrtho", (0, -8, oz), (0, 0, oz), ortho_scale=1.98, collection=col), (1080, 1920))
    cams["02_side_ortho"] = (U.add_camera("Cam_SideOrtho", (-8, 0, oz), (0, 0, oz), ortho_scale=1.98, collection=col), (1080, 1920))
    cams["03_rear_ortho"] = (U.add_camera("Cam_RearOrtho", (0, 8, oz), (0, 0, oz), ortho_scale=1.98, collection=col), (1080, 1920))
    for k in ("01_front_ortho", "02_side_ortho", "03_rear_ortho"):
        cams[k][0].data.clip_end = 40.0
    cams["04_three_quarter"] = (U.add_camera("Cam_ThreeQuarter", (-3.05, -4.25, 1.22), (0.0, 0.0, 0.88), lens=70, collection=col), (1080, 1920))
    cams["05_hero_three_quarter"] = (U.add_camera("Cam_Hero", (0.98, -1.80, 1.42), (0.06, -0.12, 1.31), lens=60,
                                                  collection=col, dof=(1.82, 4.0)), (1600, 2000))
    cams["06_face_closeup"] = (U.add_camera("Cam_Face", (-0.40, -0.92, 1.67), (0.0, -0.04, 1.635), lens=95,
                                            collection=col, dof=(0.97, 5.6)), (1600, 1600))
    cams["07_helmet_closeup"] = (U.add_camera("Cam_Helmet", (3.41, -1.13, 1.18), (3.20, 0.0, 0.99), lens=95,
                                              collection=col, dof=(1.16, 5.6)), (1920, 1440))
    cams["08_wireframe"] = (U.add_camera("Cam_Wire", (-0.34, -0.66, 1.67), (0.0, -0.05, 1.615), lens=85, collection=col), (1600, 1600))
    cams["09_clay_sculpt"] = (U.add_camera("Cam_Clay", (-1.15, -2.0, 1.45), (0.0, 0.0, 1.22), lens=62, collection=col), (1600, 2000))
    cams["10_final_material"] = (U.add_camera("Cam_Material", (1.25, -2.15, 1.42), (0.0, 0.0, 1.20), lens=62,
                                              collection=col, dof=(2.45, 5.6)), (1600, 2000))
    return cams


def make_override_materials():
    clay = M.clay("GLN_ClayOverride", color=(0.36, 0.33, 0.31), rough=0.5)
    # wireframe: thin dark lines over light clay, so the real topology reads
    wire = bpy.data.materials.new("GLN_WireOverride")
    from shader_dsl import NB
    nb = NB(wire)
    nb.clear()
    wf = nb.node("ShaderNodeWireframe", use_pixel_size=True)
    wf.inputs["Size"].default_value = 0.9
    base = nb.principled(base=(0.42, 0.42, 0.44), rough=0.55, spec=0.3)
    line = nb.principled(base=(0.010, 0.030, 0.080), rough=0.6)
    mix = nb.node("ShaderNodeMixShader")
    nb.links.new(wf.outputs[0], mix.inputs[0])
    nb.links.new(base.outputs[0], mix.inputs[1])
    nb.links.new(line.outputs[0], mix.inputs[2])
    out = nb.node("ShaderNodeOutputMaterial")
    nb.links.new(mix.outputs[0], out.inputs["Surface"])
    return clay, wire


def set_visible(cols, show):
    for name, c in cols.items():
        c.hide_render = name not in show
        c.hide_viewport = name not in show


def _override(objs, mat):
    """Swap every material slot on objs to mat; returns a restore list."""
    saved = []
    for ob in objs:
        if ob.type != "MESH":
            continue
        for i, slot in enumerate(ob.material_slots):
            saved.append((ob, i, slot.material))
            slot.link = "OBJECT"
            slot.material = mat
    return saved


def _restore(saved):
    for ob, i, mat in saved:
        ob.material_slots[i].material = mat
        ob.material_slots[i].link = "DATA"


def render_views(scene, cols, cams, out_dir, q, views=None, overrides=None):
    os.makedirs(out_dir, exist_ok=True)
    cyc = bpy.data.objects.get("Studio_Cyc")
    floor = bpy.data.objects.get("Studio_OrthoFloor")
    for name, group in VIEWS:
        if views and name not in views:
            continue
        cam, res = cams[name]
        set_visible(cols, {"studio", group})
        ortho = name.endswith("_ortho")
        cyc.hide_render = ortho
        floor.hide_render = not ortho
        scene.cycles.samples = q["samples"]
        saved = []
        if name == "08_wireframe":
            saved = _override(cols[group].all_objects, overrides[1])
            scene.cycles.samples = max(16, q["samples"] // 3)
        elif name == "09_clay_sculpt":
            saved = _override(cols[group].all_objects, overrides[0])
            scene.cycles.samples = max(16, q["samples"] // 2)
        rx, ry = int(res[0] * q["scale"]), int(res[1] * q["scale"])
        t = time.time()
        U.render_to(scene, cam, os.path.join(out_dir, f"{name}.png"), res=(rx, ry))
        _restore(saved)
        log(f"rendered {name} ({rx}x{ry}) in {time.time() - t:.0f}s")
    cyc.hide_render = False
    floor.hide_render = True
    set_visible(cols, {"studio", "neutral", "hero", "display"})


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build(quality="preview", render_dir=None, blend_path=None, views=None, groups=("neutral", "hero", "display")):
    q = QUALITY[quality]
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    device = U.setup_cycles(scene, samples=q["samples"], res=(1600, 2000))
    log("render device:", device)
    build_world()
    mats = {
        "skin": M.skin(), "eye": M.eye(), "hair": M.hair(), "brow": M.hair("GLN_Brow", color=(0.030, 0.019, 0.013)),
        "suit": M.suit(), "glove": M.glove(), "boot": M.boot(), "helmet": M.helmet(), "visor": M.visor(),
        "hw": M.metal("GLN_HelmetHardware", color=(0.06, 0.062, 0.07), rough=0.28), "backdrop": M.backdrop(),
        "decal_white": M.decal("GLN_DecalWhite", (0.80, 0.80, 0.78)),
        "decal_blue": M.decal("GLN_DecalBlue", M.ELECTRIC_HOT, rough=0.35),
        "decal_graphite": M.decal("GLN_DecalGraphite", M.GRAPHITE, rough=0.35),
        "plinth": M.rubber("GLN_Plinth", color=(0.02, 0.02, 0.022)),
    }
    root = scene.collection
    cols = {n: U.get_collection(f"GLN_{n.capitalize()}", root) for n in ("studio", "neutral", "hero", "display")}
    cam_col = U.get_collection("GLN_Cameras", root)
    build_studio(cols["studio"], mats)

    if "neutral" in groups:
        pose = neutral_pose()
        build_body(cols["neutral"], q, pose, mats, "Neutral", B.REST_CURL, B.REST_CURL)
        suit = bpy.data.objects["Suit_Neutral"]
        suit_decals(cols["neutral"], pose, mats, suit, "Neutral")
        build_head(cols["neutral"], q, pose, mats, "Neutral", gaze=(0.0, -1.0))
    if "hero" in groups:
        pose, Mh = hero_pose()
        build_body(cols["hero"], q, pose, mats, "Hero", B.HOLD_CURL, B.HOLD_CURL)
        suit = bpy.data.objects["Suit_Hero"]
        suit_decals(cols["hero"], pose, mats, suit, "Hero")
        build_head(cols["hero"], q, pose, mats, "Hero", gaze=(-6.0, 1.0))
        build_helmet(cols["hero"], q, mats, "Hero", Mh)
    if "display" in groups:
        # helmet on a plinth for the product close-up
        plinth_h = 0.86
        bpy.ops.mesh.primitive_cylinder_add(radius=0.16, depth=plinth_h, vertices=96, location=(3.2, 0.0, plinth_h / 2))
        pl = bpy.context.object
        pl.name = "Display_Plinth"
        for c in pl.users_collection:
            c.objects.unlink(pl)
        cols["display"].objects.link(pl)
        pl.data.materials.append(mats["plinth"])
        bev = pl.modifiers.new("bevel", "BEVEL")
        bev.width = 0.01
        bev.segments = 4
        Mdisp = Matrix.Translation((3.2, 0.0, plinth_h + 0.128)) @ Matrix.Rotation(math.radians(-28), 4, "Z") \
            @ Matrix.Rotation(math.radians(-4), 4, "X")
        build_helmet(cols["display"], q, mats, "Display", Mdisp)
        # product lighting for the plinth (the studio rig is aimed at the driver)
        tgt = (3.2, 0.0, plinth_h + 0.12)
        U.add_area_light("DisplayKey", (2.35, -1.05, 1.75), tgt, 8, 0.9, size_y=1.2, color=(1.0, 0.96, 0.9), collection=cols["display"])
        U.add_area_light("DisplayRim", (3.95, 0.75, 1.55), tgt, 7, 0.25, size_y=1.2, color=(0.6, 0.75, 1.0), spread=50, collection=cols["display"])
        U.add_area_light("DisplayTop", (3.1, -0.2, 2.4), tgt, 4, 1.0, color=(1.0, 0.98, 0.95), collection=cols["display"])
        U.add_area_light("DisplayFill", (3.9, -1.2, 1.0), tgt, 1.2, 1.2, color=(0.85, 0.9, 1.0), collection=cols["display"])

    cams = build_cameras(cam_col)
    overrides = make_override_materials()
    log(f"scene built in {time.time() - t0:.0f}s")
    if blend_path:
        set_visible(cols, {"studio", "neutral", "hero", "display"})
        scene.camera = cams["05_hero_three_quarter"][0]
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blend_path), compress=True)
        log("saved", blend_path)
    if render_dir:
        render_views(scene, cols, cams, render_dir, q, views=views, overrides=overrides)
    log(f"done in {time.time() - t0:.0f}s")
    return scene


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="preview", choices=list(QUALITY))
    ap.add_argument("--renders", default=None)
    ap.add_argument("--blend", default=None)
    ap.add_argument("--views", nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=["neutral", "hero", "display"])
    a = ap.parse_args(argv)
    build(a.quality, a.renders, a.blend, a.views, tuple(a.groups))


if __name__ == "__main__":
    main()
