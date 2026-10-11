"""Full-face racing helmet for the racer: a separate, removable asset.

Box-modeled like the head:
    1. Add > Cube, Subdivide (5 cuts), To Sphere, Apply Scale.
    2. Delete the -X half, Mirror (Clipping) and Subdivision Surface.
    3. Grab each cage vertex (G) onto the helmet's outer shape: a dome on
       top, fuller and straighter below so the opening clears the jaw.
    4. Delete the faces under the neck line to open the helmet.
    5. Visor opening: select the front band at eye level, Inset, Extrude it
       inwards and crease its rim; the visor itself is a copy of those faces
       (Duplicate, Separate) so it can come off.
    6. Solidify for the shell thickness.

Colours: bone-white shell with an oxblood slash (a shader band in the
helmet's own coordinates, so it moves with it), dark tinted visor. No text,
numbers or logos.

    import helmet
    helmet.build()            # fitted on the head
    helmet.place(STAND)       # or moved onto its stand
"""

import math

import bmesh
import bpy
from mathutils import Vector

import looks
import sculpt_kit as K

SHELL, VISOR = "HLM_Shell", "HLM_Visor"
CENTER = Vector((0.0, 0.010, 1.660))   # fitted over the head
HALF_X, FRONT, BACK = 0.130, 0.160, 0.148
CHIN = 0.016                           # chin bar juts forward below the visor
UP, DOWN = 0.172, 0.150                # dome above the centre, fuller sides below it
P_FLAT, Q_UP, Q_DOWN = 2.2, 2.2, 4.0   # horizontal and vertical superellipse exponents
NECK_FRONT, NECK_BACK = 1.548, 1.575   # opening line, chin bar to nape
VISOR_Z = (1.640, 1.732)               # brow to mid-nose
VISOR_WRAP = math.radians(64)          # wraps round to the sides
SHELL_THICKNESS = 0.010
STAND = Vector((0.86, -0.12, 1.0))     # beside his right hand, for the presentation


def _collection(name="CHR_Helmet"):
    vl = bpy.context.view_layer
    vl.active_layer_collection = vl.layer_collection.children[name]


def _radius(d):
    """Distance from CENTER to the outer shape along unit direction d."""
    b = BACK
    if d.y < 0:
        z = CENTER.z + d.z * 0.15
        t = min(1.0, max(0.0, (1.645 - z) / 0.07))
        b = FRONT + CHIN * t * t * (3 - 2 * t)
    c, q = (UP, Q_UP) if d.z > 0 else (DOWN, Q_DOWN)
    h = (abs(d.x) / HALF_X) ** P_FLAT + (abs(d.y) / b) ** P_FLAT
    f = h ** (q / P_FLAT) + (abs(d.z) / c) ** q
    return f ** (-1.0 / q)


def _shape(name):
    ob = bpy.data.objects[name]
    mw = ob.matrix_world
    bm = bmesh.from_edit_mesh(ob.data)
    moves = []
    for v in bm.verts:
        p = mw @ v.co
        d = (p - CENTER).normalized()
        t = CENTER + d * _radius(d)
        if abs(p.x) < 1e-6:
            t.x = 0.0
        moves.append((v.index, tuple(t)))
    K.drag(name, moves)


def _neck_line(p):
    f = min(1.0, max(0.0, (p.y - CENTER.y + 0.10) / 0.25))
    return NECK_FRONT + (NECK_BACK - NECK_FRONT) * f


def _visor_edges(p):
    """Bottom and top line of the visor opening: an eye-port, a little
    narrower in height towards the sides."""
    s = (p.x / HALF_X) ** 2
    return VISOR_Z[0] + 0.006 + 0.012 * s, VISOR_Z[1] - 0.004 - 0.010 * s


def _in_visor(p):
    return VISOR_Z[0] < p.z < VISOR_Z[1] and abs(math.atan2(p.x, -(p.y - CENTER.y))) < VISOR_WRAP


def shell_material():
    """Bone white with an oxblood slash: a band where the helmet's own
    coordinates cross a tilted plane (mirrored with |x|), so the slash
    sweeps from the brow back and down each side and meets in a chevron."""
    m = bpy.data.materials.get("MAT_HelmetShell") or bpy.data.materials.new("MAT_HelmetShell")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type not in ("OUTPUT_MATERIAL", "BSDF_PRINCIPLED"):
            nt.nodes.remove(n)
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.28
    b.inputs["Coat Weight"].default_value = 0.6
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ax = nt.nodes.new("ShaderNodeMath")
    ax.operation = "ABSOLUTE"
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = Vector((-0.45, 0.55, 0.70)).normalized()
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["X"], ax.inputs[0])
    nt.links.new(ax.outputs[0], comb.inputs["X"])
    nt.links.new(sep.outputs["Y"], comb.inputs["Y"])
    nt.links.new(sep.outputs["Z"], comb.inputs["Z"])
    nt.links.new(comb.outputs[0], dot.inputs[0])

    def band(offset, half_width):
        sub = nt.nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = offset
        nt.links.new(dot.outputs["Value"], sub.inputs[0])
        ab = nt.nodes.new("ShaderNodeMath")
        ab.operation = "ABSOLUTE"
        nt.links.new(sub.outputs[0], ab.inputs[0])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.interpolation_type = "SMOOTHSTEP"
        mr.inputs["From Min"].default_value = half_width + 0.0008
        mr.inputs["From Max"].default_value = half_width - 0.0008
        nt.links.new(ab.outputs[0], mr.inputs["Value"])
        return mr

    main, thin = band(0.020, 0.018), band(0.054, 0.004)
    mx = nt.nodes.new("ShaderNodeMath")
    mx.operation = "MAXIMUM"
    nt.links.new(main.outputs[0], mx.inputs[0])
    nt.links.new(thin.outputs[0], mx.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = looks.srgb(looks.PALETTE["bone"])
    mix.inputs["B"].default_value = looks.srgb(looks.PALETTE["suit"])
    nt.links.new(mx.outputs[0], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    return m


def visor_material():
    m = bpy.data.materials.get("MAT_Visor") or bpy.data.materials.new("MAT_Visor")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.012, 0.013, 0.016, 1.0)
    b.inputs["Roughness"].default_value = 0.05
    b.inputs["Metallic"].default_value = 0.3
    return m


def build():
    _collection()
    K.mode("OBJECT")
    K.op(bpy.ops.mesh.primitive_cube_add, size=2.0, location=tuple(CENTER))
    ob = bpy.context.view_layer.objects.active
    ob.name = ob.data.name = SHELL
    K.mode("EDIT")
    K.op(bpy.ops.mesh.select_all, action="SELECT")
    K.op(bpy.ops.mesh.subdivide, number_cuts=7)
    K.op(bpy.ops.transform.tosphere, value=1.0)
    K.mode("OBJECT")
    r = sum(v.co.length for v in ob.data.vertices) / len(ob.data.vertices)
    ob.scale = (HALF_X / r, FRONT / r, UP / r)
    K.op(bpy.ops.object.transform_apply, location=False, rotation=False, scale=True)
    K.mode("EDIT")
    K.sel(SHELL, lambda co: co.x < -1e-5)
    K.op(bpy.ops.mesh.delete, type="VERT")
    K.mode("OBJECT")
    mirror = ob.modifiers.new("Mirror", "MIRROR")
    mirror.use_clip = mirror.use_mirror_merge = True
    K.mode("EDIT")
    _shape(SHELL)
    # open the neck
    K.sel(SHELL, lambda c: c.z < _neck_line(c), kind="FACE")
    K.op(bpy.ops.mesh.delete, type="FACE")
    # grab the opening's edge onto the neck line, so it runs clean, not in steps
    mw = ob.matrix_world
    bm = bmesh.from_edit_mesh(ob.data)
    edge = [(v.index, mw @ v.co) for v in bm.verts if v.is_boundary and (mw @ v.co).z < CENTER.z - 0.04]
    K.drag(SHELL, [(i, (p.x, p.y, _neck_line(p))) for i, p in edge])
    # visor opening: Inset, crease its rim, Extrude in
    K.sel(SHELL, _in_visor, kind="FACE")
    bm = bmesh.from_edit_mesh(ob.data)
    sel = [f for f in bm.faces if f.select]
    rim = {v.index: mw @ v.co for f in sel for e in f.edges
           if sum(1 for lf in e.link_faces if lf.select) == 1 for v in e.verts}
    mid = sum(VISOR_Z) / 2
    moves = []
    for i, p in rim.items():
        lo, hi = _visor_edges(p)
        if p.z < mid - 0.015:
            moves.append((i, (p.x, p.y, lo)))
        elif p.z > mid + 0.015:
            moves.append((i, (p.x, p.y, hi)))
    port = [f.index for f in sel]
    K.drag(SHELL, moves)
    K.sel(SHELL, lambda c: False, kind="FACE")    # face select mode, nothing selected
    bm = bmesh.from_edit_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    for i in port:
        bm.faces[i].select_set(True)
    bmesh.update_edit_mesh(ob.data)
    K.op(bpy.ops.mesh.inset, thickness=0.004, depth=0.0, use_boundary=False)
    bm = bmesh.from_edit_mesh(ob.data)
    port = [f.index for f in bm.faces if f.select]
    K.op(bpy.ops.mesh.region_to_loop)
    K.op(bpy.ops.transform.edge_crease, value=0.9)
    K.sel(SHELL, lambda c: False, kind="FACE")
    bm = bmesh.from_edit_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    for i in port:
        bm.faces[i].select_set(True)
    bmesh.update_edit_mesh(ob.data)
    K.op(bpy.ops.mesh.extrude_region_shrink_fatten, TRANSFORM_OT_shrink_fatten={"value": -0.007})
    bm = bmesh.from_edit_mesh(ob.data)
    visor_faces = {f.index for f in bm.faces if f.select}
    # the visor is a copy of the recessed faces, pushed back out to sit in the opening
    K.op(bpy.ops.mesh.duplicate)
    K.op(bpy.ops.mesh.separate, type="SELECTED")
    K.mode("OBJECT")
    vis = next(o for o in bpy.context.selected_objects if o != ob)
    vis.name = vis.data.name = VISOR
    K.focus(VISOR, mode="EDIT")
    K.op(bpy.ops.mesh.select_all, action="SELECT")
    K.op(bpy.ops.transform.shrink_fatten, value=0.004)
    K.mode("OBJECT")
    vs = vis.modifiers.new("Solidify", "SOLIDIFY")
    vs.thickness = 0.002
    # shell thickness after the subdivision
    sub = ob.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = sub.render_levels = 2
    so = ob.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = SHELL_THICKNESS
    so.offset = -1.0
    so.use_even_offset = False      # even thickness spikes at the creased visor recess
    so.use_quality_normals = True
    so.use_rim = True
    vsub = vis.modifiers.new("Subdivision", "SUBSURF")
    vsub.levels = vsub.render_levels = 2
    with K.override():
        bpy.context.view_layer.objects.active = vis
        bpy.ops.object.modifier_move_to_index(modifier="Subdivision", index=1)
    for o in (ob, vis):
        bpy.context.view_layer.objects.active = o
        K.focus(o.name, mode="OBJECT")
        K.op(bpy.ops.object.shade_smooth)
    ob.data.materials.append(shell_material())
    vis.data.materials.append(visor_material())
    vis.parent = ob
    vis.matrix_parent_inverse = ob.matrix_world.inverted()
    return {"shell_faces": len(ob.data.polygons), "visor_faces": len(visor_faces)}


def place(location=STAND, yaw=-0.5):
    """Move the helmet (the visor follows) off the head, e.g. onto its stand."""
    ob = bpy.data.objects[SHELL]
    ob.location = location
    ob.rotation_euler = (0.0, 0.0, yaw)


def stand(location=STAND):
    """A simple display stand (Add > Cylinder twice, Add > UV Sphere, Join)
    with the helmet set on it."""
    _collection()
    K.mode("OBJECT")
    x, y, z = location
    parts = []
    K.op(bpy.ops.mesh.primitive_cylinder_add, vertices=32, radius=0.15, depth=0.03, location=(x, y, 0.015))
    parts.append(bpy.context.view_layer.objects.active)
    K.op(bpy.ops.mesh.primitive_cylinder_add, vertices=24, radius=0.018, depth=z, location=(x, y, z / 2))
    parts.append(bpy.context.view_layer.objects.active)
    K.op(bpy.ops.mesh.primitive_uv_sphere_add, segments=24, ring_count=12, radius=0.07, location=(x, y, z + 0.02))
    rest = bpy.context.view_layer.objects.active
    rest.scale = (1.0, 1.2, 0.8)
    parts.append(rest)
    for o in bpy.context.view_layer.objects:
        o.select_set(o in parts)
    bpy.context.view_layer.objects.active = parts[0]
    K.op(bpy.ops.object.join)
    st = bpy.context.view_layer.objects.active
    st.name = st.data.name = "HLM_Stand"
    bevel = st.modifiers.new("Bevel", "BEVEL")
    bevel.width = 0.004
    bevel.segments = 2
    K.op(bpy.ops.object.shade_smooth_by_angle)
    st.data.materials.clear()
    st.data.materials.append(looks.cloth("MAT_Stand", "glove", 0.35))
    place(location)
    return st
