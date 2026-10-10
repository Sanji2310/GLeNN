"""GLeNN racing helmet: a separate, removable hard-surface asset.

The shell is generated as one clean quad grid over a parametric helmet
surface. Columns run around the helmet (azimuth) and rows run from the crown
down to the bottom edge (elevation). The eye port is cut along grid lines, so
the opening has proper edge flow and the UVs come straight from the
parameters. A quad patch closes the crown, so there is no pole.

Objects (collection GLN_Helmet, all parented to HLM_Root):
  HLM_Shell   composite shell, Solidify + Subdivision
  HLM_Visor   tinted visor, separate surface over the eye port
  HLM_Gasket  rubber seal around the eye port
  HLM_Trim    rubber trim along the bottom edge
  HLM_Pivots  visor pivot covers
  HLM_Spoiler rear spoiler lip
  HLM_Intake  crown air intake (HLM_IntakeMouth is its dark inlet)
  HLM_Vents   chin vents
  HLM_Number  racing number on the chin bar

Run inside Blender 5.2.2:
    import build_helmet; build_helmet.run()
"""

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

import glenn_common as G
from shader_dsl import NB

OWNER = "helmet"

# Grid resolution: 64 columns around, so the crown patch is 16 x 16.
COLS = 64
CAP_THETA = 80.0
# Fixed rows (degrees of elevation) down to -30, then rows that stretch
# to the bottom edge, which changes height around the helmet.
ROWS_FIXED = [80, 71, 62, 53, 44, 36, 29, 22, 16, 11, 6, 1, -5, -12, -20, -30]
ROWS_LOWER = 6

PORT_PHI = 61.875          # half-width of the eye port (11 columns)
PORT_THETA = (-5.0, 16.0)  # bottom and top rows of the eye port
VISOR_PHI = 73.125         # 13 columns
VISOR_THETA = (-12.0, 22.0)

# Size (metres). Origin is the wearer's eye line, centre of the head.
A = 0.136        # half width
B_FRONT = 0.150  # forward reach
B_BACK = 0.170   # rear reach
C_TOP = 0.158    # crown height above the eye line
C_LOW = 0.150    # depth below the eye line
SQUARE = 2.35    # superellipse exponent for flatter, faster sides


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def bottom_theta(phi):
    """Elevation of the bottom edge: low at the chin, raised behind the jaw,
    low again at the nape."""
    a = abs(phi)
    return -58.0 + 15.0 * math.exp(-((a - 100.0) / 32.0) ** 2) - 2.0 * smoothstep(140, 180, a)


def surface(phi, theta):
    """Point on the outer shell for azimuth phi (0 = front) and elevation."""
    p, t = math.radians(phi), math.radians(theta)
    sp, cp = math.sin(p), -math.cos(p)          # x, y of the azimuth direction
    # superellipse horizontal section: flatter sides, rounder front and back
    n = SQUARE
    k = (abs(sp) ** n + abs(cp) ** n) ** (-1.0 / n)
    b = B_FRONT if cp < 0 else B_BACK
    ct = math.cos(t)
    x = A * k * sp * ct
    y = b * k * cp * ct
    z = (C_TOP if t > 0 else C_LOW) * math.sin(t)
    # chin bar: push forward and slightly down at the front, below the port
    front = max(0.0, math.cos(p)) ** 2
    chin = smoothstep(-8.0, -40.0, theta) * front
    y -= 0.042 * chin
    z -= 0.012 * chin
    # slight taper of the lower sides toward the neck
    x *= 1.0 - 0.06 * smoothstep(-20.0, -55.0, theta) * (1 - front)
    return Vector((x, y, z))


def build_shell_mesh(name):
    """Return a bmesh of the whole closed-top shell (port not yet cut) and
    a lookup of the (col, row) faces."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    phis = [-180.0 + 360.0 * c / COLS for c in range(COLS)]

    rows = []        # rows[r][c] = BMVert
    thetas = []      # thetas[r][c]
    for r, th in enumerate(ROWS_FIXED):
        rows.append([bm.verts.new(surface(ph, th)) for ph in phis])
        thetas.append([th] * COLS)
    for i in range(1, ROWS_LOWER + 1):
        row, trow = [], []
        for ph in phis:
            th = -30.0 + (bottom_theta(ph) + 30.0) * i / ROWS_LOWER
            row.append(bm.verts.new(surface(ph, th)))
            trow.append(th)
        rows.append(row)
        thetas.append(trow)

    def uv_of(c, th):
        return ((phis[c] + 180.0) / 360.0, (th + 90.0) / 180.0)

    cells = {}
    for r in range(len(rows) - 1):
        for c in range(COLS):
            c2 = (c + 1) % COLS
            f = bm.faces.new((rows[r][c], rows[r][c2], rows[r + 1][c2], rows[r + 1][c]))
            cells[(c, r)] = f
            uvs = [uv_of(c, thetas[r][c]), uv_of(c2, thetas[r][c2]),
                   uv_of(c2, thetas[r + 1][c2]), uv_of(c, thetas[r + 1][c])]
            if c2 == 0:   # wrap the seam at the back
                uvs[1] = (uvs[1][0] + 1.0, uvs[1][1])
                uvs[2] = (uvs[2][0] + 1.0, uvs[2][1])
            for loop, coord in zip(f.loops, uvs):
                loop[uv].uv = coord

    # Crown: 16 x 16 Coons patch on the top ring, projected onto the shell.
    ring = rows[0]
    n = COLS // 4
    start = COLS // 8             # corner at phi = -135 degrees

    def bnd(b):
        return ring[(start + b) % COLS]

    grid = {}
    for i in range(n + 1):
        grid[(i, 0)] = bnd(i)
        grid[(n, i)] = bnd(n + i)
        grid[(i, n)] = bnd(3 * n - i)
        grid[(0, i)] = bnd((4 * n - i) % (4 * n))
    P = lambda i, j: grid[(i, j)].co.copy()
    for i in range(1, n):
        for j in range(1, n):
            u, v = i / n, j / n
            co = ((1 - v) * P(i, 0) + v * P(i, n) + (1 - u) * P(0, j) + u * P(n, j)
                  - ((1 - u) * (1 - v) * P(0, 0) + u * (1 - v) * P(n, 0)
                     + (1 - u) * v * P(0, n) + u * v * P(n, n)))
            # project onto the crown of the shell
            phi = math.degrees(math.atan2(co.x, -co.y))
            d = surface(phi, 0.0)
            horiz = math.hypot(co.x, co.y) / max(math.hypot(d.x, d.y), 1e-6)
            theta = math.degrees(math.acos(min(1.0, horiz)))
            grid[(i, j)] = bm.verts.new(surface(phi, theta))
    for i in range(n):
        for j in range(n):
            f = bm.faces.new((grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]))
            for loop in f.loops:
                co = loop.vert.co
                phi = math.degrees(math.atan2(co.x, -co.y))
                horiz = math.hypot(co.x, co.y) / max(math.hypot(*surface(phi, 0.0).xy), 1e-6)
                loop[uv].uv = ((phi + 180.0) / 360.0, (math.degrees(math.acos(min(1.0, horiz))) + 90.0) / 180.0)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm, cells, phis


def cell_center(phis, c, r):
    ph = phis[c] + 180.0 / COLS
    if r < len(ROWS_FIXED) - 1:
        th = 0.5 * (ROWS_FIXED[r] + ROWS_FIXED[r + 1])
    else:
        th = -40.0
    return ph, th


def shell():
    bm, cells, phis = build_shell_mesh("HLM_Shell")
    port = [f for (c, r), f in cells.items()
            if abs(cell_center(phis, c, r)[0]) < PORT_PHI
            and PORT_THETA[0] < cell_center(phis, c, r)[1] < PORT_THETA[1]]
    bmesh.ops.delete(bm, geom=port, context="FACES")
    mesh = bpy.data.meshes.new("HLM_Shell")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new("HLM_Shell", mesh)
    sol = obj.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.012
    sol.offset = -1.0
    sol.use_rim = True
    sol.use_even_offset = True
    sol.edge_crease_rim = 0.8
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 2
    return obj


def visor():
    bm, cells, phis = build_shell_mesh("HLM_Visor")
    keep = {f for (c, r), f in cells.items()
            if abs(cell_center(phis, c, r)[0]) < VISOR_PHI
            and VISOR_THETA[0] < cell_center(phis, c, r)[1] < VISOR_THETA[1]}
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f not in keep], context="FACES")
    # offset the visor just outside the shell
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * 0.004
    mesh = bpy.data.meshes.new("HLM_Visor")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new("HLM_Visor", mesh)
    sol = obj.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.0025
    sol.offset = 1.0
    sol.use_even_offset = True
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 2
    return obj


def boundary_curve(name, points, bevel):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = bevel
    cu.bevel_resolution = 3
    cu.use_fill_caps = True
    sp = cu.splines.new("NURBS")
    sp.points.add(len(points) - 1)
    for pt, co in zip(sp.points, points):
        pt.co = (co.x, co.y, co.z, 1.0)
    sp.use_cyclic_u = True
    sp.order_u = 3
    return bpy.data.objects.new(name, cu)


def gasket():
    pts = []
    lo, hi = PORT_THETA
    steps = 22
    for i in range(steps + 1):                       # top edge, left to right
        pts.append(surface(-PORT_PHI + 2 * PORT_PHI * i / steps, hi))
    for i in range(1, 6):                            # right side, down
        pts.append(surface(PORT_PHI, hi + (lo - hi) * i / 6))
    for i in range(steps + 1):                       # bottom edge, right to left
        pts.append(surface(PORT_PHI - 2 * PORT_PHI * i / steps, lo))
    for i in range(1, 6):                            # left side, up
        pts.append(surface(-PORT_PHI, lo + (hi - lo) * i / 6))
    return boundary_curve("HLM_Gasket", pts, 0.0045)


def trim():
    pts = [surface(-180.0 + 360.0 * i / 96, bottom_theta(-180.0 + 360.0 * i / 96)) for i in range(96)]
    pts = [p + Vector((0, 0, 0.002)) for p in pts]
    return boundary_curve("HLM_Trim", pts, 0.0075)


def pivots():
    bm = bmesh.new()
    for s in (1, -1):
        c = surface(s * 90.0, 4.0)
        ret = bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.020, radius2=0.017,
                                    depth=0.008)
        verts = ret["verts"]
        # orient along the shell normal (roughly X) and move into place
        bmesh.ops.rotate(bm, verts=verts, cent=(0, 0, 0),
                         matrix=Matrix.Rotation(math.radians(90 * s), 3, "Y"))
        bmesh.ops.translate(bm, verts=verts, vec=c + Vector((s * 0.006, 0, 0)))
    mesh = bpy.data.meshes.new("HLM_Pivots")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("HLM_Pivots", mesh)
    bev = obj.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments, bev.limit_method = 0.0015, 2, "ANGLE"
    return obj


def surface_frame(phi, theta, eps=0.5):
    """Point, outward normal, and unit tangents (along phi, along theta)."""
    p = surface(phi, theta)
    tp = (surface(phi + eps, theta) - surface(phi - eps, theta)).normalized()
    tt = (surface(phi, theta + eps) - surface(phi, theta - eps)).normalized()
    n = tp.cross(tt).normalized()
    if n.dot(p) < 0:
        n = -n
    return p, n, tp, tt


def spoiler():
    """Lip across the back of the crown with an upturned trailing edge."""
    bm = bmesh.new()
    span = 34.0                 # half-span in degrees around the back
    cols, theta0 = 20, 27.0
    section = [(0.0, -0.002), (0.25, 0.004), (0.6, 0.0075), (1.0, 0.0115),
               (1.0, 0.0085), (0.6, 0.0035), (0.25, 0.0005)]
    grid = []
    for i in range(cols + 1):
        t = i / cols
        phi = 180.0 - span + 2 * span * t
        base, n, tp, tt = surface_frame(phi, theta0)
        back = Vector((base.x, base.y, 0.0)).normalized()
        chord = 0.004 + 0.020 * smoothstep(0.0, 0.3, t) * smoothstep(1.0, 0.7, t)
        ring = [bm.verts.new(base + back * (chord * s) + n * h) for s, h in section]
        grid.append(ring)
    m = len(section)
    for i in range(cols):
        for j in range(m):
            j2 = (j + 1) % m
            bm.faces.new((grid[i][j], grid[i][j2], grid[i + 1][j2], grid[i + 1][j]))
    for ring in (grid[0], grid[-1]):
        bm.faces.new(ring)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("HLM_Spoiler")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new("HLM_Spoiler", mesh)
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 2
    return obj


def _ellipsoid_on_surface(name, phi, theta, size, sink=0.0, segs=(24, 12)):
    """A flattened ellipsoid laid on the shell: size = (along phi, along
    theta, along normal)."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs[0], v_segments=segs[1], radius=1.0)
    p, n, tp, tt = surface_frame(phi, theta)
    tt = n.cross(tp).normalized()
    M = Matrix((tp * size[0], tt * size[1], n * size[2])).transposed()
    for v in bm.verts:
        v.co = M @ v.co + p + n * (size[2] * sink)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    return bpy.data.objects.new(name, mesh)


def intake():
    pod = _ellipsoid_on_surface("HLM_Intake", 0.0, 56.0, (0.017, 0.056, 0.0105), sink=0.05)
    mouth = _ellipsoid_on_surface("HLM_IntakeMouth", 0.0, 41.0, (0.012, 0.0035, 0.0065), sink=0.35)
    return pod, mouth


def vents():
    bm = bmesh.new()
    for theta, width in ((-41.0, 0.020), (-45.5, 0.017), (-50.0, 0.013)):
        ob = _ellipsoid_on_surface("_tmp", 0.0, theta, (width, 0.0028, 0.002), sink=0.0, segs=(20, 8))
        bm.from_mesh(ob.data)
        mesh = ob.data
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(mesh)
    mesh = bpy.data.meshes.new("HLM_Vents")
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    return bpy.data.objects.new("HLM_Vents", mesh)


def number(shell_obj, text="61"):
    """Racing number laid on the front of the chin bar (built-in font,
    thickened and sheared for a bold italic look)."""
    cu = bpy.data.curves.new("HLM_NumberText", "FONT")
    cu.body = text
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    cu.size = 0.052
    cu.offset = 0.0013
    cu.shear = 0.18
    cu.resolution_u = 4
    tmp = bpy.data.objects.new("HLM_NumberText", cu)
    bpy.context.scene.collection.objects.link(tmp)
    dg = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp)
    bpy.data.curves.remove(cu)
    mesh.name = "HLM_Number"
    # long glyph faces need interior vertices so they can follow the curved
    # shell instead of cutting through it
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    for _ in range(2):
        long_edges = [e for e in bm.edges if e.calc_length() > 0.004]
        if long_edges:
            bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=2, use_grid_fill=True)
            bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    p, n, tp, tt = surface_frame(0.0, -27.5)
    up = n.cross(tp).normalized()
    M = Matrix((tp, up, n)).transposed()
    for v in mesh.vertices:
        v.co = M @ v.co + p + n * 0.004
    obj = bpy.data.objects.new("HLM_Number", mesh)
    obj.location = (0.0, 0.0, 0.0)
    sw = obj.modifiers.new("Conform", "SHRINKWRAP")
    sw.target = shell_obj
    sw.wrap_method = "PROJECT"
    sw.use_negative_direction = True
    sw.use_positive_direction = True
    sw.wrap_mode = "ON_SURFACE"
    sw.offset = 0.0008
    return obj


def livery_material():
    """Cream shell with an indigo crown band and chin wrap, edged by amber
    pinstripes. Written in shell object space; bake to UVs for the game."""
    P = G.PALETTE
    mat = bpy.data.materials.get("MAT_HelmetShell") or bpy.data.materials.new("MAT_HelmetShell")
    nb = NB(mat)
    nb.clear()
    x, y, z = nb.xyz(nb.texcoord("Object"))
    ax = nb.abs(x)
    # crown band: widens toward the back, kept off the face below the port top
    bw = 0.036 + 0.014 * nb.smoothstep(-0.12, 0.16, y)
    on_band_area = nb.mx(nb.smoothstep(0.040, 0.046, z), nb.smoothstep(-0.02, 0.02, y))
    band = (1.0 - nb.aa_step(bw, ax, 0.0006)) * on_band_area
    pin_in, pin_out = bw + 0.005, bw + 0.011
    band_pin = nb.aa_step(pin_in, ax, 0.0006) * (1.0 - nb.aa_step(pin_out, ax, 0.0006)) * on_band_area
    # chin wrap: everything below a line rising from the chin to the nape
    zl = -0.072 + 0.17 * (y + 0.18)
    wrap = 1.0 - nb.aa_step(zl, z, 0.0008)
    wrap_pin = nb.aa_step(zl + 0.005, z, 0.0008) * (1.0 - nb.aa_step(zl + 0.011, z, 0.0008))
    indigo = nb.mx(band, wrap)
    amber = nb.mn(nb.mx(band_pin, wrap_pin), 1.0 - indigo)
    col = nb.layer(P["cream"], [(indigo, P["indigo"]), (amber, P["amber"])])
    flake = nb.noise(nb.texcoord("Object"), 900.0, detail=1.0)
    bsdf = nb.principled(base=col, rough=0.32 + 0.06 * flake, coat=1.0, coat_rough=0.04)
    nb.output(bsdf)
    return mat


def materials():
    P = G.PALETTE
    return {
        "shell": livery_material(),
        "number": G.principled("MAT_HelmetNumber", P["amber"], roughness=0.3,
                               **{"Coat Weight": 1.0, "Coat Roughness": 0.04}),
        "spoiler": G.principled("MAT_HelmetIndigo", P["indigo"], roughness=0.3,
                                **{"Coat Weight": 1.0, "Coat Roughness": 0.04}),
        "inlet": G.principled("MAT_HelmetInlet", (0.004, 0.004, 0.005), roughness=0.6),
        "visor": G.principled("MAT_Visor", (0.02, 0.02, 0.025), roughness=0.03,
                              **{"Transmission Weight": 0.85, "IOR": 1.49}),
        "rubber": G.principled("MAT_Rubber", (0.012, 0.012, 0.014), roughness=0.75),
        "metal": G.principled("MAT_PivotMetal", (0.55, 0.55, 0.58), roughness=0.25, metallic=1.0),
    }


def run(location=(0.95, 0.35, 1.05 + 0.15), rotation_z=-35.0):
    """Build the helmet. By default it sits on the display stand."""
    G.ensure_collections()
    G.clear_owned(OWNER)
    mats = materials()
    root = bpy.data.objects.new("HLM_Root", None)
    root.empty_display_type = "SPHERE"
    root.empty_display_size = 0.05
    G.link(root, "GLN_Helmet", OWNER)
    sh = shell()
    pod, mouth = intake()
    parts = [(sh, "shell"), (visor(), "visor"), (gasket(), "rubber"),
             (trim(), "rubber"), (pivots(), "metal"), (spoiler(), "spoiler"),
             (pod, "shell"), (mouth, "inlet"), (vents(), "inlet")]
    for obj, mat in parts:
        G.link(obj, "GLN_Helmet", OWNER)
        obj.data.materials.append(mats[mat])
        obj.parent = root
    num = number(sh)
    G.link(num, "GLN_Helmet", OWNER)
    num.data.materials.append(mats["number"])
    num.parent = root
    parts.append((num, "number"))
    root.location = location
    root.rotation_euler = (0.0, 0.0, math.radians(rotation_z))
    return {"objects": [o.name for o, _ in parts],
            "shell_faces": len(parts[0][0].data.polygons)}
