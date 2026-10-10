"""Materials for the GLeNN driver: skin, eyes, hair, suit, gloves, boots,
helmet, visor and studio. All procedural, no image textures, so the .blend
is self-contained."""

import math
import bpy
from shader_dsl import NB
import shapes_head as H

# palette (linear RGB) ---------------------------------------------------------
GRAPHITE = (0.040, 0.042, 0.046)
GRAPHITE_LIGHT = (0.075, 0.078, 0.084)
BLACK = (0.010, 0.010, 0.011)
WHITE = (0.80, 0.80, 0.78)
ELECTRIC = (0.005, 0.20, 0.95)        # restrained electric blue
ELECTRIC_HOT = (0.02, 0.32, 1.0)


def _new(name):
    m = bpy.data.materials.get(name)
    if m:
        bpy.data.materials.remove(m)
    m = bpy.data.materials.new(name)
    nb = NB(m)
    nb.clear()
    return m, nb


def _ell(nb, p, c, r):
    """Normalised ellipsoid radius |(p-c)/r| as a float expression."""
    q = (p - nb.vec(*c)) * nb.vec(1.0 / r[0], 1.0 / r[1], 1.0 / r[2])
    return q.length()


def _sym(nb, p):
    x, y, z = nb.xyz(p)
    return nb.vec(nb.abs(x), y, z), (x, y, z)


# -------------------------------------------------------------------------------
def clay(name="GLN_Clay", color=(0.40, 0.36, 0.33), rough=0.55):
    m, nb = _new(name)
    p = nb.texcoord("Object")
    var = nb.noise(p, 6.0, detail=2)
    base = nb.mix(var * 0.25, color, (color[0] * 1.15, color[1] * 1.12, color[2] * 1.1))
    b = nb.principled(base=base, rough=rough, spec=0.35)
    nb.output(b)
    return m


def skin(name="GLN_Skin"):
    m, nb = _new(name)
    p0 = nb.texcoord("Object")
    # undo the stylised lower-face lift so masks sit on the sculpted features
    x0, y0, z0 = nb.xyz(p0)
    p = nb.vec(x0, y0, z0 - H.LF_LIFT * nb.clamp01((H.LF_TOP - z0) * (1.0 / H.LF_SPAN)))
    ps, (x, y, z) = _sym(nb, p)
    base = (0.50, 0.300, 0.220)
    warm = (0.62, 0.27, 0.20)
    cool = (0.42, 0.33, 0.31)
    light = (0.60, 0.39, 0.29)
    # painterly low-frequency variation
    paint = nb.noise(p, 18.0, detail=3, rough=0.55)
    paint2 = nb.noise(p, 55.0, detail=2, rough=0.5)
    red_cheek = 1.0 - nb.smoothstep(0.55, 1.15, _ell(nb, ps, (0.047, -0.082, -0.030), (0.026, 0.03, 0.026)))
    red_nose = 1.0 - nb.smoothstep(0.5, 1.2, _ell(nb, p, (0.0, -0.112, -0.036), (0.016, 0.02, 0.016)))
    red_ear = 1.0 - nb.smoothstep(0.5, 1.2, _ell(nb, ps, (0.077, 0.014, -0.012), (0.02, 0.025, 0.035)))
    forehead = nb.smoothstep(0.03, 0.07, z) * (1.0 - nb.smoothstep(-0.03, 0.04, y))
    lower = nb.smoothstep(-0.050, -0.080, z) * (1.0 - nb.smoothstep(-0.03, 0.03, y))
    jaw_side = nb.smoothstep(-0.03, -0.075, z) * nb.smoothstep(0.035, 0.06, nb.abs(x)) * (1.0 - nb.smoothstep(0.0, 0.03, y))
    under_eye = 1.0 - nb.smoothstep(0.5, 1.2, _ell(nb, ps, (0.034, -0.086, -0.013), (0.016, 0.01, 0.007)))
    lip_u = _ell(nb, ps, (0.006, -0.0955, -0.0648), (0.0160, 0.0080, 0.0052))
    lip_l = _ell(nb, ps, (0.005, -0.0945, -0.0752), (0.0145, 0.0082, 0.0062))
    lips = 1.0 - nb.smoothstep(0.85, 1.05, nb.mn(lip_u, lip_l))
    col = nb.mix(paint * 0.5, base, (0.53, 0.32, 0.24))
    col = nb.layer(col, [
        (forehead * 0.35, light),
        (red_cheek * 0.45, warm),
        (red_nose * 0.40, warm),
        (red_ear * 0.55, warm),
        (lower * 0.30, cool),
        (jaw_side * 0.22, cool),
        (under_eye * 0.25, (0.48, 0.30, 0.28)),
        (paint2 * 0.10, (0.66, 0.40, 0.30)),
        (lips * 0.70, (0.50, 0.24, 0.20)),
    ])
    rough = 0.47 - forehead * 0.07 - red_nose * 0.08 - lips * 0.12 + paint2 * 0.06
    # tertiary: subtle pores and micro breakup
    pores = nb.voronoi(p, 900.0, out="Distance")
    micro = nb.noise(p, 1600.0, detail=1)
    h = pores * 0.6 + micro * 0.4
    nrm = nb.bump(h, strength=0.06, distance=0.0002)
    b = nb.principled(base=col, rough=rough, sss=0.18, sss_radius=(1.0, 0.38, 0.20), sss_scale=0.0035,
                      spec=0.42, normal=nrm, sheen=0.04, coat=0.0)
    nb.output(b)
    return m


def eye(name="GLN_Eye", iris=(0.10, 0.19, 0.14)):
    """Eyeball shaded in object space; local -Y is the gaze direction."""
    m, nb = _new(name)
    p = nb.texcoord("Object")
    x, y, z = nb.xyz(p)
    rr = nb.sqrt(x * x + z * z) * (1.0 / H.EYE_R)
    front = nb.smoothstep(0.0, -0.002, y)
    ri = H.IRIS_R / H.EYE_R
    iris_m = (1.0 - nb.smoothstep(ri - 0.015, ri + 0.01, rr)) * front
    pupil = (1.0 - nb.smoothstep(0.155, 0.185, rr)) * front
    limbal = nb.smoothstep(ri - 0.12, ri, rr) * iris_m
    ang = nb.atan2(z, x)
    fib_coord = nb.vec(ang * 6.0, rr * 9.0, 0.0)
    fib = nb.noise(fib_coord, 3.0, detail=4, rough=0.6)
    fib2 = nb.noise(nb.vec(ang * 14.0, rr * 4.0, 1.0), 2.0, detail=2)
    iris_col = nb.mix(fib, (iris[0] * 0.6, iris[1] * 0.6, iris[2] * 0.6), (iris[0] * 1.9, iris[1] * 1.6, iris[2] * 1.3))
    iris_col = nb.mix(nb.smoothstep(0.35, 0.18, rr) * 0.7, iris_col, (0.20, 0.13, 0.05))   # warm collarette
    iris_col = nb.mix(fib2 * 0.3, iris_col, (0.03, 0.04, 0.035))
    iris_col = nb.mix(limbal * 0.85, iris_col, (0.01, 0.012, 0.012))
    sclera_noise = nb.noise(p, 900.0, detail=2)
    sclera = nb.mix(nb.smoothstep(0.55, 1.0, rr) * 0.5 + sclera_noise * 0.1, (0.78, 0.74, 0.70), (0.62, 0.42, 0.38))
    col = nb.mix(iris_m, sclera, iris_col)
    col = nb.mix(pupil, col, (0.004, 0.004, 0.004))
    # gloss + subtle iris depth via bump
    nrm = nb.bump(iris_m * fib * 0.6 - pupil * 0.4, strength=0.15, distance=0.0003)
    b = nb.principled(base=col, rough=0.35, spec=0.5, coat=1.0, coat_rough=0.015, normal=nrm, sss=0.0)
    nb.output(b)
    return m


def hair(name="GLN_Hair", color=(0.038, 0.024, 0.016)):
    m, nb = _new(name)
    p = nb.texcoord("Object")
    x, y, z = nb.xyz(p)
    # strands run front-to-back on top and fall down the front: fine across x only
    streak_coord = nb.vec(x * 380.0, y * 30.0, z * 30.0)
    streak_top = nb.noise(streak_coord, 1.0, detail=3, rough=0.6)
    # faded sides: short strands run down and back
    side_coord = nb.vec((y * 0.87 + z * 0.5) * 380.0, x * 30.0, (y * 0.5 - z * 0.87) * 30.0)
    streak_side = nb.noise(side_coord, 1.0, detail=3, rough=0.6)
    nx, _ny, _nz = nb.xyz(nb.texcoord("Normal"))
    streak = nb.mixf(nb.smoothstep(0.45, 0.8, nb.abs(nx)), streak_top, streak_side)
    clumps = nb.noise(p, 90.0, detail=2)
    col = nb.mix(streak * 0.55 + clumps * 0.2, (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7),
                 (color[0] * 2.4, color[1] * 2.1, color[2] * 1.8))
    # sun-faded tips on top
    col = nb.mix(nb.smoothstep(0.09, 0.13, z) * 0.25, col, (0.075, 0.052, 0.034))
    nrm = nb.bump(streak, strength=0.35, distance=0.0004)
    b = nb.principled(base=col, rough=0.48 + clumps * 0.1, spec=0.45, normal=nrm, sheen=0.35,
                      sheen_rough=0.35, sheen_tint=(0.6, 0.5, 0.4))
    nb.output(b)
    return m


def backdrop(name="GLN_Backdrop"):
    m, nb = _new(name)
    p = nb.texcoord("Object")
    n = nb.noise(p, 2.0, detail=2)
    col = nb.mix(n * 0.4, (0.014, 0.014, 0.016), (0.022, 0.022, 0.025))
    b = nb.principled(base=col, rough=0.75, spec=0.2)
    nb.output(b)
    return m


# -------------------------------------------------------------------------------
# racing suit (panels follow the fabric through the 'rest' attribute)
# -------------------------------------------------------------------------------

def _seg_dist2(nb, px, pz, a, b):
    """Distance from (px, pz) to segment a-b in a 2D projection, plus t."""
    ax, az = a
    bx, bz = b
    dx, dz = bx - ax, bz - az
    L2 = dx * dx + dz * dz
    t = nb.clamp01(((px - ax) * dx + (pz - az) * dz) * (1.0 / L2))
    qx = px - ax - t * dx
    qz = pz - az - t * dz
    return nb.sqrt(qx * qx + qz * qz), t


def _stitch(nb, dist, along, offset=0.0028, period=0.0042, width=0.00045):
    """Dashed stitch line running parallel to a seam."""
    line = 1.0 - nb.smoothstep(width * 0.6, width * 1.4, nb.abs(nb.abs(dist) - offset))
    dash = nb.smoothstep(0.25, 0.4, nb.abs(nb.fract(along * (1.0 / period)) - 0.5) * 2.0)
    return line * dash


def suit(name="GLN_Suit"):
    m, nb = _new(name)
    p = nb.attribute("rest")
    x, y, z = nb.xyz(p)
    ax = nb.abs(x)
    deg = 57.29578
    th = nb.atan2(x, 0.004 - y) * deg          # 0 front, +-90 sides, 180 back
    ath = nb.abs(th)
    # zones ----------------------------------------------------------------
    upper = nb.smoothstep(0.93, 0.95, z)
    arm_zone = nb.mx(nb.smoothstep(0.170, 0.176, ax) * (1.0 - nb.smoothstep(0.99, 1.0, z)),
                     nb.smoothstep(0.150, 0.156, ax) * nb.smoothstep(0.99, 1.0, z) * (1.0 - nb.smoothstep(1.35, 1.37, z)))
    # raglan sleeve seam: from the collar down to the armpit
    rag = (ax - 0.070) * 0.92 + (1.455 - z) * -0.39 - 0.0      # >0 towards the sleeve
    rag_d = (ax - (0.070 + (1.455 - z) * 0.43))
    sleeve = nb.mx(arm_zone, nb.aa_step(0.0, rag_d, 0.0008) * nb.smoothstep(1.30, 1.31, z))
    torso = upper * (1.0 - sleeve)
    leg = (1.0 - upper)
    # side panels on the torso (black stretch)
    side_d = nb.mx(72.0 - ath, ath - 116.0) * 0.0021
    side_d = nb.mx(side_d, nb.mx(0.985 - z, z - 1.34))
    side = (1.0 - nb.aa_step(0.0, side_d, 0.0007)) * torso
    # inner sleeve panel
    axis_x = 0.180 + (1.392 - z) * 0.146
    inner_d = (ax - axis_x) + 0.016
    inner = (1.0 - nb.aa_step(0.0, inner_d, 0.0007)) * sleeve
    # outer leg stripe
    leg_axis = 0.088 + (0.925 - z) * 0.036
    leg_r = 0.040 + (z - 0.09) * 0.045
    outer_d = (ax - leg_axis) - leg_r * 0.55
    legstripe = nb.aa_step(0.0, outer_d, 0.0007) * leg * (1.0 - nb.smoothstep(0.88, 0.90, z))
    # belt and collar
    belt = (1.0 - nb.aa_step(0.026, nb.abs(z - 1.060), 0.0005)) * (1.0 - sleeve)
    collar = nb.smoothstep(1.428, 1.432, z) * (1.0 - nb.smoothstep(0.110, 0.115, ax)) * (1.0 - sleeve)
    # epaulettes
    t = nb.clamp01((ax - 0.080) * (1.0 / 0.105))
    yc = 0.016 - 0.008 * t
    ep_d = nb.mx(nb.abs(y - yc) - 0.020, nb.abs(ax - 0.130) - 0.052)
    ep_d = nb.mx(ep_d, 1.365 - z)
    epaulette = 1.0 - nb.aa_step(0.0, ep_d, 0.0006)
    # GLeNN blade: asymmetric diagonal from the left shoulder to the right hip
    front = 1.0 - nb.smoothstep(-0.01, 0.01, y)
    bd, bt = _seg_dist2(nb, x, z, (0.118, 1.425), (-0.095, 1.115))
    bw = 0.034 - 0.022 * bt
    blade_d = bd - bw
    blade = (1.0 - nb.aa_step(0.0, blade_d, 0.0007)) * front * torso * (1.0 - side)
    blue_line = (1.0 - nb.aa_step(0.0016, nb.abs(blade_d - 0.0065), 0.0005)) * front * torso * (1.0 - side)
    back = nb.smoothstep(-0.01, 0.01, y)
    kd, kt = _seg_dist2(nb, x, z, (-0.120, 1.405), (0.085, 1.180))
    kw = 0.030 - 0.020 * kt
    bblade_d = kd - kw
    bblade = (1.0 - nb.aa_step(0.0, bblade_d, 0.0007)) * back * torso * (1.0 - side)
    bblue = (1.0 - nb.aa_step(0.0016, nb.abs(bblade_d - 0.0065), 0.0005)) * back * torso * (1.0 - side)
    # electric pinstripes along the side-panel front edge and sleeve seam
    side_edge = (1.0 - nb.aa_step(0.0012, nb.abs((72.0 - ath) * 0.0021 + 0.0035), 0.0004)) * torso * \
        nb.smoothstep(0.985, 0.99, z) * (1.0 - nb.smoothstep(1.335, 1.34, z))
    sleeve_pin = (1.0 - nb.aa_step(0.0011, nb.abs(ax - axis_x - 0.040), 0.0004)) * sleeve * (1.0 - nb.smoothstep(1.30, 1.32, z)) \
        * nb.smoothstep(0.92, 0.93, z)
    cuff_pin = (1.0 - nb.aa_step(0.0011, nb.abs(z - 0.915 + (ax - 0.25) * 0.15), 0.0004)) * sleeve
    leg_pin = (1.0 - nb.aa_step(0.0011, nb.abs(outer_d + 0.004), 0.0004)) * leg * (1.0 - nb.smoothstep(0.88, 0.90, z)) \
        * nb.smoothstep(0.17, 0.18, z)
    belt_pin = (1.0 - nb.aa_step(0.0011, nb.abs(z - 1.0315), 0.0004)) * (1.0 - sleeve)
    collar_pin = (1.0 - nb.aa_step(0.0012, nb.abs(z - 1.474 - y * 0.12), 0.0004)) * collar
    blue = nb.mx(blue_line, bblue, side_edge, sleeve_pin, cuff_pin, leg_pin, belt_pin, collar_pin)
    # stitching along every panel edge
    st = nb.mx(_stitch(nb, side_d, z), _stitch(nb, inner_d, z), _stitch(nb, outer_d, z),
               _stitch(nb, blade_d, z + x * 0.5) * front * torso, _stitch(nb, bblade_d, z - x * 0.5) * back * torso,
               _stitch(nb, rag_d, z) * nb.smoothstep(1.30, 1.31, z),
               _stitch(nb, nb.abs(z - 1.060) - 0.026, th * 0.0021) * (1.0 - sleeve),
               _stitch(nb, ep_d, ax, offset=0.0024))
    # colour ------------------------------------------------------------------
    col = nb.layer(GRAPHITE, [
        (sleeve * 0.6, GRAPHITE_LIGHT),
        (nb.mx(side, inner, legstripe, belt, collar, epaulette), BLACK),
        (nb.mx(blade, bblade), WHITE),
        (blue, ELECTRIC_HOT),
        (st * 0.55, (0.20, 0.20, 0.21)),
    ])
    # fabric: fine twill weave, stretch panels coarser
    pw = p * 2600.0
    wx, wy, wz = nb.xyz(pw)
    weave = nb.sin(wx + wz) * nb.sin(wy - wz * 0.7) * 0.5 + 0.5
    rib = nb.sin(z * 2200.0) * 0.5 + 0.5
    stretch = nb.mx(side, inner)
    fabric = nb.mixf(stretch, weave, rib)
    seam_groove = nb.mn(nb.abs(side_d), nb.abs(inner_d), nb.abs(outer_d), nb.abs(rag_d) + (1.0 - nb.smoothstep(1.30, 1.31, z)),
                        nb.abs(blade_d) + (1.0 - front * torso), nb.abs(bblade_d) + (1.0 - back * torso),
                        nb.abs(nb.abs(z - 1.060) - 0.026))
    groove = nb.smoothstep(0.0, 0.0012, seam_groove)
    zip_line = nb.smoothstep(0.0, 0.0008, ax) + (1.0 - nb.smoothstep(-0.01, 0.0, -y))
    height = fabric * 0.25 + groove * 0.8 + nb.clamp01(zip_line) * 0.6 - st * 0.25
    nrm = nb.bump(height, strength=0.35, distance=0.0006)
    rough = 0.80 - nb.mx(blade, bblade) * 0.12 - blue * 0.2 + stretch * 0.06
    bsdf = nb.principled(base=col, rough=rough, spec=0.32, sheen=0.45, sheen_rough=0.45,
                         sheen_tint=(0.7, 0.75, 0.85), normal=nrm)
    nb.output(bsdf)
    return m


def glove(name="GLN_Glove"):
    m, nb = _new(name)
    p = nb.attribute("rest")
    u, v, w = nb.xyz(p)
    back = nb.smoothstep(0.004, 0.008, w)
    palm = 1.0 - nb.smoothstep(-0.010, -0.006, w)
    guard_d = _ell(nb, p, (0.0894, 0.0, 0.0120), (0.0170, 0.0378, 0.0050)) - 1.0
    guard = 1.0 - nb.aa_step(0.0, guard_d, 0.02)
    guard_pin = (1.0 - nb.aa_step(0.05, nb.abs(guard_d - 0.12), 0.02)) * back
    cuff = 1.0 - nb.smoothstep(-0.002, 0.002, u + 0.004)
    strap = (1.0 - nb.aa_step(0.0115, nb.abs(u + 0.022), 0.0005)) * nb.smoothstep(0.006, 0.012, w + 0.03)
    strap_pin = (1.0 - nb.aa_step(0.0010, nb.abs(u + 0.0345), 0.0004))
    col = nb.layer(BLACK, [
        (back * 0.8, GRAPHITE),
        (guard, GRAPHITE_LIGHT),
        (strap, WHITE),
        (nb.mx(guard_pin, strap_pin), ELECTRIC_HOT),
        (cuff * (1.0 - strap) * 0.7, (0.02, 0.02, 0.022)),
    ])
    dots = 1.0 - nb.smoothstep(0.18, 0.25, nb.voronoi(p, 520.0, out="Distance", randomness=0.0))
    leather = nb.noise(p, 900.0, detail=3)
    height = leather * 0.3 + dots * palm * 0.8
    nrm = nb.bump(height, strength=0.25, distance=0.0005)
    rough = 0.62 - palm * dots * 0.25 - guard * 0.15
    b = nb.principled(base=col, rough=rough, spec=0.35, sheen=0.25, sheen_rough=0.5, normal=nrm)
    nb.output(b)
    return m


def boot(name="GLN_Boot"):
    m, nb = _new(name)
    p = nb.attribute("rest")
    x, y, z = nb.xyz(p)
    sole = 1.0 - nb.smoothstep(-0.082, -0.0805, z)
    # compact heel counter: low on the back of the boot only
    low = 1.0 - nb.aa_step(-0.034, z, 0.0006)
    edge = y + 0.030 - (z + 0.09) * 0.45
    heel = (1.0 - nb.aa_step(0.0, edge, 0.0006)) * (1.0 - sole) * low
    pin = (1.0 - nb.aa_step(0.0010, nb.abs(edge - 0.006), 0.0004)) * (1.0 - sole) * (1.0 - nb.aa_step(-0.028, z, 0.0006))
    strap = (1.0 - nb.aa_step(0.0095, nb.abs(z + 0.012), 0.0005))
    col = nb.layer(BLACK, [
        (heel, WHITE), (pin, ELECTRIC_HOT), (strap * 0.8, GRAPHITE), (sole, (0.03, 0.03, 0.032)),
    ])
    nrm = nb.bump(nb.noise(p, 700.0, detail=3) * 0.4 + sole * nb.wave(p, 300.0, direction="Y") * 0.5,
                  strength=0.2, distance=0.0005)
    b = nb.principled(base=col, rough=0.55 - heel * 0.15 + sole * 0.25, spec=0.4, normal=nrm, sheen=0.2)
    nb.output(b)
    return m


# -------------------------------------------------------------------------------
# helmet: white composite, graphite blade, electric-blue precision lines
# -------------------------------------------------------------------------------

HELMET_WHITE = (0.82, 0.82, 0.80)


def helmet(name="GLN_Helmet"):
    m, nb = _new(name)
    p = nb.texcoord("Object")
    n_obj = nb.texcoord("Normal")
    x, y, z = nb.xyz(p)
    ax = nb.abs(x)
    deg = 57.29578
    th = nb.atan2(x, 0.02 - y) * deg
    ath = nb.abs(th)
    # interior lining (faces pointing into the shell)
    c = nb.vec(0.0, 0.015, -0.01)
    inner = 1.0 - nb.smoothstep(-0.15, 0.15, ((p - c) * 10.0).dot(n_obj))
    # graphite speed blade: sweeps up from the chin to the rear
    line_z = -0.050 + (y + 0.12) * 0.42 - (ath - 90.0) * 0.0
    blade_d = z - line_z
    blade = 1.0 - nb.aa_step(0.0, blade_d, 0.0007)
    blade_pin = 1.0 - nb.aa_step(0.0014, nb.abs(blade_d - 0.0065), 0.0005)
    # second, thinner blue line (control)
    blade_pin2 = 1.0 - nb.aa_step(0.0007, nb.abs(blade_d - 0.0120), 0.0004)
    # top spine: graphite stripe with a precise blue centre line
    top = nb.smoothstep(0.075, 0.085, z)
    spine_d = ax - (0.020 - (y + 0.05) * 0.04)
    spine = (1.0 - nb.aa_step(0.0, spine_d, 0.0006)) * top
    spine_pin = (1.0 - nb.aa_step(0.0009, ax, 0.0004)) * top
    spine_edge = (1.0 - nb.aa_step(0.0008, nb.abs(spine_d - 0.0040), 0.0004)) * top
    # forward chevrons on the sides (speed)
    chev = 0.0
    for i, y0 in enumerate((0.018, 0.040, 0.062)):
        cd = (y - y0) - nb.abs(z - 0.030) * 0.85
        stroke = (1.0 - nb.aa_step(0.0038 - i * 0.0006, nb.abs(cd), 0.0005)) * (1.0 - nb.smoothstep(0.024, 0.026, nb.abs(z - 0.030)))
        chev = nb.mx(chev, stroke * nb.smoothstep(70.0, 80.0, ath) * (1.0 - nb.smoothstep(150.0, 160.0, ath)) * (1.0 - i * 0.25))
    # precision ticks above the eye-port
    tick_band = (1.0 - nb.aa_step(0.0035, nb.abs(z - 0.064), 0.0004)) * (1.0 - nb.smoothstep(38.0, 42.0, ath))
    ticks = tick_band * (1.0 - nb.smoothstep(0.12, 0.2, nb.abs(nb.fract(th * 0.5) - 0.5) * 2.0))
    long_ticks = (1.0 - nb.aa_step(0.0060, nb.abs(z - 0.066), 0.0004)) * (1.0 - nb.smoothstep(0.03, 0.06, nb.abs(nb.fract(th * 0.1) - 0.5) * 2.0)) \
        * (1.0 - nb.smoothstep(38.0, 42.0, ath))
    # chin tip in electric blue
    chin = (1.0 - nb.aa_step(0.0, nb.mx(ax - 0.020, z + 0.105, y + 0.135), 0.0008))
    graphite_m = nb.mx(blade, spine)
    blue_m = nb.mx(blade_pin, blade_pin2 * (1.0 - blade), spine_pin, spine_edge, chev * 0.0, chin * 0.0)
    col = nb.layer(HELMET_WHITE, [
        (graphite_m, GRAPHITE_LIGHT),
        (chev, GRAPHITE),
        (nb.mx(ticks, long_ticks) * (1.0 - graphite_m), GRAPHITE),
        (blue_m, ELECTRIC_HOT),
        (chin * blade, ELECTRIC),
        (inner, (0.012, 0.012, 0.013)),
    ])
    flake = nb.noise(p, 4000.0, detail=1)
    metal = graphite_m * 0.55 * (1.0 - inner)
    scratches = nb.noise(nb.vec(x * 40.0, y * 1800.0, z * 40.0), 1.0, detail=2)
    coat_rough = 0.025 + nb.smoothstep(0.62, 0.7, scratches) * 0.05
    coat_n = nb.bump(nb.smoothstep(0.62, 0.7, scratches), strength=0.05, distance=0.0002)
    rough = 0.32 + flake * 0.08 + inner * 0.5
    b = nb.principled(base=col, rough=rough, metal=metal, spec=0.5, coat=(1.0 - inner) * 1.0,
                      coat_rough=coat_rough, coat_normal=coat_n)
    nb.output(b)
    return m


def visor(name="GLN_Visor"):
    m, nb = _new(name)
    b = nb.principled(base=(0.030, 0.032, 0.040), rough=0.012, metal=0.0, trans=1.0, ior=1.52, spec=0.75,
                      thin_film=420.0, thin_film_ior=1.50)
    nb.output(b)
    return m


def metal(name="GLN_Metal", color=(0.55, 0.56, 0.58), rough=0.22):
    m, nb = _new(name)
    p = nb.texcoord("Object")
    brushed = nb.noise(nb.vec(nb.xyz(p)[0] * 3000.0, 0.0, 0.0), 1.0, detail=2)
    b = nb.principled(base=color, rough=rough + brushed * 0.08, metal=1.0, spec=0.5)
    nb.output(b)
    return m


def rubber(name="GLN_Rubber", color=(0.015, 0.015, 0.016)):
    m, nb = _new(name)
    p = nb.texcoord("Object")
    b = nb.principled(base=color, rough=0.7, spec=0.3, normal=nb.bump(nb.noise(p, 600.0), strength=0.1, distance=0.0004))
    nb.output(b)
    return m


def decal(name="GLN_Decal", color=WHITE, rough=0.45, emission=None):
    m, nb = _new(name)
    b = nb.principled(base=color, rough=rough, spec=0.4, coat=0.0)
    nb.output(b)
    return m
