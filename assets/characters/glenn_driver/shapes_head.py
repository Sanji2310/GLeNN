"""Head, eyes, ears, neck, brows and hair for the GLeNN driver.

Head-local space: origin on the eye line at the face midline, +X is the
character's left, -Y is forward (the face looks down -Y), +Z is up.
Units are metres. The design is original: it is assembled from primary
forms (skull, jaw, cheekbones, neck), secondary forms (brow, lids, nose,
lips, ears, planes) and a small amount of tertiary breakup.
"""

import numpy as np
from sdf_core import (local_op, sd_loft, v3, norm, rot_x, rot_y, rot_z, rot_euler, length,
                      sd_sphere, sd_ellipsoid, sd_capsule, sd_round_cone,
                      sd_box, sd_ellipse2, sd_plane, smin, smax, ssub,
                      smoothstep, value_noise)

# eye layout ---------------------------------------------------------------
EYE_R = 0.0148            # enlarged for the stylised, animated read
EYE_X = 0.0345
EYE_Y = -0.0690
EYE_Z = 0.0
IRIS_R = 0.0080


def eye_center(side):
    return v3(side * EYE_X, EYE_Y, EYE_Z)


def mirror_x(P):
    Q = P.copy()
    Q[:, 0] = np.abs(Q[:, 0])
    return Q


def _lid_planes(side):
    """Return (point, normal) pairs for the upper / lower lid margins."""
    E = eye_center(side)
    # focused, slightly narrowed eyes: upper lid sits on the top of the iris,
    # lower lid is raised a touch (controlled tension, not a squint)
    up_p = E + v3(0, -EYE_R, 0.0050)
    lo_p = E + v3(0, -EYE_R, -0.0062)
    tilt = 4.0 * side           # lateral canthus slightly higher
    Ru = rot_y(-tilt)
    n_up = Ru @ norm(v3(0, 0.62, 1.0))
    n_lo = Ru @ norm(v3(0, 0.55, -1.0))
    return (up_p, n_up), (lo_p, n_lo)


def eye_aperture_sdf(P, side):
    """Region (<0) visible between the lids, used for masks."""
    (up_p, n_up), (lo_p, n_lo) = _lid_planes(side)
    a = -((P - up_p) @ n_up)
    b = -((P - lo_p) @ n_lo)
    return np.maximum(a, b)


def _eyelids(P, side):
    E = eye_center(side)
    (up_p, n_up), (lo_p, n_lo) = _lid_planes(side)
    shell_u = sd_sphere(P, E, EYE_R + 0.0030)
    shell_l = sd_sphere(P, E, EYE_R + 0.0024)
    upper = smax(shell_u, -((P - up_p) @ n_up), 0.0016)
    lower = smax(shell_l, -((P - lo_p) @ n_lo), 0.0016)
    # keep lids to the eye area
    clip = sd_ellipsoid(P, E + v3(0, -0.004, 0), v3(0.0215, 0.020, 0.0175))
    upper = smax(upper, clip, 0.003)
    lower = smax(lower, clip, 0.003)
    return upper, lower


def _ear(P, side):
    """Stylised ear built in its own plane."""
    C = v3(side * 0.0745, 0.0135, -0.0125)
    R = rot_z(side * 14.0) @ rot_x(-12.0)        # flare out, lean back
    q = (P - C) @ R
    u = -q[:, 1]       # forward
    v = q[:, 2]        # up
    w = q[:, 0] * side  # outward
    # egg shaped outline, fuller at the top
    a = 0.0165 + 0.0035 * smoothstep(-0.02, 0.02, v)
    b = 0.0300
    e2 = sd_ellipse2(u + 0.001, v, a, b)
    plate = np.maximum(e2, np.abs(w - 0.0010) - 0.0030)
    plate = smax(plate, -w - 0.006, 0.002)
    rim = np.sqrt((np.abs(e2 + 0.0018)) ** 2 + (w - 0.0032) ** 2) - 0.0026
    rim = np.where(u > 0.012, rim + 0.004 * smoothstep(0.012, 0.016, u), rim)   # helix opens at front
    d = smin(plate, rim, 0.0025)
    # antihelix ridge
    e3 = sd_ellipse2(u + 0.002, v - 0.002, a * 0.62, b * 0.62)
    anti = np.sqrt(e3 ** 2 + (w - 0.0030) ** 2) - 0.0018
    anti = np.where(v < -0.012, anti + 0.01, anti)
    d = smin(d, anti, 0.002)
    # concha bowl
    concha = sd_ellipsoid(np.stack([u, v, w], -1), v3(-0.0025, -0.0035, 0.0072), v3(0.0085, 0.0105, 0.0065))
    d = smax(d, -concha, 0.0018)
    # lobe and tragus
    lobe = sd_ellipsoid(np.stack([u, v, w], -1), v3(0.0005, -0.0255, 0.0012), v3(0.0075, 0.0085, 0.0040))
    tragus = sd_ellipsoid(np.stack([u, v, w], -1), v3(-0.0120, -0.0060, 0.0038), v3(0.0035, 0.0050, 0.0030))
    d = smin(d, lobe, 0.004)
    d = smin(d, tragus, 0.002)
    return d


# face loft: z, (a, yc, y_front, y_back, n_front, n_back)
FACE_Z = [0.084, 0.052, 0.025, 0.008, -0.012, -0.034, -0.054, -0.074, -0.090, -0.104, -0.115]
FACE_ROWS = [
    (0.0585, -0.010, -0.0790, 0.060, 2.4, 2.0),
    (0.0680, -0.012, -0.0905, 0.060, 2.6, 2.0),
    (0.0705, -0.015, -0.0975, 0.060, 2.9, 2.0),
    (0.0712, -0.018, -0.0930, 0.060, 3.1, 2.0),
    (0.0722, -0.020, -0.0915, 0.058, 3.3, 2.0),
    (0.0682, -0.022, -0.0930, 0.030, 3.0, 2.2),
    (0.0636, -0.025, -0.0950, 0.018, 2.7, 2.5),
    (0.0612, -0.028, -0.0955, 0.010, 2.5, 2.8),
    (0.0590, -0.030, -0.0935, 0.004, 2.4, 3.0),
    (0.0500, -0.040, -0.0950, -0.018, 2.3, 2.9),
    (0.0340, -0.055, -0.0915, -0.046, 2.2, 2.4),
]


def skull_sdf(P):
    d = sd_ellipsoid(P, v3(0, 0.012, 0.032), v3(0.0745, 0.0975, 0.0905))
    d = smax(d, np.abs(P[:, 0]) - 0.0715 - 0.010 * smoothstep(0.03, -0.04, P[:, 2]), 0.024)
    face = sd_loft(P, FACE_Z, FACE_ROWS, -0.1205, 0.086, k_cap=0.014)
    return smin(d, face, 0.022)


def _nose(dd, Q):
    Qm = mirror_x(Q)
    bridge = sd_round_cone(Q, v3(0, -0.0895, 0.0060), v3(0, -0.1125, -0.0320), 0.0050, 0.0068)
    wall = sd_round_cone(Qm, v3(0.0045, -0.0930, -0.0060), v3(0.0090, -0.1015, -0.0340), 0.0052, 0.0070)
    tip = sd_ellipsoid(Q, v3(0, -0.1120, -0.0362), v3(0.0088, 0.0085, 0.0078))
    alae = sd_ellipsoid(Qm, v3(0.0118, -0.0985, -0.0415), v3(0.0058, 0.0075, 0.0060), rot_z(-18))
    colu = sd_capsule(Q, v3(0, -0.1085, -0.0435), v3(0, -0.1005, -0.0462), 0.0038)
    nose = smin(bridge, wall, 0.006)
    nose = smin(nose, tip, 0.010)
    nose = smin(nose, alae, 0.006)
    nose = smin(nose, colu, 0.004)
    dd = smin(dd, nose, 0.011)
    # nostrils, read from below only
    dd = ssub(dd, sd_ellipsoid(Qm, v3(0.0064, -0.1030, -0.0472), v3(0.0032, 0.0048, 0.0020), rot_z(-25)), 0.0016)
    return dd


def _mouth(dd, Q):
    Qm = mirror_x(Q)
    # upper lip: central tubercle and two lobes, restrained volume
    u_c = sd_ellipsoid(Q, v3(0, -0.0958, -0.0646), v3(0.0050, 0.0032, 0.0030))
    u_l = sd_ellipsoid(Qm, v3(0.0085, -0.0940, -0.0652), v3(0.0100, 0.0036, 0.0030), rot_z(-14) @ rot_x(-10))
    lo_l = sd_ellipsoid(Qm, v3(0.0058, -0.0934, -0.0750), v3(0.0095, 0.0042, 0.0042), rot_z(-8) @ rot_x(8))
    lips = smin(smin(u_c, u_l, 0.006), lo_l, 0.003)
    dd = smin(dd, lips, 0.0060)
    # soft philtrum groove
    dd = ssub(dd, sd_capsule(Q, v3(0, -0.1000, -0.0510), v3(0, -0.1005, -0.0592), 0.0014), 0.0030)
    # mouth line: restrained, corners firm and neutral
    mx = Q[:, 0]
    mz = -0.0693 - 0.0012 * (mx / 0.021) ** 2
    slit = np.maximum(np.abs(Q[:, 2] - mz) - 0.00065, np.abs(mx) - 0.0215)
    slit = np.maximum(slit, Q[:, 1] + 0.087)
    dd = smax(dd, -slit, 0.0011)
    dd = ssub(dd, sd_sphere(Qm, v3(0.0222, -0.0880, -0.0702), 0.0016), 0.0022)
    # mentolabial sulcus
    dd = ssub(dd, sd_capsule(Q, v3(-0.011, -0.0990, -0.0865), v3(0.011, -0.0990, -0.0865), 0.0030), 0.006)
    return dd


def _eyes(dd, Q):
    Qm = mirror_x(Q)
    # brow shelf, lateral brow and the tension between the brows
    brow = sd_round_cone(Qm, v3(0.006, -0.0900, 0.0175), v3(0.0505, -0.0770, 0.0245), 0.0072, 0.0060)
    dd = smin(dd, brow, 0.012)
    dd = smin(dd, sd_ellipsoid(Qm, v3(0.0100, -0.0935, 0.0150), v3(0.0070, 0.0036, 0.0055)), 0.006)
    # orbits
    orbit = sd_ellipsoid(Qm, v3(EYE_X, -0.0905, 0.0020), v3(0.0195, 0.0165, 0.0145), rot_z(-6))
    dd = ssub(dd, orbit, 0.009)
    for side in (1, -1):
        E = eye_center(side)
        ball = sd_sphere(Q, E, EYE_R - 0.0009)
        upper, lower = _eyelids(Q, side)
        fold = sd_ellipsoid(Q, E + v3(side * 0.0010, -0.0090, 0.0108), v3(0.0165, 0.0062, 0.0050), rot_z(side * -6))
        dd = smin(dd, fold, 0.006)
        dd = smin(dd, upper, 0.0050)
        dd = smin(dd, lower, 0.0045)
        # lower-lid tension pad
        dd = smin(dd, sd_ellipsoid(Q, E + v3(side * 0.0012, -0.0105, -0.0085), v3(0.0115, 0.0040, 0.0034)), 0.004)
        dd = np.minimum(dd, ball)
    return dd


# compact lower face: everything below the cheekbones is drawn up a little
# (stylised proportions); shaders apply the same warp to their masks
LF_TOP, LF_SPAN, LF_LIFT = -0.030, 0.092, 0.008


def lower_face_warp(P):
    """Map stylised space back to the sculpt space the features are built in."""
    Q = P.copy()
    Q[:, 2] = P[:, 2] - LF_LIFT * np.clip((LF_TOP - P[:, 2]) / LF_SPAN, 0.0, 1.0)
    return Q


def head_sdf(P, detail=True):
    X = lower_face_warp(P)
    Pm = mirror_x(X)
    d = skull_sdf(X)

    # cheekbones: sculptural, slightly exaggerated
    cheek = sd_ellipsoid(Pm, v3(0.0530, -0.0660, -0.0075), v3(0.0160, 0.0080, 0.0075), rot_z(-32) @ rot_x(-6))
    d = smin(d, cheek, 0.016)
    # chin pad and jaw corner
    d = smin(d, sd_ellipsoid(X, v3(0, -0.0895, -0.1060), v3(0.0165, 0.0095, 0.0115)), 0.008)

    d = local_op(d, X, (-0.040, -0.145, -0.070), (0.040, -0.065, 0.030), _nose)
    d = local_op(d, X, (-0.045, -0.125, -0.105), (0.045, -0.065, -0.035), _mouth)
    d = local_op(d, X, (-0.092, -0.135, -0.055), (0.092, -0.030, 0.060), _eyes)

    # ears
    d = local_op(d, X, (0.050, -0.020, -0.060), (0.100, 0.050, 0.030), lambda dd, Q: smin(dd, _ear(Q, 1), 0.005))
    d = local_op(d, X, (-0.100, -0.020, -0.060), (-0.050, 0.050, 0.030), lambda dd, Q: smin(dd, _ear(Q, -1), 0.005))

    # neck: strong, athletic
    neck = sd_round_cone(X, v3(0, 0.022, -0.250), v3(0, 0.016, -0.050), 0.0650, 0.0560)
    scm = sd_round_cone(Pm, v3(0.0540, 0.0260, -0.0450), v3(0.0150, -0.0400, -0.2050), 0.0145, 0.0120)
    neck = smin(neck, scm, 0.018)
    neck = smin(neck, sd_ellipsoid(X, v3(0, -0.0395, -0.1440), v3(0.0070, 0.0050, 0.0090)), 0.010)
    d = smin(d, neck, 0.014)

    if detail:
        d = d + 0.00022 * value_noise(P, 26.0, seed=3)
    return d


# --------------------------------------------------------------------------
# surface helpers
# --------------------------------------------------------------------------

def _project_points(f, pts, iters=12):
    from sdf_core import project
    return project(f, np.asarray(pts, dtype=np.float64), 0.0004, iters=iters)


def _normal(f, pts, eps=0.0004):
    from sdf_core import grad
    g = grad(f, np.asarray(pts, dtype=np.float64), eps)
    return g / np.linalg.norm(g, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# brows: short, clean sculpted masses; inner ends drawn slightly down
# --------------------------------------------------------------------------

_BROW_SEGS = None


def _brow_segments():
    """Short, clean graphic brows: flattened against the skin, inner ends drawn
    slightly down for a focused, competitive read."""
    segs = []
    f = lambda P: head_sdf(P, detail=False)
    for side in (1, -1):
        guide = np.array([[0.0085, -0.1010, 0.0150], [0.0195, -0.0990, 0.0205],
                          [0.0320, -0.0940, 0.0240], [0.0435, -0.0872, 0.0242],
                          [0.0510, -0.0812, 0.0218]])
        if side < 0:
            guide = guide + np.array([0.0, 0.0, 0.0006])   # tiny asymmetry
        guide[:, 0] *= side
        on = _project_points(f, guide)
        n = _normal(f, on)
        core = on + n * 0.0002
        radii = [0.0024, 0.0029, 0.0027, 0.0020, 0.0011]
        for i in range(len(core) - 1):
            segs.append((core[i], core[i + 1], radii[i], radii[i + 1], n[i], 2.4))
    return segs


def _lash_segments():
    """Upper lash line: a dark graphic stroke along each upper lid margin with
    a small lateral flick, the animated-feature way of framing the eye."""
    segs = []
    for side in (1, -1):
        E = eye_center(side)
        (up_p, n_up), _ = _lid_planes(side)
        r_s = EYE_R + 0.0021
        dpl = (up_p - E) @ n_up
        c = E + n_up * dpl - n_up * 0.0004
        rc = np.sqrt(max(r_s ** 2 - dpl ** 2, 1e-8))
        a = v3(1.0, 0.0, 0.0) - n_up * n_up[0]
        a = a / np.linalg.norm(a)
        b = np.cross(n_up, a)
        ts = np.radians(np.linspace(28.0, 152.0, 11))
        pts = np.array([c + rc * (np.cos(t) * a - np.sin(t) * b) for t in ts])
        # lateral end first
        if side < 0:
            pts = pts[::-1]
        m = len(pts)
        u = np.linspace(0.0, 1.0, m)            # 0 lateral -> 1 medial
        radii = 0.00078 * (1.0 - 0.45 * u ** 2) + 0.00005
        for i in range(m - 1):
            rad = norm(pts[i] - E)
            segs.append((pts[i], pts[i + 1], radii[i], radii[i + 1], rad, 1.4))
        # flick: out, back and slightly up from the lateral canthus
        p0 = pts[0]
        flick = p0 + v3(side * 0.0032, 0.0016, 0.0012)
        segs.append((p0, flick, radii[0], 0.00050, norm(p0 - E), 1.4))
    return segs


def brows_sdf(P):
    global _BROW_SEGS
    from sdf_core import union_segments
    if _BROW_SEGS is None:
        _BROW_SEGS = _brow_segments() + _lash_segments()
    d = union_segments(P, _BROW_SEGS, k=0.0008, cell=0.006)
    return d


# --------------------------------------------------------------------------
# hair: short faded sides, textured top that falls forward and slightly to
# the character's left; restrained volume, clean sculpted clumps
# --------------------------------------------------------------------------

HEAD_C = v3(0.0, 0.010, 0.020)
CROWN = v3(0.010, 0.052, 0.112)


def _scalp(P):
    return skull_sdf(lower_face_warp(P))


def _hairline_zmin(P):
    """Lowest z with hair for each point's azimuth around the head."""
    q = P - HEAD_C
    th = np.degrees(np.arctan2(q[:, 0], -q[:, 1]))    # 0 = front, +90 = left side
    a = np.abs(th)
    tab_a = [0, 20, 34, 46, 56, 64, 80, 96, 112, 130, 150, 180]
    tab_z = [0.075, 0.073, 0.068, 0.058, 0.044, 0.026, 0.024, 0.026, 0.010, -0.020, -0.040, -0.046]
    return np.interp(a, tab_a, tab_z), a


def hair_thickness(P):
    zmin, a = _hairline_zmin(P)
    z = P[:, 2]
    # sideburns: narrow band in front of the ear
    sb_z = 0.006 + 0.0045 * np.abs(a - 66.0)          # short pointed sideburn
    zmin = np.minimum(zmin, sb_z)
    m = smoothstep(zmin - 0.002, zmin + 0.010, z)
    top = smoothstep(0.050, 0.095, z) * smoothstep(150, 95, a)
    side_fade = 0.0010 + 0.0028 * smoothstep(0.0, 0.06, z - zmin)
    t = side_fade + 0.0130 * top
    return t * m - 0.0040 * (1 - m)


def _flow(p):
    """Hair flow direction (unnormalised, world) at a point on the head."""
    to = p - CROWN
    fwd = v3(0.30, -1.0, 0.0)
    w_back = smoothstep(0.02, 0.06, p[1])
    radial = to / (np.linalg.norm(to) + 1e-9)
    side = np.sign(p[0]) * smoothstep(0.040, 0.068, abs(p[0]))
    d = fwd * (1 - w_back) + radial * (w_back + 0.2)
    d = d + v3(side * 0.9, 0.0, -0.7 * abs(side))
    # over the front of the forehead the hair falls down, not into the skull
    d = d + v3(0.0, 0.0, -1.3) * smoothstep(-0.045, -0.072, p[1])
    return d


def hair_envelope(p):
    """Target outer height of the hair above the scalp."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    top = smoothstep(0.050, 0.098, z)
    front = smoothstep(0.05, -0.06, y)
    return 0.0030 + top * (0.0105 + 0.0045 * front)


_HAIR_SEGS = None


def _drop_to_scalp(f, sx, sy):
    lo, hi = 0.16, -0.03
    for _ in range(32):
        mid = (lo + hi) / 2
        if f(np.array([[sx, sy, mid]]))[0] < 0:
            hi = mid
        else:
            lo = mid
    return np.array([sx, sy, lo])


def _hair_segments(seed=7):
    """Large, coherent clumps for the textured top. Each clump rides at a
    near-constant height inside the hair mass so the top reads as clean
    parallel ridges (not shingles); front clumps continue past the hairline
    as a short fringe lying close to the forehead."""
    rng = np.random.default_rng(seed)
    f = _scalp
    segs = []
    seeds = []
    for row, gy in enumerate(np.arange(-0.060, 0.072, 0.0185)):
        off = 0.0092 if row % 2 else 0.0
        for gx in np.arange(-0.064, 0.0641, 0.0185):
            seeds.append((gx + off + rng.uniform(-0.003, 0.003), gy + rng.uniform(-0.003, 0.003), 1))
    # a few extra, narrower clumps along the front for a broken fringe edge
    for gx in np.linspace(-0.045, 0.050, 7):
        seeds.append((gx + rng.uniform(-0.004, 0.004), -0.058 + rng.uniform(-0.004, 0.004), 2))
    for sx, sy, level in seeds:
        p = _drop_to_scalp(f, sx, sy)
        if hair_thickness(p[None])[0] < 0.0065:
            continue
        L = (0.068 if level == 1 else 0.048) * rng.uniform(0.9, 1.1)
        w0 = (0.0122 if level == 1 else 0.0090) * rng.uniform(0.9, 1.1)
        jitter = rot_z(rng.uniform(-4, 4))
        n_steps = 8
        pts = [p]
        q = p.copy()
        for s in range(n_steps):
            d = jitter @ _flow(q)
            nrm = _normal(f, q[None])[0]
            d = d - nrm * (d @ nrm)
            d = d / (np.linalg.norm(d) + 1e-9)
            q = _project_points(f, (q + d * (L / n_steps))[None], iters=4)[0]
            zmin_q, a_q = _hairline_zmin(q[None])
            # stop short of the hairline; the front edge belongs to the fringe
            if q[2] < zmin_q[0] + (0.012 if a_q[0] < 60.0 else -0.002):
                break
            pts.append(q.copy())
        if len(pts) < 3:
            continue
        pts = np.array(pts)
        nr = _normal(f, pts)
        u = np.linspace(0.0, 1.0, len(pts))
        radii = w0 * (1.0 - 0.84 * u ** 1.3)
        flat = 1.7
        env = hair_envelope(pts)
        # ride inside the mass: outer surface just proud of the shell
        height = np.maximum(env - radii / flat * 0.55, radii / flat + 0.0010)
        cen = pts + nr * height[:, None]
        for i in range(len(cen) - 1):
            segs.append((cen[i], cen[i + 1], radii[i], radii[i + 1], nr[i], flat))
    segs += _fringe_segments(rng)
    segs += _side_segments(rng)
    return segs


def _side_segments(rng):
    """Low-relief clumps on the short sides and back: they only break up the
    silhouette and catch light, the shader adds the fine strands."""
    f = _scalp
    segs = []
    for az in np.radians(np.arange(-160.0, 161.0, 13.0)):
        if abs(np.degrees(az)) < 62.0:
            continue
        for ez in (0.030, 0.062):
            az_j = az + np.radians(rng.uniform(-3.0, 3.0))
            dirn = norm(v3(np.sin(az_j), -np.cos(az_j), (ez - HEAD_C[2]) / 0.09))
            lo_t, hi_t = 0.16, 0.0
            for _ in range(30):                       # march in to the scalp
                mid = (lo_t + hi_t) / 2
                if f((HEAD_C + dirn * mid)[None])[0] < 0:
                    hi_t = mid
                else:
                    lo_t = mid
            p = HEAD_C + dirn * lo_t
            if p[2] < _hairline_zmin(p[None])[0][0] + 0.010:
                continue
            nrm = _normal(f, p[None])[0]
            d = _flow(p) + v3(0, 0, -0.5)
            d = d - nrm * (d @ nrm)
            d = d / (np.linalg.norm(d) + 1e-9)
            L = 0.030 * rng.uniform(0.85, 1.15)
            pts = [p]
            for _ in range(4):
                q = _project_points(f, (pts[-1] + d * L / 4)[None], iters=4)[0]
                if q[2] < _hairline_zmin(q[None])[0][0] + 0.004:
                    break
                pts.append(q)
            if len(pts) < 3:
                continue
            pts = np.array(pts)
            nr = _normal(f, pts)
            u = np.linspace(0, 1, len(pts))
            radii = 0.0105 * (1.0 - 0.7 * u)
            th = hair_thickness(pts)
            cen = pts + nr * (th - radii / 2.6 + 0.0009)[:, None]
            for i in range(len(cen) - 1):
                segs.append((cen[i], cen[i + 1], radii[i], radii[i + 1], nr[i], 2.6))
    return segs


# fringe clumps: (root x, tip x, tip z); they sweep forward and to the
# character's left and end in clean points above the brows
FRINGE = [(-0.050, -0.042, 0.073, 0.0098), (-0.034, -0.019, 0.063, 0.0118), (-0.015, 0.002, 0.067, 0.0110),
          (0.003, 0.022, 0.057, 0.0126), (0.023, 0.039, 0.064, 0.0114), (0.041, 0.054, 0.072, 0.0098),
          # smaller in-between pieces break the comb-like rhythm
          (-0.025, -0.012, 0.074, 0.0068), (0.013, 0.030, 0.069, 0.0072), (0.032, 0.047, 0.076, 0.0064)]


def _fringe_segments(rng):
    f = _scalp
    segs = []
    for x0, x1, zt, wr in FRINGE:
        guide = []
        for t in np.linspace(0.0, 1.0, 7):
            x = x0 + (x1 - x0) * t ** 1.5
            z = 0.104 + (zt - 0.104) * t
            y = -0.040 - 0.060 * t           # projected onto the scalp below
            guide.append((x, y, z))
        on = _project_points(f, np.array(guide))
        nr = _normal(f, on)
        u = np.linspace(0.0, 1.0, len(on))
        w0 = wr * rng.uniform(0.95, 1.05)
        radii = w0 * (1.0 - 0.84 * u ** 1.5)
        flat = 1.7
        env = hair_envelope(on)
        zmin = _hairline_zmin(on)[0]
        lie = smoothstep(zmin + 0.010, zmin - 0.004, on[:, 2])   # resting on the forehead
        height = np.maximum(env - radii / flat * 0.35, radii / flat + 0.0012)
        height = height * (1 - lie) + (radii / flat * 0.9) * lie
        cen = on + nr * height[:, None]
        for i in range(len(cen) - 1):
            segs.append((cen[i], cen[i + 1], radii[i], radii[i + 1], nr[i], flat))
    return segs


def hair_sdf(P):
    global _HAIR_SEGS
    from sdf_core import union_segments
    if _HAIR_SEGS is None:
        _HAIR_SEGS = _hair_segments()
    scalp = _scalp(P)
    t = hair_thickness(P)
    # inner fill keeps the mass solid between locks; locks carry the shape
    zmin, az = _hairline_zmin(P)
    # the front edge stays full (the fringe falls over it); sides and back fade
    front_w = smoothstep(60.0, 30.0, az)
    taper = smoothstep(zmin + 0.006 * front_w, zmin + 0.016 - 0.004 * front_w, P[:, 2])
    recess = 1.0 - 0.7 * front_w * smoothstep(0.100, 0.080, P[:, 2])   # fringe sits proud
    fill = np.minimum(t, (0.0030 + 0.0055 * smoothstep(0.055, 0.095, P[:, 2]) * recess) * taper)
    fill = np.where(t < 0, t, np.maximum(fill, np.minimum(t, 0.0012)))
    shell = scalp - fill
    d = union_segments(P, _HAIR_SEGS, k=0.0022, base=shell, cell=0.012)
    # keep a solid root below the skin so thin edges never become paper-thin
    d = smax(d, -(scalp + 0.0025), 0.0008)
    return d
