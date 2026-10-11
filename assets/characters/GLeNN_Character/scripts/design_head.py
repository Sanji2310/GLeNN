"""The GLeNN driver's head, sculpted as a signed distance field.

The head is built the way a sculptor blocks in a portrait: skull and jaw
masses first, then the brow, eye sockets, eyelids, cheekbones, nose, mouth,
chin, ears and neck, each one added or carved with a smooth blend.

Local frame: the origin is the midpoint between the eyeball centres, +X is
the character's left, -Y is forward, +Z is up, in metres. The head is
symmetric, so the field is evaluated on |x| and every bilateral feature is
written once for the +X side.

Design (original character, recorded in reports/tool_assessment.md):
  * about 24, lean and athletic, roughly 7 heads tall
  * angular jaw, strong but narrow chin, high cheekbones with lean hollows
  * straight heavy brow ridge, deep-set slightly hooded almond eyes with a
    small positive tilt
  * straight nose with a small bump on the bridge
  * defined, fairly full lips with a calm, set mouth
  * thick athlete's neck (F1 drivers train their necks)
"""

import math

import numpy as np

import sdf as S

EYE_Z = 1.655             # world height of the eye line (crown at 1.78)


class HeadDesign:
    """All shape parameters in one place, so design passes are small edits."""

    # eyes
    eye_x = 0.0330
    eye_y = -0.0610
    eye_r = 0.0140
    eye_tilt = 5.0          # degrees, outer corner up
    upper_lid = 12.0        # elevation of the upper lid margin at the front (deg)
    lower_lid = -24.5
    lid_skew = 7.0          # moves the upper lid peak toward the nose
    lid_thick = 0.0028
    orbit_r = (0.024, 0.021)   # half width and half height of the orbit recess
    orbit_depth = (0.011, 0.002, 0.007)   # recess depth above, below and toward the nose

    # skull
    cranium_c = (0.0, 0.012, 0.026)
    cranium_r = (0.0810, 0.102, 0.099)

    # jaw: gonion (angle of the jaw), chin corner and chin bottom
    gonion = (0.057, 0.030, -0.088)
    chin_corner = (0.026, -0.080, -0.114)
    menton = (0.0, -0.078, -0.121)

    # nose
    nasion = (0.0, -0.0790, 0.004)
    nose_tip = (0.0, -0.1145, -0.041)
    subnasale = (0.0, -0.0990, -0.056)

    # mouth
    mouth_z = -0.0775
    mouth_half_width = 0.0235
    lip_front = -0.1000

    # whole-head scale (1.0 = 0.25 m chin to crown)
    scale = 1.0


D = HeadDesign
F32 = np.float32


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _plane_through(q, point, normal):
    n = _unit(normal)
    return q @ n.astype(F32) - F32(np.dot(point, n))


# -- big masses ------------------------------------------------------------------------

def _cranium(q):
    return S.ellipsoid(q, D.cranium_c, D.cranium_r)


def _mandible(q):
    """Jaw block: planar sides from the jaw angle to the chin, a flat
    underside, a near-vertical chin front and a top that rises toward the ear."""
    G, C, M = (np.array(v) for v in (D.gonion, D.chin_corner, D.menton))
    lean = _unit((0.13, 0.0, 1.0))
    n_side = np.cross(C - G, lean)
    if n_side[0] < 0:
        n_side = -n_side
    side = _plane_through(q, G, n_side)
    gb = np.array([0.0, G[1] + 0.002, G[2] - 0.004])
    t = gb - M
    n_bot = _unit((0.0, t[2], -t[1]))
    if n_bot[2] > 0:
        n_bot = -n_bot
    bottom = _plane_through(q, M, n_bot)
    front = _plane_through(q, (0, -0.094, -0.100), (0, -1.0, 0.05))
    bevel = _plane_through(q, (0, M[1] - 0.010, M[2] + 0.003), (0, -0.7, -0.7))
    top = _plane_through(q, (0, -0.090, -0.096), (0, -0.42, 0.91))
    back = _plane_through(q, (0, G[1] + 0.016, G[2] + 0.02), (0, 1.0, 0.30))
    d = S.smax(side, bottom, 0.011)
    d = S.smax(d, front, 0.016)
    d = S.smax(d, bevel, 0.012)
    d = S.smax(d, back, 0.020)
    return S.smax(d, top, 0.030)


def _midface(q):
    """Cheek masses either side of the nose, and the barrel of the mouth."""
    cheeks = S.ellipsoid(q, (0.030, -0.062, -0.048), (0.026, 0.024, 0.026))
    barrel = S.ellipsoid(q, (0, -0.056, -0.073), (0.036, 0.0445, 0.034))
    buccal = S.ellipsoid(q, (0.040, -0.046, -0.064), (0.022, 0.030, 0.026))
    d = S.smin(cheeks, barrel, 0.016)
    return S.smin(d, buccal, 0.022)


def _brow(q):
    return S.tube(q, [(0.0, -0.0835, 0.0115), (0.018, -0.0845, 0.0145), (0.034, -0.0795, 0.0160),
                      (0.050, -0.0705, 0.0135), (0.060, -0.0600, 0.0080)],
                  [0.0042, 0.0058, 0.0062, 0.0055, 0.0040])


def _cheekbone(q):
    zyg = S.ellipsoid(q, (0.051, -0.057, -0.020), (0.018, 0.015, 0.011), S.rot(0, -10, -25))
    arch = S.capsule(q, (0.055, -0.050, -0.016), (0.064, 0.000, -0.012), 0.0060)
    return S.smin(zyg, arch, 0.012)


# -- eyes --------------------------------------------------------------------------------

def _eye_local(q):
    return S.local(q, (D.eye_x, D.eye_y, 0.0), S.rot(0, D.eye_tilt, 0))


def _orbit_bowl(q):
    """Smooth bowl over the orbit: 1 at the eye, falling to 0 at the rim.
    The face is pushed in by orbit_depth times this, so the eye sits in a
    recess that follows the face instead of a hole cut into it."""
    dx = (q[:, 0] - F32(D.eye_x + 0.0015)) / F32(D.orbit_r[0])
    dz = (q[:, 2] - F32(-0.0015)) / F32(D.orbit_r[1])
    rho = np.sqrt(dx * dx + dz * dz)
    top, bottom, medial = D.orbit_depth
    depth = bottom + (top - bottom) * S.smoothstep(-0.7, 0.5, dz)
    depth = depth + medial * S.smoothstep(0.1, -0.8, dx) * (1.0 - S.smoothstep(0.2, 0.9, dz))
    return depth * (1.0 - S.smoothstep(0.30, 1.0, rho))


def _eye_mound(q):
    """The surface the lids follow: the eyeball plus lid thickness, with the
    soft fold under the brow and fullness under the lower lid."""
    e = _eye_local(q)
    r = D.eye_r
    d = S.length(e) - F32(r + 0.0006 + D.lid_thick)
    hood = S.ellipsoid(e, (0.0015, -0.0078, 0.0102), (0.0155, 0.0062, 0.0046), S.rot(-25, 0, 0))
    d = S.smin(d, hood, 0.0045)
    under = S.ellipsoid(e, (0.0005, -0.0088, -0.0112), (0.0135, 0.0045, 0.0036), S.rot(20, 0, 0))
    return S.smin(d, under, 0.004)


def _fissure(q):
    """The eye opening: a wedge between the two lid planes, in front of the
    eye centre and only as deep as the lids."""
    e = _eye_local(q)
    tu, tl, sk = (math.radians(a) for a in (D.upper_lid, D.lower_lid, D.lid_skew))
    n_u = _unit((math.sin(sk), math.sin(tu), math.cos(tu)))
    n_l = _unit((-0.6 * math.sin(sk), math.sin(tl), math.cos(tl)))
    w = S.smax(e @ n_u.astype(F32), -(e @ n_l.astype(F32)), 0.0010)
    w = S.smax(w, e[:, 1] - F32(0.003), 0.004)
    return S.smax(w, S.length(e) - F32(D.eye_r + D.lid_thick + 0.003), 0.001)


# -- nose and mouth ------------------------------------------------------------------------

def _nose(q):
    nx, ny, nz = D.nasion
    tx, ty, tz = D.nose_tip
    supratip = (0.0, ty + 0.0065, tz + 0.0095)
    bump = (0.0, ny + 0.45 * (supratip[1] - ny) - 0.0020, nz + 0.45 * (supratip[2] - nz))
    dorsum = S.tube(q, [(nx, ny, nz), bump, supratip], [0.0048, 0.0056, 0.0058], k=0.006)
    walls = S.ellipsoid(q, (0.0, -0.0905, -0.030), (0.0120, 0.0150, 0.026), S.rot(-28, 0, 0))
    tip = S.ellipsoid(q, (tx, ty + 0.0080, tz + 0.0005), (0.0068, 0.0080, 0.0075))
    base = S.ellipsoid(q, (0.0, -0.0965, -0.0480), (0.0165, 0.0080, 0.0068))
    d = S.smin(dorsum, walls, 0.010)
    d = S.smin(d, tip, 0.007)
    d = S.smin(d, base, 0.006)
    # alar crease: where the wing of the nostril meets the cheek
    crease = S.tube(q, [(0.0120, -0.0915, -0.0405), (0.0170, -0.0925, -0.0455), (0.0165, -0.0955, -0.0520)],
                    [0.0007, 0.0008, 0.0006])
    d = S.smax(d, -crease, 0.0018)
    nostril = S.ellipsoid(q, (0.0055, -0.1050, -0.0520), (0.0026, 0.0050, 0.0020), S.rot(-15, 0, 18))
    return S.smax(d, -nostril, 0.0018)


ARCH_C = -0.040        # centre of the dental arch circle (y)
ARCH_R = 0.060         # its radius: the lips sit on this curve


def _arch(q):
    """Bend space around the dental arch: s runs along the lips (arc length,
    0 at the centre), t is the distance in front of the arch, z is height."""
    dy = q[:, 1] - F32(ARCH_C)
    dist = np.sqrt(q[:, 0] ** 2 + dy * dy)
    out = np.empty_like(q)
    out[:, 0] = F32(ARCH_R) * np.arctan2(q[:, 0], -dy)
    out[:, 1] = dist - F32(ARCH_R)
    out[:, 2] = q[:, 2]
    return out


def _lips(q):
    """Upper lip in three masses (centre tubercle and two sides), lower lip
    in two fuller masses, all laid along the dental arch."""
    a = _arch(q)
    mz, t0 = D.mouth_z, -D.lip_front - ARCH_R + ARCH_C
    hw = D.mouth_half_width
    tub = S.ellipsoid(a, (0.0, t0 - 0.0010, mz + 0.0030), (0.0055, 0.0036, 0.0034), S.rot(-25, 0, 0))
    side = S.ellipsoid(a, (0.0105, t0 - 0.0022, mz + 0.0028), (0.0115, 0.0033, 0.0029), S.rot(-25, 0, 0))
    upper = S.smin(tub, side, 0.003)
    low = S.ellipsoid(a, (0.0068, t0 - 0.0008, mz - 0.0046), (0.0118, 0.0050, 0.0047), S.rot(15, 0, 0))
    low_mid = S.ellipsoid(a, (0.0, t0 - 0.0006, mz - 0.0044), (0.0080, 0.0048, 0.0045), S.rot(15, 0, 0))
    lower = S.smin(low, low_mid, 0.004)
    lips = S.smin(upper, lower, 0.0012)
    # taper both lips into the corners of the mouth
    corner = np.abs(a[:, 0]) - F32(hw)
    return S.smax(lips, corner, 0.006)


def _mouth_slit(q):
    a = _arch(q)
    mz, t0 = D.mouth_z, -D.lip_front - ARCH_R + ARCH_C
    slab = np.abs(a[:, 2] - F32(mz)) - F32(0.00022)
    ends = np.abs(a[:, 0]) - F32(D.mouth_half_width + 0.0006)
    depth = -(a[:, 1] - F32(t0 - 0.010))
    d = S.smax(slab, ends, 0.0015)
    return S.smax(d, depth, 0.002)


def _philtrum(q):
    """A shallow groove down the middle of the upper lip (carved)."""
    return S.capsule(q, (0.0, D.subnasale[1] - 0.0028, D.subnasale[2] - 0.003),
                     (0.0, D.lip_front - 0.0045, D.mouth_z + 0.0075), 0.0022)


def _chin(q):
    M = D.menton
    return S.round_box(q, (0, M[1] - 0.002, M[2] + 0.014), (0.018, 0.010, 0.011), 0.008)


# -- ears and neck ---------------------------------------------------------------------------

def _ear(q):
    e = S.local(q, (0.0725, 0.014, -0.019), S.rot(-14, 0, 22)) / F32(1.1)   # x out, y back, z up
    body = S.ellipsoid(e, (0.0, 0.002, 0.0), (0.0060, 0.0165, 0.0300))
    pts, rad = [], []
    for i in range(10):
        a = math.radians(-60 + 230 * i / 9)
        pts.append((0.0035, 0.002 + 0.0145 * math.sin(a), 0.0040 + 0.0255 * math.cos(a)))
        rad.append(0.0030 if i < 8 else 0.0024)
    rim = S.tube(e, pts, rad, k=0.002)
    lobe = S.ellipsoid(e, (0.0005, 0.0040, -0.0230), (0.0045, 0.0090, 0.0085))
    ear = S.smin(S.smin(body, rim, 0.003), lobe, 0.004)
    concha = S.ellipsoid(e, (0.0060, 0.0000, -0.0060), (0.0060, 0.0085, 0.0105))
    scapha = S.ellipsoid(e, (0.0065, 0.0040, 0.0080), (0.0045, 0.0100, 0.0160))
    ear = S.smax(ear, -concha, 0.003)
    ear = S.smax(ear, -scapha, 0.002)
    anti = S.tube(e, [(0.0035, 0.0070, -0.0120), (0.0035, 0.0085, 0.0010), (0.0035, 0.0050, 0.0120),
                      (0.0035, -0.0040, 0.0170)], [0.0021, 0.0022, 0.0020, 0.0016], k=0.002)
    ear = S.smin(ear, anti, 0.0025)
    tragus = S.ellipsoid(e, (0.0040, -0.0095, -0.0060), (0.0030, 0.0035, 0.0045))
    ear = S.smin(ear, tragus, 0.002)
    stalk = S.ellipsoid(e, (-0.0050, -0.0020, -0.0040), (0.0080, 0.0100, 0.0180))
    return S.smin(ear, stalk, 0.004) * F32(1.1)


def _neck(q):
    col = S.round_cone(q, (0, 0.030, -0.240), (0, 0.032, -0.092), 0.063, 0.055)
    scm = S.capsule(q, (0.050, 0.040, -0.066), (0.014, -0.030, -0.215), 0.0080)
    adam = S.ellipsoid(q, (0, -0.019, -0.140), (0.0060, 0.0050, 0.0090))
    d = S.smin(col, scm, 0.03)
    return S.smin(d, adam, 0.010)


# -- assembly ---------------------------------------------------------------------------------

def field_local(p, ears=True):
    """Head skin in the local frame (eyeballs and brows are separate).
    ears=False gives the scalp the hair is built on."""
    q = S.mirror_x(p)
    B = S.boxed
    d = _cranium(q)
    d = S.smin(d, _mandible(q), 0.035)
    d = S.smin(d, _midface(q), 0.035)
    d = S.smin(d, B(q, (-0.02, -0.12, -0.02), (0.09, -0.03, 0.045), _brow), 0.018)
    d = S.smin(d, B(q, (0.01, -0.10, -0.06), (0.10, 0.04, 0.03), _cheekbone), 0.020)
    d = S.smax(d, -B(q, (0.03, -0.08, -0.10), (0.10, 0.00, -0.02),
                     lambda r: S.ellipsoid(r, (0.076, -0.034, -0.058), (0.006, 0.012, 0.011))), 0.018)
    d = S.smin(d, B(q, (-0.01, -0.11, -0.14), (0.05, -0.05, -0.08), _chin), 0.014)
    # eyes
    eye_box = ((0.0, -0.118, -0.035), (0.075, -0.035, 0.035))
    inb = np.all((q >= np.array(eye_box[0], dtype=F32)) & (q <= np.array(eye_box[1], dtype=F32)), axis=1)
    ii = np.nonzero(inb)[0]
    if len(ii):
        qe = q[ii]
        de = d[ii] + _orbit_bowl(qe)
        de = S.smin(de, _eye_mound(qe), 0.004)
        de = S.smax(de, -(S.length(_eye_local(qe)) - F32(D.eye_r + 0.0003)), 0.0006)
        de = S.smax(de, -_fissure(qe), 0.0012)
        d = d.copy()
        d[ii] = de
    # nose and mouth
    d = S.smax(d, -B(q, (-0.02, -0.11, -0.012), (0.03, -0.07, 0.024),
                     lambda r: S.ellipsoid(r, (0, -0.0915, 0.0055), (0.0075, 0.0060, 0.0060))), 0.008)
    d = S.smin(d, B(q, (-0.01, -0.14, -0.075), (0.04, -0.06, 0.02), _nose), 0.006)
    mouth_box = ((-0.01, -0.12, -0.11), (0.045, -0.06, -0.045))
    d = S.smax(d, -B(q, *mouth_box, _philtrum), 0.0030)
    d = S.smin(d, B(q, *mouth_box, _lips), 0.0040)
    d = S.smax(d, -B(q, *mouth_box, _mouth_slit), 0.0012)
    # soft hollow between the lower lip and the chin
    d = S.smax(d, -B(q, *mouth_box, lambda r: S.ellipsoid(
        r, (0, D.lip_front + 0.0060, D.mouth_z - 0.0170), (0.0150, 0.0040, 0.0060))), 0.0080)
    # ears and neck
    if ears:
        d = S.smin(d, B(q, (0.04, -0.03, -0.075), (0.11, 0.06, 0.035), _ear), 0.003)
    d = S.smin(d, _neck(q), 0.025)
    return d


def field(p):
    """World-space field: local design placed at the eye line and scaled."""
    s = F32(D.scale)
    q = p - np.array([0.0, 0.0, EYE_Z], dtype=F32)
    return field_local(q / s) * s


def bounds():
    s = D.scale
    lo = np.array([-0.105, -0.140, -0.235]) * s + np.array([0, 0, EYE_Z])
    hi = np.array([0.105, 0.130, 0.135]) * s + np.array([0, 0, EYE_Z])
    return lo, hi


def eye_centers():
    s = D.scale
    return [np.array([sx * D.eye_x, D.eye_y, 0.0]) * s + np.array([0, 0, EYE_Z]) for sx in (1, -1)]
