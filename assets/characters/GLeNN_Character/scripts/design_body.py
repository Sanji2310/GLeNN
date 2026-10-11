"""The GLeNN driver's body in his racing suit, with gloves and boots, as
signed distance fields in world space (metres, facing -Y, feet at z = 0).

The body is never seen bare, so the suit is sculpted directly: the torso,
sleeves and legs carry the fit of a one-piece fireproof suit (roomy enough to
move in, gathered at the waistband and cuffs), with the construction details
a real suit has: a mandarin collar, shoulder epaulettes for extraction,
a covered front zip, a waistband, stretch side panels and seams.

Pose: a relaxed A-pose for the turnaround and for rigging. Arms hang about
35 degrees from the body with a slight bend at the elbow; the hands are
relaxed with the palms facing the thighs.

Proportions (1.78 m to the crown, about 7 heads tall): shoulders 0.44 m
across the deltoids, waist 0.27 m, hips 0.33 m, legs a little long for a
lean, athletic line.
"""

import math

import numpy as np

import sdf as S

F32 = np.float32


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class Joints:
    neck_base = np.array([0.0, 0.020, 1.445])
    shoulder = np.array([0.183, 0.016, 1.418])
    upper_dir = _unit((0.574, 0.050, -0.819))      # 35 degrees off vertical
    upper_len = 0.290
    fore_dir = _unit((0.500, -0.120, -0.866))
    fore_len = 0.255
    hip = np.array([0.095, 0.015, 0.905])
    knee = np.array([0.101, 0.002, 0.495])
    ankle = np.array([0.108, 0.035, 0.088])

    @classmethod
    def elbow(cls):
        return cls.shoulder + cls.upper_dir * cls.upper_len

    @classmethod
    def wrist(cls):
        return cls.elbow() + cls.fore_dir * cls.fore_len


J = Joints


# -- suit ------------------------------------------------------------------------------

def _torso(q):
    chest = S.ellipsoid(q, (0, 0.006, 1.290), (0.150, 0.104, 0.165))
    yoke = S.ellipsoid(q, (0, 0.016, 1.385), (0.178, 0.094, 0.072))
    belly = S.ellipsoid(q, (0, 0.002, 1.110), (0.143, 0.094, 0.150))
    pelvis = S.ellipsoid(q, (0, 0.012, 0.955), (0.144, 0.100, 0.105))
    glute = S.ellipsoid(q, (0.060, 0.056, 0.905), (0.068, 0.058, 0.085))
    trap = S.capsule(q, (0.040, 0.034, 1.468), (0.160, 0.024, 1.438), 0.034)
    delt = S.ellipsoid(q, (0.188, 0.012, 1.402), (0.058, 0.062, 0.072), S.rot(0, -20, 0))
    lats = S.ellipsoid(q, (0.120, 0.030, 1.250), (0.045, 0.070, 0.110), S.rot(0, 12, 0))
    d = S.smin(chest, yoke, 0.05)
    d = S.smin(d, belly, 0.06)
    d = S.smin(d, pelvis, 0.06)
    d = S.smin(d, glute, 0.04)
    d = S.smin(d, trap, 0.04)
    d = S.smin(d, lats, 0.04)
    return S.smin(d, delt, 0.035)


def _arm(q):
    s, e, w = J.shoulder, J.elbow(), J.wrist()
    upper = S.round_cone(q, s + J.upper_dir * 0.02, e, 0.050, 0.041)
    bicep = S.ellipsoid(q, s + J.upper_dir * 0.13 + np.array([0, -0.012, 0]), (0.040, 0.044, 0.075),
                        S.rot(0, -35, 0))
    fore = S.round_cone(q, e, w, 0.041, 0.031)
    bulge = S.ellipsoid(q, e + J.fore_dir * 0.07 + np.array([0, -0.004, 0]), (0.038, 0.040, 0.065),
                        S.rot(0, -30, 0))
    d = S.smin(upper, bicep, 0.02)
    d = S.smin(d, fore, 0.025)
    d = S.smin(d, bulge, 0.02)
    # the sleeve stops at the wrist, under the glove gauntlet
    return S.smax(d, (q - w) @ J.fore_dir.astype(F32) - F32(-0.02), 0.004)


def _leg(q):
    h, k, a = J.hip, J.knee, J.ankle
    thigh = S.round_cone(q, h, k, 0.083, 0.054)
    quad = S.ellipsoid(q, h + (k - h) * 0.42 + np.array([0.004, -0.018, 0]), (0.060, 0.060, 0.150))
    shin = S.round_cone(q, k, a + np.array([0, 0, 0.02]), 0.054, 0.051)
    calf = S.ellipsoid(q, k + (a - k) * 0.30 + np.array([0, 0.020, 0]), (0.046, 0.046, 0.110))
    d = S.smin(thigh, quad, 0.03)
    d = S.smin(d, shin, 0.03)
    return S.smin(d, calf, 0.03)


# -- livery panels ------------------------------------------------------------------
# Indigo suit with cream panels: a V-shaped shoulder yoke, a side panel that
# runs from the armpit down the outside of the leg, and a stripe along the top
# of each sleeve. Each panel is outlined by a thin amber pinstripe.
# cream_panels() is a signed distance to the panels in metres (positive
# inside), so the sculpted seams and the material share the same edges.

YOKE_Z0, YOKE_SLOPE = 1.318, 0.28          # the yoke's V points down to the zip
SIDE_Y = ((0.10, 0.030), (0.50, 0.010), (0.90, 0.012), (1.10, 0.010), (1.34, 0.018))


def _yoke(q):
    return q[:, 2] - F32(YOKE_Z0) - F32(YOKE_SLOPE) * np.minimum(q[:, 0], F32(0.20))


def _side_panel(q):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    zz, yy = zip(*SIDE_Y)
    yc = np.interp(z, zz, yy).astype(F32)
    w = (0.021 + 0.027 * S.smoothstep(0.86, 1.05, z)) * (1.0 - S.smoothstep(1.20, 1.33, z))
    d = w - np.abs(y - yc)
    # outer half only: keeps the panel off the inner thigh (and off the arms)
    d = np.minimum(d, x - (0.100 - 0.030 * S.smoothstep(0.88, 0.98, z)))
    return np.minimum(d, F32(0.215) - x)


def _arm_frame(q):
    """Closest point on the arm's bone line: (offset from it, outward-up
    direction, forward-back direction, distance along the forearm)."""
    s, e, w = J.shoulder, J.elbow(), J.wrist()
    best_o = None
    for a, b in ((s, e), (e, w)):
        ab = (b - a).astype(F32)
        u = ab / np.linalg.norm(ab)
        t = np.clip(((q - a.astype(F32)) @ ab) / F32(ab @ ab), 0.0, 1.0)
        o = q - (a.astype(F32) + t[:, None] * ab)
        n_out = np.array([-u[2], 0.0, u[0]], dtype=F32)
        n_out /= np.linalg.norm(n_out)
        fb = np.cross(u, n_out).astype(F32)
        r = np.sqrt((o * o).sum(axis=1))
        if best_o is None:
            best_o, best_n, best_b, best_r = o, np.tile(n_out, (len(q), 1)), np.tile(fb, (len(q), 1)), r
        else:
            m = r < best_r
            best_o[m], best_n[m], best_b[m], best_r[m] = o[m], n_out, fb, r[m]
    return best_o, best_n, best_b, best_r


def _sleeve_stripe(q):
    o, n_out, fb, r = _arm_frame(q)
    d = F32(0.020) - np.abs((o * fb).sum(axis=1))
    d = np.minimum(d, (o * n_out).sum(axis=1))
    return np.where(r < 0.075, d, F32(-1.0))


def cream_panels(q):
    """Signed distance to the cream panels (positive inside), mirrored q."""
    return np.maximum(np.maximum(_yoke(q), _side_panel(q)), _sleeve_stripe(q))


def amber_panels(q):
    """Solid amber areas: the waistband and the sleeve cuffs."""
    z = q[:, 2]
    band = F32(0.016) - np.abs(z - F32(1.035))
    c = (q - J.wrist().astype(F32)) @ J.fore_dir.astype(F32)
    cuff = np.minimum(c + F32(0.045), F32(-0.020) - c)
    return np.maximum(band, np.where(_arm_frame(q)[3] < 0.075, cuff, F32(-1.0)))


def _offset_band(q, d, mask, amount):
    """Raise (amount > 0) or sink the surface where mask is 1."""
    return d - F32(amount) * mask


def _details(q, d):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    # waistband
    band = 1.0 - S.smoothstep(0.016, 0.022, np.abs(z - F32(1.035)))
    d = _offset_band(q, d, band, 0.004)
    # covered front zip: a placket down the centre front
    plk = (1.0 - S.smoothstep(0.008, 0.012, x)) * S.smoothstep(1.06, 1.08, z) * (1.0 - S.smoothstep(1.43, 1.45, z))
    plk *= S.smoothstep(0.0, 0.04, -y)
    d = _offset_band(q, d, plk, 0.0025)
    # epaulettes along the top of each shoulder
    a, b = np.array([0.060, 0.022, 1.468]), np.array([0.205, 0.012, 1.445])
    ab = b - a
    t = np.clip(((q - a) @ ab) / F32(ab @ ab), 0, 1)
    off = q - (a + t[:, None] * ab)
    lat = np.sqrt(off[:, 1] ** 2 + (off[:, 0] * 0.0) ** 2)
    ep = (1.0 - S.smoothstep(0.014, 0.018, lat)) * (1.0 - S.smoothstep(0.98, 1.0, t)) * S.smoothstep(0.0, 0.05, t)
    ep *= S.smoothstep(1.40, 1.43, z)
    d = _offset_band(q, d, ep, 0.003)
    # seams: shallow grooves along the edges of the cream panels
    seams = 1.0 - S.smoothstep(0.0008, 0.0016, np.abs(cream_panels(q)))
    # knee panels
    kn = np.sqrt(((x - J.knee[0]) / 0.046) ** 2 + ((z - J.knee[2] - 0.01) / 0.075) ** 2)
    knee = (1.0 - S.smoothstep(0.80, 1.0, kn)) * S.smoothstep(0.0, 0.03, -(y - J.knee[1]))
    d = _offset_band(q, d, knee, 0.0015)
    return _offset_band(q, d, seams, -0.0012)


def _collar(q):
    c = np.array([0.0, 0.024, 1.462])
    r = np.sqrt(q[:, 0] ** 2 + (q[:, 1] - F32(c[1])) ** 2)
    ring = np.abs(r - F32(0.066)) - F32(0.0035)
    height = np.abs(q[:, 2] - F32(c[2]) - F32(0.10) * (q[:, 1] - F32(c[1]))) - F32(0.022)
    d = S.smax(ring, height, 0.003)
    # small gap at the front where the collar closes
    gap = S.smax(np.abs(q[:, 0]) - F32(0.003), q[:, 1] - F32(-0.03), 0.001)
    return S.smax(d, -gap, 0.0015)


def suit(p):
    q = S.mirror_x(p)
    B = S.boxed
    d = _torso(q)
    d = S.smin(d, B(q, (0.10, -0.14, 0.90), (0.56, 0.14, 1.52), _arm), 0.03)
    d = S.smin(d, B(q, (0.0, -0.14, 0.05), (0.24, 0.16, 1.02), _leg), 0.035)
    d = _details(q, d)
    # leave the neck open where the head's neck comes through
    neck_hole = S.round_cone(q, (0, 0.022, 1.36), (0, 0.020, 1.60), 0.050, 0.050)
    d = S.smax(d, -neck_hole, 0.008)
    d = S.smin(d, B(q, (-0.1, -0.1, 1.40), (0.1, 0.12, 1.50), _collar), 0.003)
    # the legs stop inside the boot tops
    ankle_cut = F32(0.100) - q[:, 2]
    return S.smax(d, ankle_cut, 0.004)


# -- gloves ----------------------------------------------------------------------------

def _hand_frame():
    f = J.fore_dir
    palm_n = _unit(np.array([-1.0, 0.0, 0.0]) - f * np.dot([-1.0, 0.0, 0.0], f))   # palm faces the thigh
    side = np.cross(f, palm_n)                                                         # toward the thumb
    if side[1] > 0:
        side = -side
    return f, palm_n, side


def _glove(q):
    w = J.wrist()
    f, pn, sd = _hand_frame()
    R = np.stack([sd, pn, f], axis=1).astype(F32)       # local x = thumb side, y = palm, z = along hand
    h = S.local(q, w, R)
    palm = S.round_box(h, (0.0, 0.0, 0.055), (0.041, 0.015, 0.046), 0.012)
    d = palm
    # fingers: (lateral offset, length scale, curl)
    for ox, ls, curl in ((0.028, 0.92, 22), (0.010, 1.00, 26), (-0.009, 0.97, 30), (-0.026, 0.82, 34)):
        base = np.array([ox, 0.002, 0.098])
        pts = [base]
        ang = 0.0
        for seg in (0.040, 0.026, 0.021):
            ang += math.radians(curl)
            step = np.array([0.0, math.sin(ang), math.cos(ang)]) * seg * ls
            pts.append(pts[-1] + step)
        d = S.smin(d, S.tube(h, pts, [0.0108, 0.0103, 0.0097, 0.0090]), 0.006)
    thumb = [np.array([0.036, 0.010, 0.030]), np.array([0.050, 0.020, 0.060]), np.array([0.052, 0.030, 0.088]),
             np.array([0.048, 0.036, 0.108])]
    d = S.smin(d, S.tube(h, thumb, [0.0145, 0.0122, 0.0108, 0.0100]), 0.008)
    # gauntlet over the sleeve cuff
    gaunt = S.round_cone(h, (0, 0, -0.075), (0, 0, 0.012), 0.046, 0.034)
    gaunt = S.smax(gaunt, -S.round_cone(h, (0, 0, -0.09), (0, 0, 0.0), 0.040, 0.029), 0.002)
    return S.smin(d, gaunt, 0.006)


def gloves(p):
    q = S.mirror_x(p)
    return S.boxed(q, (0.36, -0.15, 0.70), (0.68, 0.12, 1.08), _glove)


# -- boots -----------------------------------------------------------------------------

def _boot(q):
    a = J.ankle
    f = q - np.array([a[0], 0.0, 0.0], dtype=F32)
    sole = S.round_box(f, (0.004, -0.060, 0.010), (0.044, 0.132, 0.010), 0.008)
    # slight toe spring: lift the front of the sole
    toe = S.ellipsoid(f, (0.006, -0.165, 0.036), (0.046, 0.052, 0.032))
    vamp = S.ellipsoid(f, (0.002, -0.080, 0.050), (0.048, 0.105, 0.045))
    heel = S.ellipsoid(f, (0.000, 0.040, 0.055), (0.042, 0.045, 0.052))
    shaft = S.round_cone(f, (0.0, 0.032, 0.060), (0.0, 0.036, 0.150), 0.050, 0.046)
    d = S.smin(vamp, toe, 0.02)
    d = S.smin(d, heel, 0.03)
    d = S.smin(d, shaft, 0.03)
    d = S.smax(d, -(f[:, 2] - F32(0.006)), 0.004)
    d = S.smin(d, sole, 0.004)
    # collar of the boot: open at the top, a little flare
    top = f[:, 2] - F32(0.152)
    d = S.smax(d, top, 0.004)
    # ankle strap
    strap = S.smax(np.abs(f[:, 2] - F32(0.105)) - F32(0.011),
                   S.round_cone(f, (0.0, 0.034, 0.06), (0.0, 0.036, 0.15), 0.054, 0.049), 0.002)
    return S.smin(d, strap, 0.002)


def boots(p):
    q = S.mirror_x(p)
    return S.boxed(q, (0.0, -0.26, -0.01), (0.20, 0.13, 0.20), _boot)


def bounds(part):
    return {"suit": ((-0.62, -0.18, 0.08), (0.62, 0.18, 1.52)),
            "gloves": ((-0.70, -0.16, 0.68), (0.70, 0.14, 1.10)),
            "boots": ((-0.20, -0.26, -0.01), (0.20, 0.13, 0.20))}[part]
