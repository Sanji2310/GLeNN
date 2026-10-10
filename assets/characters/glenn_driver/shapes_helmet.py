"""Original GLeNN racing helmet.

Helmet-local space matches the head: origin on the eye line, +X left, -Y
forward, +Z up. The shell, visor and hardware are separate meshes so they
can carry their own materials (painted composite with clear coat, dark
transparent visor, metallic pivots).
"""

import numpy as np
from sdf_core import (v3, norm, rot_x, rot_y, rot_z, sd_sphere, sd_ellipsoid, sd_capsule,
                      sd_round_cone, sd_box, sd_cyl, sd_torus, smin, smax, ssub, raised,
                      engrave, smoothstep, local_op)


def outer_raw(P):
    d = sd_ellipsoid(P, v3(0.0, 0.018, 0.026), v3(0.120, 0.156, 0.134))
    d = smax(d, P[:, 2] - 0.152, 0.06)        # slightly flatter crown
    # chin bar: forward and down
    d = smin(d, sd_ellipsoid(P, v3(0.0, -0.066, -0.068), v3(0.100, 0.080, 0.068)), 0.050)
    # flatter, faster side planes
    d = smax(d, np.abs(P[:, 0]) - 0.118 + 0.010 * smoothstep(0.05, -0.10, P[:, 2]), 0.035)
    # bottom edge: low at the chin, sweeping up behind the jaw, low again at the nape
    y = P[:, 1]
    zcut = -0.128 + 0.026 * smoothstep(-0.06, 0.02, y) - 0.012 * smoothstep(0.05, 0.13, y)
    d = smax(d, zcut - P[:, 2], 0.014)
    return d


def aperture_region(P):
    """Eye-port opening (negative inside), extruded radially."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    th = np.degrees(np.arctan2(x, -(y - 0.02)))
    u = np.abs(th) - 58.0
    zc = 0.014 + 0.004 * smoothstep(20, 58, np.abs(th))
    half_h = 0.0255 - 0.006 * smoothstep(30, 58, np.abs(th))
    v = np.abs(z - zc) - half_h
    # rounded rectangle in (angle, height); angle scaled to metres
    uu = u * 0.0021
    q = np.stack([np.maximum(uu, -1), np.maximum(v, -1)], -1)
    r = 0.010
    qo = np.maximum(np.stack([uu + r, v + r], -1), 0.0)
    out = np.sqrt((qo ** 2).sum(-1)) + np.minimum(np.maximum(uu + r, v + r), 0.0) - r
    front = -(y + 0.02)            # only on the front half
    return np.maximum(out, -front - 0.01)


def shell_sdf(P):
    outer = outer_raw(P)
    # top air intake scoop
    scoop = sd_ellipsoid(P, v3(0.0, -0.050, 0.130), v3(0.020, 0.044, 0.0065), rot_x(-18))
    outer = smin(outer, scoop, 0.016)
    # rear spoiler lip
    spoiler = sd_ellipsoid(P, v3(0.0, 0.150, 0.094), v3(0.058, 0.011, 0.0055))
    outer = smin(outer, spoiler, 0.016)
    shell = smax(outer, -(outer + 0.019), 0.003)
    shell = smax(shell, -aperture_region(P), 0.004)
    # intake mouth
    slot = sd_box(P, v3(0.0, -0.100, 0.127), v3(0.014, 0.030, 0.0030), 0.0022, rot_x(-18))
    shell = smax(shell, -slot, 0.002)
    # chin vents
    for i, z in enumerate((-0.074, -0.086, -0.098)):
        w = 0.022 - 0.004 * i
        vent = sd_box(P, v3(0.0, -0.140, z), v3(w, 0.030, 0.0022), 0.0018)
        shell = smax(shell, -vent, 0.0015)
    # panel shut-lines that the livery follows
    return shell


def visor_sdf(P):
    outer = outer_raw(P)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    th = np.degrees(np.arctan2(x, -(y - 0.02)))
    sheet = np.abs(outer - 0.0042) - 0.0011
    zc = 0.016 + 0.006 * smoothstep(30, 85, np.abs(th))
    half_h = 0.040 - 0.012 * smoothstep(40, 90, np.abs(th))
    band = np.maximum(np.abs(z - zc) - half_h, np.abs(th) - 92.0)
    d = smax(sheet, band, 0.002)
    # lift tab on the lower left edge
    tab = sd_box(P, v3(0.050, -0.122, -0.026), v3(0.012, 0.004, 0.006), 0.002, rot_z(-24))
    tab = smax(tab, np.abs(outer - 0.0042) - 0.0016, 0.001)
    return smin(d, tab, 0.002)


def hardware_sdf(P):
    d = np.full(len(P), 1.0)
    for s in (1, -1):
        # visor pivot covers
        c = v3(s * 0.1235, 0.012, 0.020)
        R = np.stack([v3(0, 1, 0), v3(0, 0, 1), v3(s, 0, 0)], axis=1)
        disc = sd_cyl(P, c, 0.021, 0.0030, R)
        disc = smin(disc, sd_cyl(P, c + v3(s * 0.003, 0, 0), 0.007, 0.0025, R), 0.002)
        d = np.minimum(d, disc)
        # HANS anchor posts
        post = sd_box(P, v3(s * 0.080, 0.085, -0.100), v3(0.010, 0.016, 0.012), 0.004)
        d = np.minimum(d, post)
    # neck seal / lining ring at the bottom opening
    ring = sd_torus(P, v3(0.0, 0.020, -0.098), 0.080, 0.011, rot_x(8))
    ring = smax(ring, sd_ellipsoid(P, v3(0.0, 0.012, -0.05), v3(0.112, 0.140, 0.12)), 0.004)
    d = np.minimum(d, ring)
    return d


BOUNDS = (v3(-0.135, -0.175, -0.150), v3(0.135, 0.180, 0.180))
