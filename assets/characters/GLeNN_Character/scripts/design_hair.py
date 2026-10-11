"""The GLeNN driver's hair, sculpted as a signed distance field.

Style: short and practical, faded close at the sides and back, with a
textured top that is brushed up off the forehead and swept back into a short
quiff. Built in two layers, the way a stylised hair sculpt is blocked in:

  cap     a shell that follows the scalp. It is thin over the ears and the
          nape (the fade) and fuller on top, and it tapers to nothing at the
          hairline so the hair grows out of the skin instead of sitting on it.
  clumps  tapered ribbons of hair, each a swept ellipse that follows the
          scalp. They rise off the hairline, crest at the quiff and lie back
          toward the crown. Clumps are blended with a small radius so the
          grooves between them stay readable as separate hair masses.

Clumps are written in scalp coordinates around the head centre C0 (in the
head's local frame): azimuth phi in degrees (0 = front, + toward the
character's left), elevation theta in degrees (0 = level with C0), and the
height of the clump's core above the skin in metres. The hair is not mirrored:
it parts and sweeps slightly differently on each side.
"""

import math

import numpy as np

import design_head as H
import sdf as S

F32 = np.float32
C0 = np.array([0.0, 0.010, 0.030], dtype=F32)

# lowest elevation of the hair at each azimuth (|phi|), in degrees
HAIRLINE = [(0, 25.0), (20, 24.0), (38, 20.0), (52, 9.0), (64, -14.0), (74, -27.0), (82, -28.0),
            (90, -9.0), (104, -5.0), (118, -16.0), (128, -36.0), (150, -52.0), (180, -57.0)]


def scalp(p):
    """Head skin without the ears, in the local frame."""
    return H.field_local(p, ears=False)


def angles(p):
    d = p - C0
    r = np.sqrt((d * d).sum(axis=1))
    phi = np.degrees(np.arctan2(d[:, 0], -d[:, 1]))
    theta = np.degrees(np.arcsin(np.clip(d[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    return phi.astype(F32), theta.astype(F32)


def theta_min(phi):
    a, t = zip(*HAIRLINE)
    return np.interp(np.abs(phi), a, t).astype(F32)


# grooves between hair masses on top: (angle around the front-back axis in
# degrees, half width in degrees, depth in metres, where they start and end
# along the head from front (0) to back (1))
GROOVES = [(-44, 3.5, 0.0040, 0.10, 0.75), (-31, 3.0, 0.0050, 0.00, 0.85), (-18, 3.2, 0.0055, 0.05, 0.95),
           (-6, 2.6, 0.0045, 0.20, 0.80), (5, 3.4, 0.0060, 0.00, 1.00), (17, 3.0, 0.0050, 0.10, 0.90),
           (29, 3.2, 0.0055, 0.00, 0.80), (41, 3.5, 0.0045, 0.15, 0.70), (-56, 4.0, 0.0030, 0.30, 0.65),
           (54, 4.0, 0.0030, 0.30, 0.60)]
# finer strand grooves between the main ones
FINE = [(v, 1.3, 0.0016, 0.05, 0.9) for v in (-50, -37.5, -24.5, -12, -0.5, 11, 23, 35, 47.5)]
SWEEP = 10.0     # the grooves curve this many degrees toward the back


def cap_thickness(p, phi, theta):
    """Hair thickness over the scalp: close fade on the sides and nape,
    fuller on top, highest in the quiff above the forehead."""
    d = p - C0
    d = d / np.maximum(np.sqrt((d * d).sum(axis=1)), 1e-6)[:, None]
    above = theta - theta_min(phi)
    fade = S.smoothstep(0.0, 9.0, above)
    top = S.smoothstep(0.30, 0.85, d[:, 2])
    front = S.smoothstep(-0.35, 0.55, -d[:, 1])
    back = S.smoothstep(0.2, 0.8, d[:, 1]) * (1.0 - top) * S.smoothstep(-0.55, -0.1, d[:, 2])
    t = 0.0026 + 0.0022 * back + top * (0.0085 + front * 0.0125)
    # quiff: the front lifts quickly off the hairline
    lift = S.smoothstep(0.0, 14.0, above) * front * top
    t = t + 0.003 * lift
    # grooves along the sweep (lines of constant angle around the y axis)
    v = np.degrees(np.arctan2(d[:, 0], np.maximum(d[:, 2], 1e-3) + 0.25))
    u = np.clip((d[:, 1] + 0.75) / 1.5, 0.0, 1.0)          # 0 at the front, 1 at the back
    g = np.zeros(len(p), dtype=F32)
    for v0, hw, depth, u0, u1 in GROOVES + FINE:
        vv = v - (v0 + SWEEP * np.sign(v0) * u)
        prof = 1.0 - S.smoothstep(0.0, hw, np.abs(vv))
        span = S.smoothstep(u0, u0 + 0.12, u) * (1.0 - S.smoothstep(u1 - 0.12, u1, u))
        g = np.maximum(g, F32(depth) * prof * span)
    t = t - g * top * S.smoothstep(4.0, 16.0, above)
    return fade * t - F32(0.0006) * (1.0 - fade)


def cap(p):
    phi, theta = angles(p)
    sc = scalp(p)
    d = sc - cap_thickness(p, phi, theta)
    # keep only a shell (hollow under the skin) and stop at the hairline
    edge = (theta - theta_min(phi) + 1.5) * F32(0.0015)
    d = S.smax(d, -edge, 0.002)
    return S.smax(d, -(sc + F32(0.003)), 0.001)


# -- clumps ------------------------------------------------------------------------------

def _direction(phi, theta):
    p, t = math.radians(phi), math.radians(theta)
    return np.array([math.sin(p) * math.cos(t), -math.cos(p) * math.cos(t), math.sin(t)])


def on_scalp(phi, theta, height):
    """Scalp point (local) at the given angles, lifted along the normal."""
    d = _direction(phi, theta)
    p = (C0 + d * 0.12).astype(F32)[None, :]
    for _ in range(40):
        v = scalp(p)
        if abs(float(v[0])) < 1e-6:
            break
        p = p - (v[:, None] * d.astype(F32)[None, :])     # march along the ray
    n = S.gradient(scalp, p, 2e-4)[0]
    n = n / np.linalg.norm(n)
    return p[0] + n * height, n


def _catmull(points, samples):
    pts = [points[0] + (points[0] - points[1])] + list(points) + [points[-1] + (points[-1] - points[-2])]
    out = []
    segs = len(points) - 1
    for s in range(samples):
        t = s / (samples - 1) * segs
        i = min(int(t), segs - 1)
        u = t - i
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u))
    return np.array(out)


class Clump:
    """A tapered ribbon of hair along a curve, with an elliptical section
    whose flat side faces the scalp."""

    def __init__(self, path, width, thick, samples=14, root=0.55, tip=0.08):
        ctrl, nrm = zip(*(on_scalp(*c) for c in path))
        self.pts = _catmull(np.array(ctrl), samples).astype(F32)
        n = _catmull(np.array(nrm), samples)
        self.nrm = (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(F32)
        t = np.linspace(0.0, 1.0, samples)
        # width swells after the root and tapers to the tip
        prof = (root + (1 - root) * np.sin(np.pi * np.minimum(t * 1.6, 1.0) / 2)) * (1 - t * (1 - tip)) ** 0.9
        self.w = (width * prof).astype(F32)
        self.h = (thick * np.sqrt(prof)).astype(F32)
        pad = float(max(width, thick)) + 0.006
        self.lo = self.pts.min(axis=0) - pad
        self.hi = self.pts.max(axis=0) + pad

    def __call__(self, p):
        return S.boxed(p, self.lo, self.hi, self._field)

    def _field(self, p):
        best = np.full(len(p), S.FAR, dtype=F32)
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            ba = b - a
            L2 = F32(ba @ ba)
            t = np.clip(((p - a) @ ba) / L2, 0.0, 1.0)
            o = p - (a + t[:, None] * ba)
            tan = ba / np.sqrt(L2)
            n = self.nrm[i] * (1 - t[:, None]) + self.nrm[i + 1] * t[:, None]
            n = n - (n @ tan)[:, None] * tan
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            bv = np.cross(tan[None, :], n)
            w = self.w[i] * (1 - t) + self.w[i + 1] * t
            h = self.h[i] * (1 - t) + self.h[i + 1] * t
            x = (o * bv).sum(axis=1)
            y = (o * n).sum(axis=1)
            z = o @ tan
            rc = np.minimum(w, h)
            k0 = np.sqrt((x / w) ** 2 + (y / h) ** 2 + (z / rc) ** 2)
            k1 = np.sqrt((x / (w * w)) ** 2 + (y / (h * h)) ** 2 + (z / (rc * rc)) ** 2)
            best = np.minimum(best, k0 * (k0 - 1.0) / np.maximum(k1, 1e-9))
        return best


def _front_clumps():
    """A few chunky locks at the front edge of the quiff, so the silhouette
    breaks into hair masses instead of a smooth helmet edge.
    (phi, root theta, crest height, sweep, width) per lock."""
    specs = [(-22, 27.0, 0.019, -6, 0.0075), (-9, 28.0, 0.022, -2, 0.0080), (4, 28.0, 0.023, 3, 0.0085),
             (17, 27.5, 0.021, 7, 0.0080), (29, 26.0, 0.017, 10, 0.0070)]
    out = []
    for phi, th0, crest, sweep, width in specs:
        path = [(phi, th0, 0.004), (phi + sweep * 0.3, th0 + 6.0, crest * 0.85),
                (phi + sweep * 0.7, th0 + 15.0, crest), (phi + sweep, th0 + 26.0, crest * 0.75)]
        out.append(Clump(path, width, 0.0040, samples=12, root=0.7, tip=0.15))
    return out


_CLUMPS = None


def clumps():
    global _CLUMPS
    if _CLUMPS is None:
        _CLUMPS = []
    return _CLUMPS


def field_local(p):
    d = cap(p)
    for c in clumps():
        d = S.smin(d, c(p), 0.0025)
    return d


def field(p):
    s = F32(H.D.scale)
    q = (p - np.array([0.0, 0.0, H.EYE_Z], dtype=F32)) / s
    return field_local(q) * s


def bounds():
    s = H.D.scale
    lo = np.array([-0.100, -0.125, -0.110]) * s + np.array([0, 0, H.EYE_Z])
    hi = np.array([0.100, 0.135, 0.165]) * s + np.array([0, 0, H.EYE_Z])
    return lo, hi
