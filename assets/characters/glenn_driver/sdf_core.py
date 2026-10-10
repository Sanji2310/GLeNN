"""Signed-distance modelling core for the GLeNN driver build.

Pure numpy. Every shape in the character is described as a signed distance
field (negative inside), then turned into a watertight quad mesh with a
narrow-band surface-nets mesher and snapped back onto the exact surface.
This lets the character be "sculpted" from primary / secondary / tertiary
forms in code and rebuilt in any pose.
"""

import math
import numpy as np

# --------------------------------------------------------------------------
# vector helpers
# --------------------------------------------------------------------------

def v3(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def length(q):
    return np.sqrt(np.einsum("ij,ij->i", q, q))


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rot_euler(x=0.0, y=0.0, z=0.0):
    """Rotation applied X then Y then Z (Blender 'XYZ')."""
    return rot_z(z) @ rot_y(y) @ rot_x(x)


def rot_axis(axis, deg):
    axis = norm(axis)
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = axis
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def frame_from(fwd, up_hint):
    """Orthonormal frame (columns x, y, z) with z along fwd."""
    z = norm(fwd)
    x = np.cross(up_hint, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(v3(1, 0, 0), z)
    x = norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# primitives (P: (N,3) world points)
# --------------------------------------------------------------------------

def to_local(P, c, R=None):
    q = P - c
    if R is not None:
        q = q @ R  # R columns are local axes in world space
    return q


def sd_sphere(P, c, r):
    return length(P - c) - r


def sd_ellipsoid(P, c, r, R=None):
    q = to_local(P, c, R)
    r = np.asarray(r, dtype=np.float64)
    k0 = length(q / r)
    k1 = length(q / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)


def sd_capsule(P, a, b, r):
    pa = P - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return length(pa - h[:, None] * ba) - r


def sd_round_cone(P, a, b, r1, r2):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ba = b - a
    l2 = ba @ ba
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = P - a
    y = pa @ ba
    z = y - l2
    xv = pa * l2 - y[:, None] * ba
    x2 = np.einsum("ij,ij->i", xv, xv)
    y2 = y * y * l2
    z2 = z * z * l2
    k = math.copysign(1.0, rr) * rr * rr * x2
    d_end = np.sqrt(x2 + z2) * il2 - r2
    d_start = np.sqrt(x2 + y2) * il2 - r1
    d_mid = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    out = np.where(np.sign(y) * a2 * y2 < k, d_start, d_mid)
    out = np.where(np.sign(z) * a2 * z2 > k, d_end, out)
    return out


def sd_box(P, c, b, r=0.0, R=None):
    q = np.abs(to_local(P, c, R)) - (np.asarray(b) - r)
    return length(np.maximum(q, 0.0)) + np.minimum(q.max(axis=1), 0.0) - r


def sd_torus(P, c, R_major, r_minor, R=None):
    """Torus in the local XY plane."""
    q = to_local(P, c, R)
    ring = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) - R_major
    return np.sqrt(ring * ring + q[:, 2] ** 2) - r_minor


def sd_cyl(P, c, r, h, R=None):
    """Capped cylinder along local Z, half height h."""
    q = to_local(P, c, R)
    dx = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) - r
    dz = np.abs(q[:, 2]) - h
    return np.minimum(np.maximum(dx, dz), 0.0) + np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dz, 0) ** 2)


def sd_ellipse2(u, v, a, b):
    """Approximate 2D ellipse distance."""
    k0 = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    k1 = np.sqrt((u / (a * a)) ** 2 + (v / (b * b)) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)


def sd_plane(P, p0, n):
    return (P - p0) @ norm(n)


# --------------------------------------------------------------------------
# booleans
# --------------------------------------------------------------------------

def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def ssub(a, b, k):
    """a minus b (smooth)."""
    return smax(a, -b, k)


def raised(d_base, region, height, k_edge=0.002, k_join=0.0):
    """Raise the surface by `height` inside `region` (an sdf, <0 inside)."""
    pad = smax(d_base - height, region, k_edge)
    return smin(d_base, pad, k_join) if k_join > 0 else np.minimum(d_base, pad)


def engrave(d_base, region, depth, k=0.001):
    """Cut a groove of `depth` where region < 0."""
    cutter = smax(region, -(d_base + depth), k)
    return smax(d_base, -cutter, k)


def softmin_weights(ds, tau):
    D = np.stack(ds, axis=0)
    m = D.min(axis=0)
    W = np.exp(-(D - m) / tau)
    return W / W.sum(axis=0)


# --------------------------------------------------------------------------
# noise (value noise, deterministic) for tertiary breakup
# --------------------------------------------------------------------------

def _hash3(ix, iy, iz, seed=0):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float64) / 65535.0


def value_noise(P, scale, seed=0):
    q = P * scale
    i = np.floor(q).astype(np.int64)
    f = q - i
    u = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1]) * (u[:, 2] if dz else 1 - u[:, 2])
                out = out + w * _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out * 2.0 - 1.0


# --------------------------------------------------------------------------
# narrow-band surface nets
# --------------------------------------------------------------------------

def _eval_chunked(f, P, chunk=400_000):
    out = np.empty(len(P), dtype=np.float64)
    for s in range(0, len(P), chunk):
        out[s:s + chunk] = f(P[s:s + chunk])
    return out


def sample_grid(f, bmin, bmax, h, coarse=4):
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    n = np.ceil((bmax - bmin) / h).astype(int) + 1
    n = ((n + coarse - 1) // coarse) * coarse + 1
    nx, ny, nz = n
    # coarse pass
    cx = np.arange(0, nx, coarse)
    cy = np.arange(0, ny, coarse)
    cz = np.arange(0, nz, coarse)
    G = np.stack(np.meshgrid(cx, cy, cz, indexing="ij"), -1).reshape(-1, 3)
    dc = _eval_chunked(f, bmin + G * h).reshape(len(cx), len(cy), len(cz)).astype(np.float32)
    # nearest coarse node for every fine node
    ix = np.minimum((np.arange(nx) + coarse // 2) // coarse, len(cx) - 1)
    iy = np.minimum((np.arange(ny) + coarse // 2) // coarse, len(cy) - 1)
    iz = np.minimum((np.arange(nz) + coarse // 2) // coarse, len(cz) - 1)
    D = dc[ix][:, iy][:, :, iz]
    thr = (h + coarse * h * math.sqrt(3)) * 1.6
    idx = np.nonzero(np.abs(D) < thr)
    if len(idx[0]):
        Pf = bmin + np.stack(idx, -1) * h
        D[idx] = _eval_chunked(f, Pf).astype(np.float32)
    return D


def surface_nets(D, origin, h):
    S = D < 0
    nx, ny, nz = D.shape
    ncx, ncy, ncz = nx - 1, ny - 1, nz - 1
    acc = np.zeros((ncx * ncy * ncz, 3), dtype=np.float64)
    cnt = np.zeros(ncx * ncy * ncz, dtype=np.int32)

    def cid(i, j, k):
        return (i * ncy + j) * ncz + k

    quads = []
    for axis in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, -1)
        sl1[axis] = slice(1, None)
        s0 = S[tuple(sl0)]
        s1 = S[tuple(sl1)]
        e = np.nonzero(s0 != s1)
        i, j, k = e
        d0 = D[tuple(sl0)][e].astype(np.float64)
        d1 = D[tuple(sl1)][e].astype(np.float64)
        t = d0 / (d0 - d1)
        pos = np.stack([i, j, k], -1).astype(np.float64)
        pos[:, axis] += t
        pt = origin + pos * h
        # adjacent cells: vary the two other axes by -1/0
        a1, a2 = [a for a in range(3) if a != axis]
        idx = [i, j, k]
        corners = []
        for o1 in (-1, 0):
            for o2 in (-1, 0):
                c = [idx[0].copy(), idx[1].copy(), idx[2].copy()]
                c[a1] = c[a1] + o1
                c[a2] = c[a2] + o2
                corners.append(c)
        for c in corners:
            ok = (c[0] >= 0) & (c[0] < ncx) & (c[1] >= 0) & (c[1] < ncy) & (c[2] >= 0) & (c[2] < ncz)
            ids = cid(c[0][ok], c[1][ok], c[2][ok])
            np.add.at(acc, ids, pt[ok])
            np.add.at(cnt, ids, 1)
        # quads (interior edges only)
        valid = np.ones(len(i), dtype=bool)
        for c in corners:
            valid &= (c[0] >= 0) & (c[0] < ncx) & (c[1] >= 0) & (c[1] < ncy) & (c[2] >= 0) & (c[2] < ncz)
        inside_first = s0[e][valid]
        c00, c01, c10, c11 = [cid(c[0][valid], c[1][valid], c[2][valid]) for c in corners]
        # corners order: (o1,o2) = (-1,-1), (-1,0), (0,-1), (0,0)
        # ccw in (a1,a2): (-1,-1) -> (0,-1) -> (0,0) -> (-1,0)  => normal +axis (a1 x a2 = axis for cyclic order)
        q = np.stack([c00, c10, c11, c01], -1)
        # (a1, a2) = cyclic successors of axis -> a1 x a2 == +axis only for axis 0 and 2 ordering; fix for axis 1
        cyclic = (a1 == (axis + 1) % 3)
        flip = ~inside_first if cyclic else inside_first
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    used = cnt > 0
    remap = -np.ones(len(cnt), dtype=np.int64)
    remap[used] = np.arange(used.sum())
    V = acc[used] / cnt[used][:, None]
    F = remap[np.concatenate(quads, 0)]
    return V, F


def grad(f, P, eps):
    g = np.zeros_like(P)
    for a in range(3):
        o = np.zeros(3)
        o[a] = eps
        g[:, a] = (f(P + o) - f(P - o)) / (2 * eps)
    return g


def project(f, V, eps, iters=3, chunk=200_000):
    V = V.copy()
    for s in range(0, len(V), chunk):
        P = V[s:s + chunk]
        for _ in range(iters):
            d = f(P)
            g = grad(f, P, eps)
            gg = np.maximum(np.einsum("ij,ij->i", g, g), 1e-12)
            step = (d / gg)[:, None] * g
            # clamp step to avoid jumping across thin features
            sl = length(step)
            m = sl > eps * 4
            step[m] *= (eps * 4 / sl[m])[:, None]
            P = P - step
        V[s:s + chunk] = P
    return V


def vertex_neighbors_mean(V, F):
    n = len(V)
    acc = np.zeros_like(V)
    cnt = np.zeros(n)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        np.add.at(acc, F[:, a], V[F[:, b]])
        np.add.at(cnt, F[:, a], 1)
        np.add.at(acc, F[:, b], V[F[:, a]])
        np.add.at(cnt, F[:, b], 1)
    return acc / np.maximum(cnt, 1)[:, None]


def relax(f, V, F, eps, iters=2, amount=0.5):
    for _ in range(iters):
        M = vertex_neighbors_mean(V, F)
        V = V + amount * (M - V)
        V = project(f, V, eps, iters=2)
    return V


def mesh_sdf(f, bmin, bmax, h, relax_iters=2, coarse=4):
    D = sample_grid(f, bmin, bmax, h, coarse=coarse)
    V, F = surface_nets(D, np.asarray(bmin, dtype=np.float64), h)
    V = project(f, V, h * 0.25, iters=3)
    if relax_iters:
        V = relax(f, V, F, h * 0.25, iters=relax_iters)
    return V, F


# --------------------------------------------------------------------------
# helpers for sculpting
# --------------------------------------------------------------------------

def local_op(d, P, lo, hi, op):
    """Apply op(d_sub, P_sub) only to points inside the box [lo, hi].

    Detail shapes are only evaluated where they can change the surface,
    which keeps dense meshes fast to build.
    """
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    m = np.all((P >= lo) & (P <= hi), axis=1)
    if not m.any():
        return d
    d = d.copy()
    d[m] = op(d[m], P[m])
    return d


def smooth_table(zs, table, z):
    """Cubic Hermite interpolation of table rows (len(zs) x k) at z (N,)."""
    zs = np.asarray(zs, dtype=np.float64)
    T = np.asarray(table, dtype=np.float64)
    order = np.argsort(zs)
    zs = zs[order]
    T = T[order]
    n = len(zs)
    m = np.zeros_like(T)
    for i in range(n):
        if i == 0:
            m[i] = (T[1] - T[0]) / (zs[1] - zs[0])
        elif i == n - 1:
            m[i] = (T[-1] - T[-2]) / (zs[-1] - zs[-2])
        else:
            m[i] = (T[i + 1] - T[i - 1]) / (zs[i + 1] - zs[i - 1])
    zc = np.clip(z, zs[0], zs[-1])
    k = np.clip(np.searchsorted(zs, zc) - 1, 0, n - 2)
    h = zs[k + 1] - zs[k]
    t = ((zc - zs[k]) / h)[:, None]
    h = h[:, None]
    t2 = t * t
    t3 = t2 * t
    return ((2 * t3 - 3 * t2 + 1) * T[k] + (t3 - 2 * t2 + t) * h * m[k]
            + (-2 * t3 + 3 * t2) * T[k + 1] + (t3 - t2) * h * m[k + 1])


def sd_loft(P, zs, rows, z_lo, z_hi, k_cap=0.01):
    """Superellipse loft along Z.

    rows: (a, yc, y_front, y_back, n_front, n_back) per z level.
    Produces clean planes that read as sculpted forms.
    """
    prm = smooth_table(zs, rows, P[:, 2])
    a, yc, yf, yb, nf, nb = prm.T
    x = np.abs(P[:, 0])
    dy = P[:, 1] - yc
    front = dy < 0
    b = np.where(front, yc - yf, yb - yc)
    n = np.where(front, nf, nb)
    u = x / a
    v = np.abs(dy) / np.maximum(b, 1e-5)
    r = np.power(np.power(u, n) + np.power(v, n), 1.0 / n)
    rho = np.sqrt(x * x + dy * dy)
    d = rho * (r - 1.0) / np.maximum(r, 1e-6)
    cap = np.maximum(P[:, 2] - z_hi, z_lo - P[:, 2])
    return smax(d, cap, k_cap)


class PointBins:
    """Bins query points in a coarse grid so many small shapes can each be
    evaluated only on nearby points (hair clumps, stitches, fingers)."""

    def __init__(self, P, cell=0.01):
        self.P = P
        self.cell = cell
        self.lo = P.min(axis=0) - 1e-9
        ijk = np.floor((P - self.lo) / cell).astype(np.int64)
        self.dims = ijk.max(axis=0) + 1
        key = (ijk[:, 0] * self.dims[1] + ijk[:, 1]) * self.dims[2] + ijk[:, 2]
        self.order = np.argsort(key, kind="stable")
        self.skey = key[self.order]

    def query(self, lo, hi):
        a = np.floor((np.asarray(lo) - self.lo) / self.cell).astype(np.int64)
        b = np.floor((np.asarray(hi) - self.lo) / self.cell).astype(np.int64)
        a = np.maximum(a, 0)
        b = np.minimum(b, self.dims - 1)
        if np.any(b < a):
            return np.zeros(0, dtype=np.int64)
        out = []
        for i in range(a[0], b[0] + 1):
            for j in range(a[1], b[1] + 1):
                k0 = (i * self.dims[1] + j) * self.dims[2] + a[2]
                k1 = (i * self.dims[1] + j) * self.dims[2] + b[2]
                s = np.searchsorted(self.skey, k0, "left")
                e = np.searchsorted(self.skey, k1, "right")
                if e > s:
                    out.append(self.order[s:e])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def union_segments(P, segs, k=0.0, base=None, cell=0.012):
    """Union of many round-cone segments [(a, b, ra, rb), ...] evaluated
    only near each segment. Returns min distance (or smooth-union with base)."""
    d = np.full(len(P), 1.0) if base is None else base.copy()
    if not len(segs):
        return d
    bins = PointBins(P, cell)
    for seg in segs:
        a, b, ra, rb = seg[:4]
        r = max(ra, rb) + k + 0.002
        lo = np.minimum(a, b) - r
        hi = np.maximum(a, b) + r
        idx = bins.query(lo, hi)
        if not len(idx):
            continue
        Q = P[idx]
        if len(seg) > 4 and seg[4] is not None:
            # flattened lock: squash the cross-section along the given normal
            nvec, flat = seg[4], seg[5]
            t = (b - a) / (np.linalg.norm(b - a) + 1e-12)
            nvec = nvec - t * (nvec @ t)
            nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
            q = Q - a
            Q = Q + np.outer(q @ nvec, nvec) * (flat - 1.0)
        if np.linalg.norm(b - a) < 1e-6:
            dd = sd_sphere(Q, a, max(ra, rb))
        else:
            dd = sd_round_cone(Q, a, b, ra, rb)
        d[idx] = smin(d[idx], dd, k) if k > 0 else np.minimum(d[idx], dd)
    return d
