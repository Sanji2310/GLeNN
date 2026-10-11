"""Signed distance fields in plain numpy, plus a surface-nets mesher.

This is the sculpting layer for the GLeNN character. Forms are described as
signed distance functions (negative inside, positive outside) and blended
with smooth unions and subtractions, the way a sculptor adds and carves
clay. The result is turned into a quad mesh with surface nets, and every
vertex is then projected exactly onto the surface.

The design form is not the production mesh. Clean, animation-ready topology
is projected onto the same field later (see project()).

Everything here is pure numpy, so the same code runs in the cloud test
environment and in Blender 5.2.2 on the artist's PC.

Conventions: metres, character facing -Y, +Z up. Points are (N, 3) float32
arrays. Distance functions return (N,) arrays.
"""

import math

import numpy as np

F = np.float32


# -- small helpers ------------------------------------------------------------

def vec(x):
    return np.asarray(x, dtype=F)


def length(v):
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def rot(rx=0.0, ry=0.0, rz=0.0):
    """Rotation matrix (local to world) from XYZ Euler angles in degrees."""
    x, y, z = (math.radians(a) for a in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (mz @ my @ mx).astype(F)


def local(p, c, R=None):
    q = p - vec(c)
    return q if R is None else q @ R


def mirror_x(p):
    """Fold space so a feature defined on +X also appears on -X."""
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0])
    return q


# -- primitives -----------------------------------------------------------------

def sphere(p, c, r):
    return length(p - vec(c)) - F(r)


def ellipsoid(p, c, radii, R=None):
    """Inigo Quilez's bound for an ellipsoid (exact on the surface)."""
    r = vec(radii)
    q = local(p, c, R)
    k0 = length(q / r)
    k1 = length(q / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, F(1e-9))


def round_box(p, c, half, radius, R=None):
    q = np.abs(local(p, c, R)) - vec(half) + F(radius)
    return length(np.maximum(q, 0.0)) + np.minimum(q.max(axis=1), 0.0) - F(radius)


def capsule(p, a, b, r):
    a, b = vec(a), vec(b)
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / F(ba @ ba), 0.0, 1.0)
    return length(pa - h[:, None] * ba) - F(r)


def round_cone(p, a, b, r1, r2):
    """Capsule whose radius goes from r1 at a to r2 at b (exact)."""
    a, b = vec(a), vec(b)
    ba = b - a
    l2 = F(ba @ ba)
    rr = F(r1 - r2)
    a2 = l2 - rr * rr
    il2 = F(1.0) / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    w = pa * l2 - y[:, None] * ba
    x2 = np.einsum("ij,ij->i", w, w)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d_mid = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    d_b = np.sqrt(x2 + z2) * il2 - r2
    d_a = np.sqrt(x2 + y2) * il2 - r1
    out = np.where(np.sign(y) * a2 * y2 < k, d_a, d_mid)
    return np.where(np.sign(z) * a2 * z2 > k, d_b, out)


def tube(p, points, radii, k=0.0):
    """Chain of round cones through points with per-point radii."""
    d = None
    for i in range(len(points) - 1):
        di = round_cone(p, points[i], points[i + 1], radii[i], radii[i + 1])
        d = di if d is None else (np.minimum(d, di) if k <= 0 else smin(d, di, k))
    return d


def plane(p, normal, offset):
    """Half-space n.p <= offset (n is normalised here)."""
    n = vec(normal)
    n = n / np.linalg.norm(n)
    return p @ n - F(offset)


def torus(p, c, R_major, r_minor, R=None):
    q = local(p, c, R)
    xy = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) - F(R_major)
    return np.sqrt(xy * xy + q[:, 2] ** 2) - F(r_minor)


FAR = F(1e3)


def boxed(p, lo, hi, fn):
    """Evaluate a local feature only for points inside its box [lo, hi].

    Outside the box the feature is treated as far away (+FAR), which leaves a
    union or subtraction unchanged as long as the box has a margin wider than
    the blend radius. This keeps small features (nose, lips, ears) cheap."""
    lo, hi = vec(lo), vec(hi)
    inside = np.all((p >= lo) & (p <= hi), axis=1)
    out = np.full(len(p), FAR, dtype=F)
    idx = np.nonzero(inside)[0]
    if len(idx):
        out[idx] = fn(p[idx])
    return out


# -- blending -----------------------------------------------------------------

def smin(a, b, k):
    """Polynomial smooth minimum: a soft union with blend width k (metres)."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(F(k) - np.abs(a - b), 0.0) / F(k)
    return np.minimum(a, b) - h * h * F(k * 0.25)


def smax(a, b, k):
    return -smin(-a, -b, k)


def union(a, b, k=0.0):
    return smin(a, b, k)


def subtract(a, b, k=0.0):
    """Carve b out of a."""
    return smax(a, -b, k)


def intersect(a, b, k=0.0):
    return smax(a, b, k)


def shell(d, thickness):
    return np.abs(d) - F(thickness * 0.5)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# -- evaluation and meshing -----------------------------------------------------

def grid_axes(lo, hi, h):
    lo, hi = vec(lo), vec(hi)
    n = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    return [lo[i] + F(h) * np.arange(n[i], dtype=F) for i in range(3)]


def evaluate_grid(f, axes, chunk=400_000):
    """Sample f on the grid, slab by slab along X to bound memory."""
    xs, ys, zs = axes
    ny, nz = len(ys), len(zs)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    yz = np.stack([yy.ravel(), zz.ravel()], axis=1)
    out = np.empty((len(xs), ny, nz), dtype=F)
    per = max(1, chunk // (ny * nz))
    for i0 in range(0, len(xs), per):
        sl = xs[i0:i0 + per]
        pts = np.empty((len(sl) * ny * nz, 3), dtype=F)
        pts[:, 0] = np.repeat(sl, ny * nz)
        pts[:, 1:] = np.tile(yz, (len(sl), 1))
        out[i0:i0 + len(sl)] = f(pts).reshape(len(sl), ny, nz)
    return out


def evaluate_banded(f, axes, block=4, band=1.5, chunk=400_000):
    """Sample f finely only near the surface.

    A coarse grid (every block-th node) is evaluated first. Fine nodes are
    evaluated only inside coarse cells whose corners are within band x the
    cell diagonal of the surface; elsewhere the nearest coarse value is used,
    which always has the right sign there."""
    n = [len(a) for a in axes]
    ci = []
    for k in range(3):
        idx = np.arange(0, n[k], block)
        if idx[-1] != n[k] - 1:
            idx = np.append(idx, n[k] - 1)
        ci.append(idx)
    coarse = evaluate_grid(f, [axes[k][ci[k]] for k in range(3)], chunk)
    h = float(axes[0][1] - axes[0][0])
    limit = band * block * h * math.sqrt(3.0)
    near = np.abs(coarse) < limit
    sign = coarse < 0
    # a coarse cell needs fine sampling if any corner is near or the sign changes
    cell = np.zeros(tuple(s - 1 for s in coarse.shape), dtype=bool)
    any_in = np.zeros_like(cell)
    all_in = np.ones_like(cell)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                sl = (slice(dx, dx + cell.shape[0]), slice(dy, dy + cell.shape[1]), slice(dz, dz + cell.shape[2]))
                cell |= near[sl]
                any_in |= sign[sl]
                all_in &= sign[sl]
    cell |= any_in & ~all_in
    # nearest coarse value everywhere
    nearest = [np.searchsorted(ci[k], np.arange(n[k]) - block / 2.0).clip(0, len(ci[k]) - 1) for k in range(3)]
    out = coarse[np.ix_(*nearest)]
    # fine node mask: every node inside or on the boundary of a marked cell
    owner = [np.minimum(np.searchsorted(ci[k], np.arange(n[k]), side="right") - 1, cell.shape[k] - 1)
             for k in range(3)]
    mask = cell[np.ix_(*owner)]
    for k in range(3):
        on_edge = np.isin(np.arange(n[k]), ci[k][1:-1])
        prev = np.clip(np.array(owner[k]) - 1, 0, None)
        idx = [owner[0], owner[1], owner[2]]
        idx[k] = prev
        shifted = cell[np.ix_(*idx)]
        sel = [slice(None)] * 3
        sel[k] = on_edge
        mask[tuple(sel)] |= shifted[tuple(sel)]
    ii = np.argwhere(mask)
    for s in range(0, len(ii), chunk):
        part = ii[s:s + chunk]
        pts = np.stack([axes[0][part[:, 0]], axes[1][part[:, 1]], axes[2][part[:, 2]]], axis=1).astype(F)
        out[part[:, 0], part[:, 1], part[:, 2]] = f(pts)
    return out


def surface_nets(field, axes):
    """Quad mesh of the zero level set of a sampled field.

    One vertex per cell that the surface crosses (the mean of the edge
    crossings), one quad per crossing grid edge. Returns (verts, quads)."""
    xs, ys, zs = axes
    h = np.array([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]], dtype=F)
    lo = np.array([xs[0], ys[0], zs[0]], dtype=F)
    nx, ny, nz = field.shape
    inside = field < 0
    cshape = (nx - 1, ny - 1, nz - 1)
    acc = np.zeros(cshape + (3,), dtype=F)
    cnt = np.zeros(cshape, dtype=np.int32)
    edges = []
    for axis in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[axis] = slice(0, -1)
        s1[axis] = slice(1, None)
        f0, f1 = field[tuple(s0)], field[tuple(s1)]
        cross = inside[tuple(s0)] != inside[tuple(s1)]
        idx = np.argwhere(cross)
        if not len(idx):
            edges.append((idx, None))
            continue
        a, b = f0[cross], f1[cross]
        t = a / (a - b)
        pos = idx.astype(F) * h + lo
        pos[:, axis] += t * h[axis]
        # each edge touches up to four cells
        o1, o2 = [ax for ax in range(3) if ax != axis]
        for d1 in (0, 1):
            for d2 in (0, 1):
                c = idx.copy()
                c[:, o1] -= d1
                c[:, o2] -= d2
                ok = (c[:, o1] >= 0) & (c[:, o1] < cshape[o1]) & (c[:, o2] >= 0) & (c[:, o2] < cshape[o2]) \
                    & (c[:, axis] < cshape[axis])
                cc = c[ok]
                np.add.at(acc, (cc[:, 0], cc[:, 1], cc[:, 2]), pos[ok])
                np.add.at(cnt, (cc[:, 0], cc[:, 1], cc[:, 2]), 1)
        edges.append((idx, inside[tuple(s0)][cross]))
    active = cnt > 0
    vid = np.full(cshape, -1, dtype=np.int64)
    vid[active] = np.arange(int(active.sum()))
    verts = acc[active] / cnt[active][:, None]
    quads = []
    for axis, (idx, first_inside) in enumerate(edges):
        if first_inside is None:
            continue
        o1, o2 = [ax for ax in range(3) if ax != axis]
        ok = (idx[:, o1] >= 1) & (idx[:, o2] >= 1) & (idx[:, o1] < cshape[o1]) & (idx[:, o2] < cshape[o2]) \
            & (idx[:, axis] < cshape[axis])
        e, fi = idx[ok], first_inside[ok]
        corners = []
        for d1, d2 in ((1, 1), (0, 1), (0, 0), (1, 0)):
            c = e.copy()
            c[:, o1] -= d1
            c[:, o2] -= d2
            corners.append(vid[c[:, 0], c[:, 1], c[:, 2]])
        q = np.stack(corners, axis=1)
        # orient so normals point out of the solid
        flip = fi if axis == 1 else ~fi
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    quads = np.concatenate(quads) if quads else np.zeros((0, 4), dtype=np.int64)
    return verts, quads


def gradient(f, p, eps=2e-4):
    g = np.empty_like(p)
    for i in range(3):
        o = np.zeros(3, dtype=F)
        o[i] = eps
        g[:, i] = (f(p + o) - f(p - o)) / F(2 * eps)
    return g


def project(f, p, iterations=3, eps=2e-4, max_step=None):
    """Move points onto the zero level set with Newton steps."""
    p = p.astype(F).copy()
    for _ in range(iterations):
        d = f(p)
        g = gradient(f, p, eps)
        g2 = np.maximum(np.einsum("ij,ij->i", g, g), F(1e-12))
        step = (d / g2)[:, None] * g
        if max_step:
            n = length(step)
            step *= np.minimum(1.0, max_step / np.maximum(n, 1e-12))[:, None]
        p -= step
    return p


def relax(verts, quads, iterations=2, factor=0.5):
    """Uniform Laplacian smoothing over quad edges (use before project())."""
    e = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
    a, b = e[:, 0], e[:, 1]
    v = verts.copy()
    n = len(v)
    for _ in range(iterations):
        acc = np.zeros_like(v)
        cnt = np.zeros(n, dtype=F)
        np.add.at(acc, a, v[b])
        np.add.at(acc, b, v[a])
        np.add.at(cnt, a, 1)
        np.add.at(cnt, b, 1)
        v += (acc / np.maximum(cnt, 1)[:, None] - v) * F(factor)
    return v


def mesh_field(f, lo, hi, voxel, relax_iters=2, project_iters=3):
    """Field to a clean quad mesh: sample, surface nets, relax, project."""
    axes = grid_axes(lo, hi, voxel)
    field = evaluate_banded(f, axes)
    verts, quads = surface_nets(field, axes)
    # drop unused vertices (cells on the outer boundary that made no quad)
    used = np.zeros(len(verts), dtype=bool)
    used[quads.ravel()] = True
    remap = np.cumsum(used) - 1
    verts, quads = verts[used], remap[quads]
    if relax_iters:
        verts = relax(verts, quads, relax_iters)
    if project_iters:
        verts = project(f, verts, project_iters, eps=voxel * 0.25, max_step=voxel)
    return verts, quads


def to_blender_mesh(name, verts, quads, smooth=True):
    """Build a bpy mesh quickly with foreach_set."""
    import bpy
    me = bpy.data.meshes.get(name)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.ascontiguousarray(verts, dtype=np.float32).ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", np.ascontiguousarray(quads, dtype=np.int32).ravel())
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    if smooth:
        me.shade_smooth()
    return me
