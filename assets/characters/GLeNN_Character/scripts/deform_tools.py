"""Topology-preserving shape edits for restyling a base mesh.

Every function moves existing vertices only, so edge flow, UVs and vertex
groups survive. Edits are symmetric by default: a region defined on the
character's left side (+X) is mirrored to the right.

All coordinates are in the object's local space (metres), with the character
facing -Y and +Z up. Typical use:

    E = Edits(obj)
    E.scale_region(center, radius, (1.1, 1.0, 1.0))
    E.push(center, radius, (0, -0.004, 0))
    E.commit()
"""

import math

import numpy as np


def _falloff(d, radius, kind="smooth"):
    t = np.clip(1.0 - d / radius, 0.0, 1.0)
    if kind == "smooth":
        return t * t * (3.0 - 2.0 * t)
    if kind == "sphere":
        return np.sqrt(1.0 - (1.0 - t) ** 2)
    return t


class Edits:
    """Batch vertex edits on a mesh object, applied with commit()."""

    def __init__(self, obj):
        self.obj = obj
        n = len(obj.data.vertices)
        co = np.empty(n * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", co)
        self.co = co.reshape(n, 3)
        self.orig = self.co.copy()

    # -- region helpers -------------------------------------------------------
    def _weights(self, center, radius, kind="smooth", ellipsoid=(1.0, 1.0, 1.0), mirror=True):
        c = np.asarray(center, dtype=np.float64)
        e = np.asarray(ellipsoid, dtype=np.float64)
        d = np.linalg.norm((self.co - c) / e, axis=1)
        w = _falloff(d, radius, kind)
        if mirror and abs(c[0]) > 1e-6:
            cm = c * np.array([-1.0, 1.0, 1.0])
            dm = np.linalg.norm((self.co - cm) / e, axis=1)
            wm = _falloff(dm, radius, kind)
            return w, wm, cm
        return w, None, None

    # -- operations -------------------------------------------------------------
    def push(self, center, radius, offset, kind="smooth", ellipsoid=(1, 1, 1), mirror=True):
        """Translate a soft region by offset (mirrored region gets -x)."""
        off = np.asarray(offset, dtype=np.float64)
        w, wm, _ = self._weights(center, radius, kind, ellipsoid, mirror)
        self.co += w[:, None] * off
        if wm is not None:
            self.co += wm[:, None] * (off * np.array([-1.0, 1.0, 1.0]))

    def scale_region(self, center, radius, factors, kind="smooth", ellipsoid=(1, 1, 1), mirror=True,
                     pivot=None):
        """Scale a soft region about its centre (or a pivot)."""
        f = np.asarray(factors, dtype=np.float64)
        c = np.asarray(center if pivot is None else pivot, dtype=np.float64)
        w, wm, cm = self._weights(center, radius, kind, ellipsoid, mirror)
        self.co += w[:, None] * ((self.co - c) * (f - 1.0))
        if wm is not None:
            pm = c * np.array([-1.0, 1.0, 1.0])
            self.co += wm[:, None] * ((self.co - pm) * (f - 1.0))

    def inflate(self, center, radius, amount, normals, kind="smooth", ellipsoid=(1, 1, 1), mirror=True):
        """Move along the original vertex normals (positive = outward)."""
        w, wm, _ = self._weights(center, radius, kind, ellipsoid, mirror)
        tot = w if wm is None else np.maximum(w, wm)
        self.co += tot[:, None] * normals * amount

    def axis_scale(self, axis, lo, hi, factor_lo, factor_hi, scale_axes=(0, 1), pivot=(0.0, 0.0, 0.0)):
        """Scale chosen axes by a factor that blends along another axis.
        E.g. widen the shoulders: axis=2 (z), from z=lo to z=hi."""
        t = np.clip((self.co[:, axis] - lo) / (hi - lo), 0.0, 1.0)
        t = t * t * (3 - 2 * t)
        f = factor_lo + (factor_hi - factor_lo) * t
        p = np.asarray(pivot, dtype=np.float64)
        for a in scale_axes:
            self.co[:, a] = p[a] + (self.co[:, a] - p[a]) * f

    def mask(self, predicate):
        """Boolean mask from a function of the coordinate array."""
        return predicate(self.co)

    def smooth(self, mesh_edges, iterations=2, factor=0.5, weights=None):
        """Laplacian smoothing over the current shape (optionally masked)."""
        n = len(self.co)
        a, b = mesh_edges[:, 0], mesh_edges[:, 1]
        for _ in range(iterations):
            acc = np.zeros_like(self.co)
            cnt = np.zeros(n)
            np.add.at(acc, a, self.co[b])
            np.add.at(acc, b, self.co[a])
            np.add.at(cnt, a, 1)
            np.add.at(cnt, b, 1)
            avg = acc / np.maximum(cnt, 1)[:, None]
            w = factor if weights is None else factor * weights[:, None]
            self.co += (avg - self.co) * w

    def symmetrize(self, tol=1e-4):
        """Copy the +X side onto the -X side using nearest mirrored partners
        in the original shape, so tiny asymmetries from edits cancel out."""
        from mathutils.kdtree import KDTree
        kd = KDTree(len(self.orig))
        for i, c in enumerate(self.orig):
            kd.insert(c, i)
        kd.balance()
        for i, c in enumerate(self.orig):
            if c[0] < -tol:
                _, j, d = kd.find((-c[0], c[1], c[2]))
                if d is not None and d < 0.002:
                    self.co[i] = self.co[j] * np.array([-1.0, 1.0, 1.0])
            elif abs(c[0]) <= tol:
                self.co[i][0] = 0.0

    def commit(self):
        self.obj.data.vertices.foreach_set("co", self.co.ravel())
        self.obj.data.update()


def edge_array(mesh):
    e = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def normal_array(mesh):
    n = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("normal", n)
    return n.reshape(-1, 3)
