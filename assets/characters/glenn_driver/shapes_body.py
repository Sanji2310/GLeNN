"""Body, racing suit, gloves and boots for the GLeNN driver.

Everything is authored in the rest pose (a relaxed neutral stance, world
space, metres, character facing -Y). A pose is a set of joint rotations; a
posed shape is evaluated by pulling each query point back into the rest
frame of the bone that owns the part, so the same sculpt can be rebuilt in
any stance without skinning artefacts. The rest-space position of every
surface point is also returned so suit panels and stitching stay attached
to the fabric in the shader.
"""

import numpy as np
from sdf_core import (v3, norm, rot_x, rot_y, rot_z, rot_euler, rot_axis, frame_from, length,
                      sd_sphere, sd_ellipsoid, sd_capsule, sd_round_cone, sd_box, sd_cyl,
                      sd_torus, smin, smax, ssub, raised, engrave, smoothstep, value_noise,
                      softmin_weights, local_op)

HEIGHT = 1.74

# --------------------------------------------------------------------------
# skeleton
# --------------------------------------------------------------------------

JOINTS = {
    "pelvis": (None, v3(0.0, 0.000, 0.980)),
    "spine": ("pelvis", v3(0.0, 0.000, 1.090)),
    "chest": ("spine", v3(0.0, 0.000, 1.250)),
    "neck": ("chest", v3(0.0, 0.026, 1.430)),
    "head": ("neck", v3(0.0, 0.026, 1.430)),
}
for s, tag in ((1, "L"), (-1, "R")):
    JOINTS.update({
        f"clav_{tag}": ("chest", v3(s * 0.020, 0.010, 1.420)),
        f"upperarm_{tag}": (f"clav_{tag}", v3(s * 0.180, 0.010, 1.392)),
        f"forearm_{tag}": (f"upperarm_{tag}", v3(s * 0.232, 0.030, 1.118)),
        f"hand_{tag}": (f"forearm_{tag}", v3(s * 0.256, -0.006, 0.872)),
        f"thigh_{tag}": ("pelvis", v3(s * 0.088, 0.000, 0.925)),
        f"shin_{tag}": (f"thigh_{tag}", v3(s * 0.100, -0.006, 0.490)),
        f"foot_{tag}": (f"shin_{tag}", v3(s * 0.118, 0.022, 0.090)),
    })

HEAD_ORIGIN = v3(0.0, -0.006, 1.618)      # head-local origin in rest world space
HEAD_SCALE = 1.14                          # stylised head-to-body ratio


def J(name, side=1):
    p = JOINTS[name][1].copy()
    p[0] *= side
    return p


class Pose:
    """Forward kinematics from per-joint rotations (degrees, XYZ about the
    rest-pose world axes, applied at the joint)."""

    def __init__(self, rot=None, root_offset=(0, 0, 0)):
        self.rot = rot or {}
        self.root_offset = v3(*root_offset)
        self.R = {}
        self.J = {}
        self._solve()

    def _solve(self):
        def solve(name):
            if name in self.R:
                return
            parent, j = JOINTS[name]
            r = self.rot.get(name, (0, 0, 0))
            Rl = rot_euler(*r) if not isinstance(r, np.ndarray) else r
            if parent is None:
                self.R[name] = Rl
                self.J[name] = j + self.root_offset
            else:
                solve(parent)
                self.R[name] = self.R[parent] @ Rl
                self.J[name] = self.apply_parent(parent, j)

        for n in JOINTS:
            solve(n)

    def apply_parent(self, parent, p):
        jp = JOINTS[parent][1]
        return self.J[parent] + self.R[parent] @ (p - jp)

    def to_world(self, bone, p):
        jb = JOINTS[bone][1]
        return self.J[bone] + (np.asarray(p) - jb) @ self.R[bone].T

    def to_rest(self, bone, P):
        jb = JOINTS[bone][1]
        return jb + (P - self.J[bone]) @ self.R[bone]

    def matrix(self, bone):
        """4x4 world matrix mapping rest space -> posed space for a bone."""
        M = np.eye(4)
        M[:3, :3] = self.R[bone]
        jb = JOINTS[bone][1]
        M[:3, 3] = self.J[bone] - self.R[bone] @ jb
        return M


REST = Pose()


# --------------------------------------------------------------------------
# torso (rest space)
# --------------------------------------------------------------------------

def _mx(P):
    Q = P.copy()
    Q[:, 0] = np.abs(Q[:, 0])
    return Q


def torso_rest(P):
    X = P
    M = _mx(P)
    d = sd_ellipsoid(X, v3(0, 0.006, 1.292), v3(0.140, 0.108, 0.168))
    # pecs, read softly through the suit
    d = smin(d, sd_ellipsoid(M, v3(0.060, -0.068, 1.315), v3(0.068, 0.040, 0.056), rot_y(-10)), 0.045)
    # lats and shoulder blades: V-taper
    d = smin(d, sd_ellipsoid(M, v3(0.074, 0.050, 1.268), v3(0.080, 0.062, 0.132), rot_y(8)), 0.05)
    # trapezius slope into the neck and shoulders
    d = smin(d, sd_round_cone(M, v3(0.045, 0.030, 1.432), v3(0.152, 0.014, 1.418), 0.044, 0.041), 0.045)
    # waist and abdomen: narrow, athletic
    d = smin(d, sd_ellipsoid(X, v3(0, -0.006, 1.118), v3(0.126, 0.094, 0.130)), 0.07)
    # pelvis and glutes
    d = smin(d, sd_ellipsoid(X, v3(0, 0.006, 0.962), v3(0.140, 0.100, 0.108)), 0.07)
    d = smin(d, sd_ellipsoid(M, v3(0.060, 0.056, 0.905), v3(0.066, 0.056, 0.080)), 0.045)
    # crotch support
    d = smin(d, sd_ellipsoid(X, v3(0, -0.010, 0.860), v3(0.050, 0.060, 0.050)), 0.03)
    return d


def suit_details_torso(d, P):
    X = P
    M = _mx(P)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    # waist compression folds just above the belt (front and sides)
    fold_mask = smoothstep(1.150, 1.100, z) * smoothstep(1.075, 1.095, z) * smoothstep(0.06, -0.02, y)
    d = d - 0.0016 * fold_mask * np.sin((z - 1.09) * 260.0 + 0.6 * np.sin(x * 40.0)) * smoothstep(0.02, 0.10, np.abs(x) + 0.04)
    # lower-back accordion stretch panel
    back = smoothstep(0.03, 0.07, y) * smoothstep(1.095, 1.105, z) * smoothstep(1.225, 1.205, z) * smoothstep(0.075, 0.055, np.abs(x))
    d = d - 0.0011 * back * (0.5 + 0.5 * np.sin((z - 1.1) * 2.0 * np.pi / 0.012))
    # belt: raised waistband
    belt = np.abs(z - 1.060) - 0.026
    d = raised(d, belt, 0.0028, k_edge=0.0015)
    # shoulder extraction straps (epaulettes)
    t = np.clip((M[:, 0] - 0.080) / (0.185 - 0.080), 0, 1)
    yc = 0.016 - 0.008 * t
    strap = np.maximum(np.abs(M[:, 1] - yc) - 0.020, np.abs(M[:, 0] - 0.130) - 0.052)
    strap = np.maximum(strap, 1.365 - M[:, 2])
    d = raised(d, strap, 0.0034, k_edge=0.0018)
    # centre-front zip piping
    zip_r = np.maximum(np.abs(x) - 0.0045, np.maximum(z - 1.445, 1.087 - z))
    zip_r = np.maximum(zip_r, y + 0.02)
    d = raised(d, zip_r, 0.0012, k_edge=0.0008)
    return d


def collar_rest(P):
    """Stand-up mandarin collar with a small front opening and tab."""
    c0 = v3(0.0, 0.020, 1.430)
    axis_R = rot_x(4.0)                   # follows the forward neck angle
    q = (P - c0) @ axis_R
    r = np.sqrt(q[:, 0] ** 2 + (q[:, 1] * 0.93) ** 2)
    h = 0.050 + 0.010 * smoothstep(-0.04, 0.05, q[:, 1])   # taller at the back
    R0 = 0.0745
    ring = np.abs(r - R0) - 0.0040
    band = np.maximum(ring, -q[:, 2] - 0.012)
    band = smax(band, q[:, 2] - h, 0.005)            # rounded top edge
    # front opening (small V)
    gap = np.abs(q[:, 0]) - (0.002 + 0.008 * smoothstep(0.0, 0.05, q[:, 2]))
    band = smax(band, -np.maximum(gap, q[:, 1] + 0.03), 0.0015)
    # closure tab across the opening
    tab = sd_box(q, v3(0.0, -R0 - 0.0045, 0.028), v3(0.013, 0.0016, 0.0075), 0.0014)
    return smin(band, tab, 0.0015)


# --------------------------------------------------------------------------
# limbs (rest space)
# --------------------------------------------------------------------------

def _limb_side(P, side):
    Q = P.copy()
    Q[:, 0] *= side
    return Q


def upperarm_rest(P, side):
    Q = _limb_side(P, side)
    S = J("upperarm_L") + v3(0.002, 0, -0.004)
    E = J("forearm_L")
    ax = norm(E - S)
    d = sd_round_cone(Q, S - ax * 0.010, E, 0.045, 0.036)
    R = frame_from(ax, v3(0, -1, 0))
    d = smin(d, sd_ellipsoid(Q, S + ax * 0.035 + v3(0.008, 0, 0.0), v3(0.050, 0.052, 0.068), R), 0.03)   # deltoid
    d = smin(d, sd_ellipsoid(Q, S + ax * 0.15 + v3(0.0, -0.014, 0), v3(0.036, 0.034, 0.075), R), 0.025)    # biceps
    d = smin(d, sd_ellipsoid(Q, S + ax * 0.13 + v3(0.004, 0.018, 0), v3(0.036, 0.034, 0.085), R), 0.025)   # triceps
    return d


def forearm_rest(P, side):
    Q = _limb_side(P, side)
    E = J("forearm_L")
    W = J("hand_L")
    ax = norm(W - E)
    d = sd_round_cone(Q, E, W + ax * 0.012, 0.037, 0.026)
    R = frame_from(ax, v3(0, -1, 0))
    d = smin(d, sd_ellipsoid(Q, E + ax * 0.075 + v3(0.004, -0.004, 0), v3(0.043, 0.040, 0.085), R), 0.03)
    # knit cuff at the wrist
    t = (Q - W) @ ax
    cuff = np.maximum(np.abs(t + 0.010) - 0.016, sd_round_cone(Q, E, W + ax * 0.012, 0.037, 0.026) - 0.0)
    rib = 0.0006 * np.sin(np.arctan2((Q - W) @ R[:, 1], (Q - W) @ R[:, 0]) * 48.0)
    d = smin(d, np.where(np.abs(t + 0.010) < 0.018, cuff - 0.0020 + rib, 1.0), 0.002)
    # inner-elbow fabric fold
    return d


def thigh_rest(P, side):
    Q = _limb_side(P, side)
    H = J("thigh_L") + v3(0.002, 0, 0.010)
    K = J("shin_L")
    ax = norm(K - H)
    d = sd_round_cone(Q, H, K, 0.078, 0.051)
    R = frame_from(ax, v3(0, -1, 0))
    d = smin(d, sd_ellipsoid(Q, H + ax * 0.19 + v3(0.004, -0.022, 0), v3(0.055, 0.046, 0.145), R), 0.04)   # quads
    d = smin(d, sd_ellipsoid(Q, H + ax * 0.21 + v3(0.0, 0.026, 0), v3(0.052, 0.044, 0.135), R), 0.04)     # hamstrings
    return d


def shin_rest(P, side):
    Q = _limb_side(P, side)
    K = J("shin_L")
    A = J("foot_L")
    ax = norm(A - K)
    d = sd_round_cone(Q, K + v3(0, 0, 0.01), A, 0.052, 0.036)
    R = frame_from(ax, v3(0, -1, 0))
    d = smin(d, sd_ellipsoid(Q, K + ax * 0.12 + v3(0.0, 0.026, 0), v3(0.046, 0.044, 0.110), R), 0.035)  # calf
    d = smin(d, sd_ellipsoid(Q, K + v3(0.0, -0.030, -0.005), v3(0.038, 0.022, 0.040)), 0.02)            # knee cap
    # knee stretch panel ridges
    kz = Q[:, 2] - 0.475
    knee = smoothstep(0.05, 0.03, np.abs(kz)) * smoothstep(-0.02, -0.04, Q[:, 1] - K[1])
    d = d - 0.0009 * knee * np.sin(kz * 2 * np.pi / 0.011)
    # ankle cuff
    t = Q[:, 2] - 0.140
    d = smin(d, np.where(np.abs(t) < 0.022, sd_round_cone(Q, K, A, 0.052, 0.036) - 0.0022
                         + 0.0006 * np.sin(np.arctan2(Q[:, 1] - A[1], Q[:, 0] - A[0]) * 44.0), 1.0), 0.002)
    return d


# --------------------------------------------------------------------------
# suit assembly (posed)
# --------------------------------------------------------------------------

def _torso_points(P, pose):
    """Pull points back to torso rest space with a continuous spine blend."""
    z = P[:, 2] - pose.root_offset[2]
    w_sp = smoothstep(0.98, 1.12, z)
    w_ch = smoothstep(1.16, 1.30, z)
    Ps = pose.to_rest("pelvis", P)
    Pp = pose.to_rest("spine", P)
    Pc = pose.to_rest("chest", P)
    out = Ps * (1 - w_sp)[:, None] + Pp * w_sp[:, None]
    out = out * (1 - w_ch)[:, None] + Pc * w_ch[:, None]
    return out


def suit_parts(P, pose):
    """Return list of (name, distance, rest_points)."""
    parts = []
    Pt = _torso_points(P, pose)
    dt = torso_rest(Pt)
    dt = suit_details_torso(dt, Pt)
    Pn = pose.to_rest("chest", P)
    dc = collar_rest(Pn)
    parts.append(("torso", dt, Pt))
    parts.append(("collar", dc, Pn))
    for s, tag in ((1, "L"), (-1, "R")):
        Pu = pose.to_rest(f"upperarm_{tag}", P)
        Pf = pose.to_rest(f"forearm_{tag}", P)
        Pth = pose.to_rest(f"thigh_{tag}", P)
        Psh = pose.to_rest(f"shin_{tag}", P)
        parts.append((f"upperarm_{tag}", upperarm_rest(Pu, s), Pu))
        parts.append((f"forearm_{tag}", forearm_rest(Pf, s), Pf))
        parts.append((f"thigh_{tag}", thigh_rest(Pth, s), Pth))
        parts.append((f"shin_{tag}", shin_rest(Psh, s), Psh))
    return parts


def _combine_suit(parts):
    D = {n: d for n, d, _ in parts}
    arms = []
    legs = []
    for tag in ("L", "R"):
        arm = smin(D[f"upperarm_{tag}"], D[f"forearm_{tag}"], 0.022)
        arms.append(arm)
        leg = smin(D[f"thigh_{tag}"], D[f"shin_{tag}"], 0.03)
        legs.append(leg)
    d = D["torso"]
    for leg in legs:
        d = smin(d, leg, 0.035)
    for arm in arms:
        d = smin(d, arm, 0.020)
    d = np.minimum(d, D["collar"])
    return d


def suit_sdf(P, pose=REST):
    return _combine_suit(suit_parts(P, pose))


def suit_rest_coords(P, pose=REST, tau=0.006):
    parts = suit_parts(P, pose)
    W = softmin_weights([d for _, d, _ in parts], tau)
    R = sum(W[i][:, None] * parts[i][2] for i in range(len(parts)))
    return R


# --------------------------------------------------------------------------
# gloves
# --------------------------------------------------------------------------

FINGERS = {
    # name: (knuckle u, v, lengths, radius)
    "index": (0.094, 0.0275, (0.043, 0.026, 0.022), 0.0098),
    "middle": (0.097, 0.0080, (0.047, 0.029, 0.023), 0.0102),
    "ring": (0.093, -0.0115, (0.044, 0.027, 0.022), 0.0097),
    "pinky": (0.085, -0.0295, (0.035, 0.021, 0.019), 0.0086),
}

REST_CURL = {
    "index": (16, 30, 16, 0), "middle": (20, 36, 20, 0), "ring": (24, 40, 22, 0),
    "pinky": (28, 44, 24, 0), "thumb": (8, 14, 12),
}
GRIP_CURL = {
    "index": (52, 78, 42, 0), "middle": (58, 84, 44, 0), "ring": (62, 86, 44, 0),
    "pinky": (66, 88, 46, 0), "thumb": (30, 26, 22),
}
# open, cupped hand resting on a curved surface (helmet held in both hands)
HOLD_CURL = {
    "index": (10, 14, 8, 0), "middle": (12, 16, 9, 0), "ring": (14, 18, 10, 0),
    "pinky": (16, 20, 10, 0), "thumb": (4, 6, 4),
}
SPREAD = {"index": -5.0, "middle": 0.0, "ring": 4.0, "pinky": 9.0}


def hand_frame(pose, side):
    """Hand rest frame: columns (u fingers, v thumb side, w back of hand)."""
    tag = "L" if side > 0 else "R"
    E = JOINTS[f"forearm_{tag}"][1]
    Wr = JOINTS[f"hand_{tag}"][1]
    u = norm(Wr - E)
    w = v3(side * 1.0, 0.0, 0.0)
    w = norm(w - u * (w @ u))
    v = np.cross(u, w) * side
    return np.stack([u, v, w], axis=1)


def hand_segments(curl, side, scale=1.04):
    """Finger/thumb segments in canonical hand space (u, v, w)."""
    segs = []
    for name, (ku, kv, lens, r) in FINGERS.items():
        mcp, pip, dip, splay = curl[name]
        base = v3(ku, kv, 0.002) * scale
        direction = rot_z(SPREAD[name]) @ v3(1.0, 0.0, 0.0)
        R = rot_y(0.0)
        p = base
        bend = 0.0
        radii = [r, r * 0.93, r * 0.86, r * 0.78]
        for i, (L, ang) in enumerate(zip(lens, (mcp, pip, dip))):
            bend += ang
            # flexion bends towards the palm (-w); rotate about v
            dvec = rot_axis(v3(0, 1, 0), bend) @ direction
            q = p + dvec * L * scale
            segs.append((f"{name}{i}", p, q, radii[i] * scale, radii[i + 1] * scale))
            p = q
    # thumb
    cmc = v3(0.018, 0.022, -0.012) * scale
    t1, t2, t3 = curl["thumb"]
    tdir = norm(v3(0.80, 0.42, -0.42))
    lens = (0.042, 0.033, 0.028)
    radii = (0.0128, 0.0112, 0.0102, 0.0090)
    p = cmc
    axis = norm(np.cross(tdir, v3(0, 0, 1)))
    bend = 0.0
    for i, (L, ang) in enumerate(zip(lens, (t1, t2, t3))):
        bend += ang
        dvec = rot_axis(axis, -bend) @ tdir
        q = p + dvec * L * scale
        segs.append((f"thumb{i}", p, q, radii[i] * scale, radii[i + 1] * scale))
        p = q
    return segs


def glove_local_sdf(Q, segs, scale=1.04):
    """Glove in canonical hand space."""
    u, v, w = Q[:, 0], Q[:, 1], Q[:, 2]
    palm = sd_box(Q, v3(0.050, 0.000, -0.003) * scale, v3(0.046, 0.041, 0.0130) * scale, 0.0115 * scale)
    palm = smin(palm, sd_ellipsoid(Q, v3(0.030, 0.022, -0.012) * scale, v3(0.030, 0.018, 0.014) * scale), 0.012)  # thenar
    palm = smin(palm, sd_ellipsoid(Q, v3(0.040, -0.026, -0.008) * scale, v3(0.040, 0.014, 0.012) * scale), 0.010)  # hypothenar
    wrist = sd_round_cone(Q, v3(-0.050, 0.0, -0.002), v3(0.012, 0.0, -0.002) * scale, 0.0285, 0.0285)
    wrist = smax(wrist, np.abs(w + 0.002) - 0.024, 0.008)
    d = smin(palm, wrist, 0.02)
    fingers = np.full(len(Q), 1.0)
    for name, a, b, ra, rb in segs:
        fingers = smin(fingers, sd_round_cone(Q, a, b, ra, rb), 0.004)
    d = smin(d, fingers, 0.006)
    # knuckle guard: raised padded plate on the back of the hand
    guard = sd_ellipsoid(Q, v3(0.086, 0.000, 0.0115) * scale, v3(0.016, 0.036, 0.0042) * scale)
    d = smin(d, guard, 0.003)
    # gauntlet cuff with strap
    cuff = sd_round_cone(Q, v3(-0.052, 0.0, -0.002), v3(-0.004, 0.0, -0.002), 0.0335, 0.0305)
    cuff = smax(cuff, np.abs(w + 0.002) - 0.028, 0.008)
    d = smin(d, cuff, 0.006)
    strap = np.maximum(np.abs(u + 0.022) - 0.011, cuff - 0.0024)
    d = np.minimum(d, strap)
    return d


def glove_parts(P, pose, side, curl):
    tag = "L" if side > 0 else "R"
    F = hand_frame(pose, side)
    Wr = JOINTS[f"hand_{tag}"][1]
    Pr = pose.to_rest(f"hand_{tag}", P)
    Q = (Pr - Wr) @ F       # canonical (u, v, w)
    segs = hand_segments(curl, side)
    return glove_local_sdf(Q, segs), Q


def glove_sdf(P, pose, side, curl):
    return glove_parts(P, pose, side, curl)[0]


def glove_rest_coords(P, pose, side, curl):
    return glove_parts(P, pose, side, curl)[1]


# --------------------------------------------------------------------------
# boots
# --------------------------------------------------------------------------

def boot_local(Q):
    """Boot in foot space: x lateral, y forward, z up (ankle at origin)."""
    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    zf = z + 0.090      # height above the floor
    heel = sd_ellipsoid(Q, v3(0.0, -0.035, -0.052), v3(0.036, 0.045, 0.040))
    mid = sd_ellipsoid(Q, v3(0.0, 0.050, -0.058), v3(0.046, 0.085, 0.034))
    ball = sd_ellipsoid(Q, v3(0.002, 0.125, -0.066), v3(0.050, 0.055, 0.024))
    toe = sd_ellipsoid(Q, v3(0.002, 0.165, -0.070), v3(0.042, 0.035, 0.021))
    d = smin(smin(smin(heel, mid, 0.03), ball, 0.03), toe, 0.02)
    shaft = sd_round_cone(Q, v3(0.0, -0.004, -0.050), v3(0.0, 0.004, 0.065), 0.044, 0.041)
    d = smin(d, shaft, 0.03)
    d = smax(d, -zf + 0.004, 0.004)    # flat underside
    # thin racing sole
    sole = sd_box(Q, v3(0.0, 0.060, -0.0860), v3(0.050, 0.130, 0.0045), 0.004)
    sole = smax(sole, smin(sd_ellipsoid(Q, v3(0.0, 0.065, -0.07), v3(0.056, 0.145, 0.06)), heel - 0.004, 0.03), 0.004)
    d = smin(d, sole, 0.003)
    # padded collar at the top
    collar = sd_torus(Q, v3(0.0, 0.004, 0.065), 0.041, 0.0055)
    d = smin(d, collar, 0.004)
    # ankle strap
    strap = np.maximum(np.abs(z + 0.012) - 0.010, shaft - 0.0028)
    d = np.minimum(d, strap)
    return d


def boot_frame(side):
    tag = "L" if side > 0 else "R"
    R = rot_z(side * 8.0)      # slight toe-out
    # foot space: x lateral (outward), y forward (-Y world), z up
    B = np.stack([v3(side, 0, 0), v3(0, -1, 0), v3(0, 0, 1)], axis=1)
    return R @ B


def boot_sdf(P, pose, side):
    tag = "L" if side > 0 else "R"
    A = JOINTS[f"foot_{tag}"][1]
    Pr = pose.to_rest(f"foot_{tag}", P)
    Q = (Pr - A) @ boot_frame(side)
    return boot_local(Q)


def boot_rest_coords(P, pose, side):
    tag = "L" if side > 0 else "R"
    A = JOINTS[f"foot_{tag}"][1]
    Pr = pose.to_rest(f"foot_{tag}", P)
    return (Pr - A) @ boot_frame(side)


# --------------------------------------------------------------------------
# arm IK: place the wrist and orient the hand, elbow towards a pole
# --------------------------------------------------------------------------

def _frame(a, h):
    a = norm(a)
    h = h - a * (h @ a)
    h = norm(h)
    return np.stack([a, h, np.cross(a, h)], axis=1)


def _twist_angle(R, axis):
    """Angle of the twist component of rotation R about axis (degrees)."""
    w = np.sqrt(max(1.0 + R[0, 0] + R[1, 1] + R[2, 2], 1e-12)) / 2.0
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (4.0 * w)
    return np.degrees(2.0 * np.arctan2(v @ axis, w))


def solve_arm(rot, side, wrist, hand_axes, pole, twist_share=0.75):
    """Return a copy of `rot` with upperarm/forearm/hand set so the wrist sits
    at `wrist` and the hand frame matches `hand_axes` (columns u, v, w in
    world). `pole` is the direction the elbow points. Most of the axial twist
    goes to the forearm so the glove cuff stays aligned with the sleeve."""
    tag = "L" if side > 0 else "R"
    rot = dict(rot)
    for k in (f"upperarm_{tag}", f"forearm_{tag}", f"hand_{tag}"):
        rot.pop(k, None)
    base = Pose(rot)
    S = base.J[f"upperarm_{tag}"]
    s0, e0, w0 = (JOINTS[f"{b}_{tag}"][1] for b in ("upperarm", "forearm", "hand"))
    l1, l2 = np.linalg.norm(e0 - s0), np.linalg.norm(w0 - e0)
    d_vec = wrist - S
    d = min(np.linalg.norm(d_vec), (l1 + l2) * 0.999)
    n = norm(d_vec)
    ca = np.clip((l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d), -1.0, 1.0)
    p = norm(pole - n * (pole @ n))
    E = S + l1 * (ca * n + np.sqrt(1 - ca ** 2) * p)
    Wt = S + n * d
    hinge0 = v3(-1.0, 0.0, 0.0)                      # rest flexion axis
    a_post = norm(E - S)
    f_post = norm(Wt - E)
    h_post = norm(np.cross(a_post, f_post))
    Rua_w = _frame(a_post, h_post) @ _frame(e0 - s0, hinge0).T
    Rfa_w = _frame(f_post, h_post) @ _frame(w0 - e0, hinge0).T
    Rp = base.R[f"clav_{tag}"]
    Rua_l = Rp.T @ Rua_w
    Fr = hand_frame(REST, side)
    Rt = hand_axes @ Fr.T                            # target hand rotation (world)
    f0 = norm(w0 - e0)
    Rrel = Rfa_w.T @ Rt
    tw = _twist_angle(Rrel, f0) * twist_share
    Rfa_l = Rua_w.T @ Rfa_w @ rot_axis(f0, tw)
    Rfa_w2 = Rua_w @ Rfa_l
    Rh_l = Rfa_w2.T @ Rt
    rot[f"upperarm_{tag}"] = Rua_l
    rot[f"forearm_{tag}"] = Rfa_l
    rot[f"hand_{tag}"] = Rh_l
    return rot
