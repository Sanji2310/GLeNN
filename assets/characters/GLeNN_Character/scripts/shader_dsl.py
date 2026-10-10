"""A tiny expression layer over Blender shader nodes.

Lets panel layouts, livery graphics and painterly masks be written as plain
arithmetic (a + b * c, smoothstep, min, ...) that compiles to node graphs,
so the suit seams and helmet graphics stay crisp at any render distance.
"""

import bpy


class NB:
    def __init__(self, mat):
        self.mat = mat
        try:
            mat.use_nodes = True
        except AttributeError:
            pass
        self.nt = mat.node_tree
        self.nodes = self.nt.nodes
        self.links = self.nt.links
        self.col = 0
        self.row = 0

    def clear(self):
        self.nodes.clear()

    def node(self, kind, **props):
        n = self.nodes.new(kind)
        n.location = (self.col * 180, -self.row * 60)
        self.row += 1
        if self.row > 40:
            self.row = 0
            self.col += 1
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, src, dst):
        self.links.new(_sock(src), dst)

    def feed(self, sock, val):
        if isinstance(val, (int, float)):
            sock.default_value = val
        elif isinstance(val, (tuple, list)):
            v = list(val)
            if len(sock.default_value) == 4 and len(v) == 3:
                v = v + [1.0]
            sock.default_value = v
        else:
            self.links.new(_sock(val), sock)

    # --- scalar math -------------------------------------------------------
    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.feed(n.inputs[i], v)
        return F(self, n.outputs[0])

    def vmath(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.feed(n.inputs[0], a)
        if b is not None:
            self.feed(n.inputs[1], b)
        if scale is not None:
            self.feed(n.inputs["Scale"], scale)
        out = n.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs["Vector"]
        return F(self, out) if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else V(self, out)

    def const(self, x):
        n = self.node("ShaderNodeValue")
        n.outputs[0].default_value = x
        return F(self, n.outputs[0])

    def vec(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for s, val in zip(n.inputs, (x, y, z)):
            self.feed(s, val)
        return V(self, n.outputs[0])

    def smoothstep(self, e0, e1, x):
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        self.feed(n.inputs["Value"], x)
        self.feed(n.inputs["From Min"], e0)
        self.feed(n.inputs["From Max"], e1)
        n.inputs["To Min"].default_value = 0.0
        n.inputs["To Max"].default_value = 1.0
        return F(self, n.outputs[0])

    def linstep(self, e0, e1, x):
        n = self.node("ShaderNodeMapRange", interpolation_type="LINEAR", clamp=True)
        self.feed(n.inputs["Value"], x)
        self.feed(n.inputs["From Min"], e0)
        self.feed(n.inputs["From Max"], e1)
        return F(self, n.outputs[0])

    def aa_step(self, edge, x, width=0.0004):
        """Crisp but anti-aliased step (x > edge)."""
        return self.smoothstep(edge - width, edge + width, x)

    def mn(self, *xs):
        out = xs[0]
        for x in xs[1:]:
            out = self.math("MINIMUM", out, x)
        return out

    def mx(self, *xs):
        out = xs[0]
        for x in xs[1:]:
            out = self.math("MAXIMUM", out, x)
        return out

    def abs(self, x):
        return self.math("ABSOLUTE", x)

    def sqrt(self, x):
        return self.math("SQRT", x)

    def sin(self, x):
        return self.math("SINE", x)

    def fract(self, x):
        return self.math("FRACT", x)

    def atan2(self, y, x):
        return self.math("ARCTAN2", y, x)

    def clamp01(self, x):
        return self.math("ADD", x, 0.0, clamp=True)

    # --- vectors / inputs -------------------------------------------------------
    def texcoord(self, which="Object"):
        n = self.node("ShaderNodeTexCoord")
        return V(self, n.outputs[which])

    def attribute(self, name):
        n = self.node("ShaderNodeAttribute", attribute_name=name)
        return V(self, n.outputs["Vector"])

    def xyz(self, vec):
        n = self.node("ShaderNodeSeparateXYZ")
        self.feed(n.inputs[0], vec)
        return F(self, n.outputs[0]), F(self, n.outputs[1]), F(self, n.outputs[2])

    def noise(self, vec, scale, detail=2.0, rough=0.5, dims="3D", out="Fac", distortion=0.0):
        n = self.node("ShaderNodeTexNoise")
        try:
            n.noise_dimensions = dims
        except (AttributeError, TypeError):
            pass
        self.feed(n.inputs["Vector"], vec)
        self.feed(n.inputs["Scale"], scale)
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        if "Distortion" in n.inputs:
            n.inputs["Distortion"].default_value = distortion
        return F(self, n.outputs[out]) if out == "Fac" else V(self, n.outputs[out])

    def voronoi(self, vec, scale, feature="F1", out="Distance", randomness=1.0):
        n = self.node("ShaderNodeTexVoronoi", feature=feature)
        self.feed(n.inputs["Vector"], vec)
        self.feed(n.inputs["Scale"], scale)
        if "Randomness" in n.inputs:
            n.inputs["Randomness"].default_value = randomness
        return F(self, n.outputs[out])

    def wave(self, vec, scale, kind="BANDS", direction="X", distortion=0.0, profile="SIN"):
        n = self.node("ShaderNodeTexWave", wave_type=kind)
        try:
            if kind == "BANDS":
                n.bands_direction = direction
            else:
                n.rings_direction = direction
            n.wave_profile = profile
        except (AttributeError, TypeError):
            pass
        self.feed(n.inputs["Vector"], vec)
        self.feed(n.inputs["Scale"], scale)
        n.inputs["Distortion"].default_value = distortion
        return F(self, n.outputs["Fac"])

    # --- colour ---------------------------------------------------------------------
    def mix(self, fac, a, b, blend="MIX"):
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.feed(n.inputs[0], fac)
        self.feed(n.inputs[6], a)
        self.feed(n.inputs[7], b)
        return V(self, n.outputs[2])

    def mixf(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="FLOAT")
        self.feed(n.inputs[0], fac)
        self.feed(n.inputs[2], a)
        self.feed(n.inputs[3], b)
        return F(self, n.outputs[0])

    def layer(self, base, layers):
        """layers: list of (mask, colour)."""
        out = base
        for m, c in layers:
            out = self.mix(m, out, c)
        return out

    def bump(self, height, strength=1.0, distance=0.001, normal=None):
        n = self.node("ShaderNodeBump")
        self.feed(n.inputs["Height"], height)
        n.inputs["Strength"].default_value = strength
        n.inputs["Distance"].default_value = distance
        if normal is not None:
            self.feed(n.inputs["Normal"], normal)
        return V(self, n.outputs["Normal"])

    def principled(self, **inputs):
        n = self.node("ShaderNodeBsdfPrincipled")
        alias = {
            "base": ["Base Color"], "rough": ["Roughness"], "metal": ["Metallic"],
            "sss": ["Subsurface Weight", "Subsurface"], "sss_radius": ["Subsurface Radius"],
            "sss_scale": ["Subsurface Scale"], "spec": ["Specular IOR Level", "Specular"],
            "coat": ["Coat Weight", "Clearcoat"], "coat_rough": ["Coat Roughness", "Clearcoat Roughness"],
            "coat_normal": ["Coat Normal", "Clearcoat Normal"],
            "sheen": ["Sheen Weight", "Sheen"], "sheen_rough": ["Sheen Roughness"], "sheen_tint": ["Sheen Tint"],
            "trans": ["Transmission Weight", "Transmission"], "ior": ["IOR"], "alpha": ["Alpha"],
            "normal": ["Normal"], "aniso": ["Anisotropic"], "aniso_rot": ["Anisotropic Rotation"],
            "tangent": ["Tangent"], "emission": ["Emission Color", "Emission"],
            "emission_strength": ["Emission Strength"], "thin_film": ["Thin Film Thickness"],
            "thin_film_ior": ["Thin Film IOR"], "coat_tint": ["Coat Tint"],
        }
        for k, val in inputs.items():
            for name in alias.get(k, [k]):
                if name in n.inputs:
                    self.feed(n.inputs[name], val)
                    break
        return n

    def output(self, shader_node, displacement=None):
        out = self.node("ShaderNodeOutputMaterial")
        self.links.new(shader_node.outputs[0], out.inputs["Surface"])
        if displacement is not None:
            self.links.new(_sock(displacement), out.inputs["Displacement"])
        return out


def _sock(x):
    return x.s if isinstance(x, (F, V)) else x


class F:
    """Float socket expression."""

    def __init__(self, nb, s):
        self.nb = nb
        self.s = s

    def _op(self, op, other, rev=False):
        return self.nb.math(op, other, self) if rev else self.nb.math(op, self, other)

    def __add__(self, o): return self._op("ADD", o)
    def __radd__(self, o): return self._op("ADD", o, True)
    def __sub__(self, o): return self._op("SUBTRACT", o)
    def __rsub__(self, o): return self._op("SUBTRACT", o, True)
    def __mul__(self, o): return self._op("MULTIPLY", o)
    def __rmul__(self, o): return self._op("MULTIPLY", o, True)
    def __truediv__(self, o): return self._op("DIVIDE", o)
    def __rtruediv__(self, o): return self._op("DIVIDE", o, True)
    def __neg__(self): return self.nb.math("MULTIPLY", self, -1.0)
    def __pow__(self, o): return self._op("POWER", o)


class V:
    """Vector / colour socket expression."""

    def __init__(self, nb, s):
        self.nb = nb
        self.s = s

    def __add__(self, o): return self.nb.vmath("ADD", self, o)
    def __sub__(self, o): return self.nb.vmath("SUBTRACT", self, o)

    def __mul__(self, o):
        if isinstance(o, (int, float)):
            return self.nb.vmath("SCALE", self, scale=o)
        if isinstance(o, F):
            return self.nb.vmath("SCALE", self, scale=o)
        return self.nb.vmath("MULTIPLY", self, o)

    def dot(self, o): return self.nb.vmath("DOT_PRODUCT", self, o)
    def length(self): return self.nb.vmath("LENGTH", self)
