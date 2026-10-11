"""Materials for the racer.

Blockout stage: Principled BSDF materials with the final colour palette, so
the design approval render shows the real colours. The painted, Arcane-style
shading (Shader to RGB ramps, painted highlights) replaces them in the
look-development stage.

    import looks
    looks.apply()
"""

import bpy

# Palette (sRGB hex), original to this character
PALETTE = {
    "skin": "C99272",        # warm olive-tan
    "lips": "B5705C",
    "hair": "2A1C17",        # near-black brown
    "hair_tip": "6B3A26",    # copper-brown ends
    "hair_buzz": "4E3B34",   # buzzed undercut: short hair with the skin showing through
    "sclera": "E9E2D8",
    "iris": "5E7F6A",        # grey-green
    "iris_dark": "23352C",
    "pupil": "0B0B0C",
    "suit": "5E1A1D",        # deep oxblood
    "bone": "E3DAC6",        # bone white
    "shirt": "2B2B2F",       # charcoal undershirt
    "glove": "1C1C20",
    "boot": "24242A",
}


def srgb(hex_):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return tuple(lin(int(hex_[i:i + 2], 16)) for i in (0, 2, 4)) + (1.0,)


def _principled(name, color, roughness=0.5, sss=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = srgb(PALETTE[color]) if isinstance(color, str) else color
    b.inputs["Roughness"].default_value = roughness
    if sss:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = (0.9, 0.35, 0.2)
        b.inputs["Subsurface Scale"].default_value = 0.004
    return m


def skin():
    return _principled("MAT_Skin", "skin", roughness=0.55, sss=0.15)


def hair():
    return _principled("MAT_Hair", "hair", roughness=0.45)


def buzz():
    return _principled("MAT_HairBuzz", "hair_buzz", roughness=0.85)


def eye(radius=0.0125, iris=0.0066, pupil=0.0026):
    """Sclera, iris and pupil from the eyeball's own coordinates: the
    eyeball's local Z points out through the pupil."""
    m = bpy.data.materials.get("MAT_Eye") or bpy.data.materials.new("MAT_Eye")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type not in ("OUTPUT_MATERIAL", "BSDF_PRINCIPLED"):
            nt.nodes.remove(n)
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.08
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    div = nt.nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = radius
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], div.inputs[0])
    nt.links.new(div.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    edge_iris = (1 - (iris / radius) ** 2) ** 0.5
    edge_pupil = (1 - (pupil / radius) ** 2) ** 0.5
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, srgb(PALETTE["sclera"])
    els[1].position, els[1].color = edge_iris - 0.004, srgb(PALETTE["sclera"])
    for pos, key in ((edge_iris, "iris_dark"), (edge_iris + 0.025, "iris"), (edge_pupil - 0.006, "iris"),
                     (edge_pupil, "pupil"), (1.0, "pupil")):
        e = els.new(pos)
        e.color = srgb(PALETTE[key])
    return m


def cloth(name, color, roughness=0.75):
    return _principled(name, color, roughness=roughness)


def assign(obj_name, mat):
    ob = bpy.data.objects.get(obj_name)
    if ob is None:
        return
    ob.data.materials.clear()
    ob.data.materials.append(mat)


def apply():
    assign("CHR_Head", skin())
    assign("CHR_Body", skin())
    assign("CHR_Eyes", eye())
    assign("CHR_Brows", hair())
    assign("CHR_HairCap", buzz())
    assign("CHR_HairTop", hair())
    assign("OUT_Undershirt", cloth("MAT_Shirt", "shirt", 0.85))
    assign("OUT_SleeveRoll", cloth("MAT_Shirt", "shirt", 0.85))
    assign("OUT_Gloves", cloth("MAT_Glove", "glove", 0.6))
    assign("OUT_Boots", cloth("MAT_Boot", "boot", 0.5))
    for n in ("OUT_SuitTopRoll", "OUT_SuitSleeves", "OUT_SuitKnot"):
        assign(n, cloth("MAT_SuitOxblood", "suit"))
    for n in ("OUT_SuitCuff1", "OUT_SuitCuff2"):
        assign(n, cloth("MAT_SuitBone", "bone"))
