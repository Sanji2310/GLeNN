"""Shared helpers for the GLeNN character scripts.

Target: Blender 5.2.2 LTS. Every builder in this folder is safe to re-run:
it clears only the objects it owns (tagged with the custom property
``gln_owner``) before rebuilding them, so nothing gets duplicated.

The project root is the folder that contains ``scripts/``. All other
folders (project, renders, exports, ...) are resolved from it.
"""

import math
import os

import bpy
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = {name: os.path.join(ROOT, name) for name in (
    "project", "scripts", "renders", "exports", "textures", "reports", "reference", "backup")}
MASTER_BLEND = os.path.join(DIRS["project"], "GLeNN_Character_Master.blend")

# Collection layout: (name, parent). The helmet is a separate top-level
# collection so it can be hidden or exported without touching the head.
COLLECTIONS = [
    ("GLN_Character", None),
    ("GLN_Body", "GLN_Character"),
    ("GLN_Head", "GLN_Character"),
    ("GLN_Hair", "GLN_Character"),
    ("GLN_Outfit", "GLN_Character"),
    ("GLN_Helmet", None),
    ("GLN_Rig", None),
    ("GLN_Studio", None),
    ("GLN_Cameras", None),
]

# Design palette (linear sRGB). Deep indigo, warm cream, amber accent.
PALETTE = {
    "indigo": (0.030, 0.026, 0.085),
    "indigo_light": (0.090, 0.075, 0.230),
    "cream": (0.780, 0.700, 0.560),
    "amber": (0.900, 0.420, 0.060),
    "charcoal": (0.025, 0.025, 0.030),
    "skin": (0.330, 0.185, 0.118),
    "hair": (0.060, 0.032, 0.018),
    "clay": (0.500, 0.480, 0.460),
}

CHARACTER_HEIGHT = 1.78


def ensure_dirs():
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)


def collection(name):
    """Get or create a collection from COLLECTIONS, linked under its parent."""
    parents = dict(COLLECTIONS)
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    parent_name = parents.get(name)
    parent = collection(parent_name) if parent_name else bpy.context.scene.collection
    if coll.name not in parent.children:
        parent.children.link(coll)
    return coll


def ensure_collections():
    for name, _ in COLLECTIONS:
        collection(name)


def clear_owned(owner):
    """Delete objects (and their orphaned data) created by a given builder."""
    doomed = [o for o in bpy.data.objects if o.get("gln_owner") == owner]
    datas = [o.data for o in doomed if o.data is not None]
    for obj in doomed:
        bpy.data.objects.remove(obj, do_unlink=True)
    for data in datas:
        try:
            if data.users == 0:
                _remove_datablock(data)
        except ReferenceError:
            pass


def _remove_datablock(data):
    for store in (bpy.data.meshes, bpy.data.curves, bpy.data.lights,
                  bpy.data.cameras, bpy.data.armatures):
        if store.get(data.name) == data:
            store.remove(data)
            return


def link(obj, coll_name, owner):
    """Link a new object into a collection and tag it with its builder."""
    obj["gln_owner"] = owner
    collection(coll_name).objects.link(obj)
    return obj


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def principled(name, color, roughness=0.5, metallic=0.0, **inputs):
    """Get or create a simple Principled BSDF material. Extra inputs are
    passed by socket name, e.g. principled(..., **{"Coat Weight": 1.0})."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        nodes.clear()
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (300, 0)
        mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    for key, value in inputs.items():
        bsdf.inputs[key].default_value = value
    mat.diffuse_color = (*color, 1.0)
    return mat


def set_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def deg(x):
    return math.radians(x)
