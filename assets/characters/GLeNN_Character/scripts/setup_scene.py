"""Scene, studio, cameras and render settings for the GLeNN character.

Run inside Blender 5.2.2:
    import setup_scene; setup_scene.run()

Safe to re-run. It only rebuilds objects tagged gln_owner="studio" or
gln_owner="cameras".
"""

import bpy
from mathutils import Vector

import glenn_common as G

H = G.CHARACTER_HEIGHT
HELMET_STAND = Vector((0.95, 0.35, 0.0))   # where the helmet is displayed
HELMET_HEIGHT = 1.05                        # stand top

# name: (location, target, lens or None for ortho, ortho_scale, resolution)
CAMERAS = {
    "CAM_Front":        ((0.0, -8.0, H * 0.52), (0.0, 0.0, H * 0.52), None, 2.15, (1200, 1600)),
    "CAM_Side":         ((8.0, 0.0, H * 0.52), (0.0, 0.0, H * 0.52), None, 2.15, (1200, 1600)),
    "CAM_Rear":         ((0.0, 8.0, H * 0.52), (0.0, 0.0, H * 0.52), None, 2.15, (1200, 1600)),
    "CAM_ThreeQuarter": ((2.55, -4.15, 1.30), (0.0, 0.0, 0.93), 50, None, (1200, 1600)),
    "CAM_Hero":         ((1.25, -3.05, 1.52), (0.05, 0.0, 1.30), 85, None, (1600, 2000)),
    "CAM_Face":         ((0.42, -1.30, 1.66), (0.0, 0.0, 1.62), 105, None, (1600, 1600)),
    "CAM_Helmet":       ((1.55, -0.55, 1.28), (HELMET_STAND.x, HELMET_STAND.y, HELMET_HEIGHT + 0.14), 100, None, (1600, 1200)),
}


def render_settings(scene):
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.frame_start = scene.frame_end = scene.frame_current = 1
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 64
    scene.eevee.use_raytracing = True
    scene.eevee.use_shadows = True
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    # Cycles (final renders): GPU device is chosen in Preferences.
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True


def world(scene):
    w = bpy.data.worlds.get("GLN_World") or bpy.data.worlds.new("GLN_World")
    scene.world = w
    w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.012, 0.012, 0.016, 1.0)
    bg.inputs["Strength"].default_value = 1.0


def backdrop():
    """Curved cyclorama: floor sweeping up into a back wall."""
    mesh = bpy.data.meshes.new("STU_Backdrop")
    verts, faces = [], []
    w, depth, wall, radius, steps = 7.0, 3.0, 4.0, 1.4, 12
    import math
    profile = [(y, 0.0) for y in (-depth, 0.0)]
    for i in range(1, steps + 1):
        a = (math.pi / 2) * i / steps
        profile.append((radius * math.sin(a), radius * (1 - math.cos(a))))
    profile.append((radius, wall))
    for x in (-w / 2, w / 2):
        for y, z in profile:
            verts.append((x, y + 2.2, z))
    n = len(profile)
    for i in range(n - 1):
        faces.append((i, i + 1, n + i + 1, n + i))
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new("STU_Backdrop", mesh)
    G.link(obj, "GLN_Studio", "studio")
    G.set_material(obj, G.principled("MAT_Backdrop", (0.035, 0.034, 0.040), roughness=0.85))
    return obj


def helmet_stand():
    import bmesh
    mesh = bpy.data.meshes.new("STU_HelmetStand")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=0.16, radius2=0.16,
                          depth=HELMET_HEIGHT)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("STU_HelmetStand", mesh)
    obj.location = (HELMET_STAND.x, HELMET_STAND.y, HELMET_HEIGHT / 2)
    G.link(obj, "GLN_Studio", "studio")
    bev = obj.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments, bev.limit_method = 0.008, 3, "ANGLE"
    bev.harden_normals = True
    G.set_material(obj, G.principled("MAT_Stand", (0.020, 0.020, 0.024), roughness=0.35))
    return obj


def area_light(name, location, target, power, color, size):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = power
    data.color = color
    data.shape = "RECTANGLE"
    data.size, data.size_y = size
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    G.look_at(obj, target)
    G.link(obj, "GLN_Studio", "studio")
    return obj


def lights():
    # Warm key from camera-left-front, cool fill, two cool rims, soft top.
    area_light("LGT_Key", (-2.2, -2.6, 2.6), (0, 0, 1.35), 420, (1.0, 0.86, 0.72), (1.6, 1.0))
    area_light("LGT_Fill", (2.8, -2.4, 1.5), (0, 0, 1.1), 90, (0.75, 0.82, 1.0), (2.0, 2.0))
    area_light("LGT_RimL", (-1.8, 2.2, 2.2), (0, 0, 1.35), 380, (0.62, 0.74, 1.0), (0.5, 2.0))
    area_light("LGT_RimR", (1.9, 2.0, 2.1), (0, 0, 1.35), 300, (1.0, 0.70, 0.45), (0.5, 2.0))
    area_light("LGT_Top", (0.0, -0.4, 3.6), (0, 0, 1.0), 120, (1.0, 0.95, 0.9), (1.5, 1.5))


def cameras():
    for name, (loc, target, lens, ortho, _res) in CAMERAS.items():
        data = bpy.data.cameras.new(name)
        data.clip_start, data.clip_end = 0.05, 60.0
        if lens is None:
            data.type = "ORTHO"
            data.ortho_scale = ortho
        else:
            data.lens = lens
        obj = bpy.data.objects.new(name, data)
        obj.location = loc
        G.look_at(obj, target)
        G.link(obj, "GLN_Cameras", "cameras")
    bpy.context.scene.camera = bpy.data.objects["CAM_Hero"]


def run():
    G.ensure_dirs()
    scene = bpy.context.scene
    G.ensure_collections()
    G.clear_owned("studio")
    G.clear_owned("cameras")
    render_settings(scene)
    world(scene)
    backdrop()
    helmet_stand()
    lights()
    cameras()
    return {"cameras": list(CAMERAS), "root": G.ROOT}
