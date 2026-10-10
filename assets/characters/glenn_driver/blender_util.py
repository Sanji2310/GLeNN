"""Small bpy helpers shared by the build and preview scripts."""

import math
import numpy as np
import bpy
from mathutils import Vector, Matrix, Euler


def new_mesh_object(name, V, F, collection=None, smooth=True, attrs=None):
    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    nl = F.shape[1]
    me.loops.add(len(F) * nl)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, len(F) * nl, nl, dtype=np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.full(len(F), nl, dtype=np.int32))
    except (AttributeError, TypeError, RuntimeError):
        pass
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    if attrs:
        for an, arr in attrs.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 3:
                a = me.attributes.new(an, "FLOAT_VECTOR", "POINT")
                a.data.foreach_set("vector", arr.ravel())
            else:
                a = me.attributes.new(an, "FLOAT", "POINT")
                a.data.foreach_set("value", arr.ravel())
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def get_collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(col)
    return col


def set_input(node, names, value):
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in node.inputs:
            try:
                node.inputs[n].default_value = value
                return True
            except (TypeError, ValueError):
                pass
    return False


def look_at(obj, target, roll=0.0):
    d = Vector(target) - obj.location
    q = d.to_track_quat("-Z", "Y")
    obj.rotation_euler = q.to_euler()
    if roll:
        obj.rotation_euler.rotate_axis("Z", math.radians(roll))


def add_camera(name, loc, target, lens=50.0, ortho_scale=None, collection=None, dof=None):
    cam = bpy.data.cameras.new(name)
    if ortho_scale:
        cam.type = "ORTHO"
        cam.ortho_scale = ortho_scale
    else:
        cam.lens = lens
    cam.clip_start = 0.01
    cam.clip_end = 100
    if dof:
        cam.dof.use_dof = True
        cam.dof.focus_distance = dof[0]
        cam.dof.aperture_fstop = dof[1]
    ob = bpy.data.objects.new(name, cam)
    (collection or bpy.context.scene.collection).objects.link(ob)
    ob.location = Vector(loc)
    look_at(ob, target)
    return ob


def add_area_light(name, loc, target, energy, size, color=(1, 1, 1), size_y=None, spread=None, collection=None):
    l = bpy.data.lights.new(name, "AREA")
    l.energy = energy
    l.color = color
    if size_y:
        l.shape = "RECTANGLE"
        l.size = size
        l.size_y = size_y
    else:
        l.shape = "DISK"
        l.size = size
    if spread is not None:
        try:
            l.spread = math.radians(spread)
        except AttributeError:
            pass
    ob = bpy.data.objects.new(name, l)
    (collection or bpy.context.scene.collection).objects.link(ob)
    ob.location = Vector(loc)
    look_at(ob, target)
    return ob


def setup_cycles(scene, samples=64, res=(800, 800), denoise=True):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = denoise
    try:
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.02
    except AttributeError:
        pass
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.cycles.max_bounces = 8
    scene.cycles.transparent_max_bounces = 16
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = 0.3
    except TypeError:
        try:
            scene.view_settings.view_transform = "Filmic"
        except TypeError:
            pass
    # use a GPU when the host has one
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for t in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = t
            except TypeError:
                continue
            prefs.refresh_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = True
                scene.cycles.device = "GPU"
                return "GPU:" + t
        scene.cycles.device = "CPU"
    except Exception:
        scene.cycles.device = "CPU"
    return "CPU"


def render_to(scene, cam, path, res=None, samples=None):
    scene.camera = cam
    if res:
        scene.render.resolution_x, scene.render.resolution_y = res
    if samples:
        scene.cycles.samples = samples
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
