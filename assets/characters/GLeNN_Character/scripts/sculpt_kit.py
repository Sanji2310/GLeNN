"""Drive Blender's own modeling and Sculpt Mode tools from the MCP connection.

Brush strokes go through bpy.ops.sculpt.brush_stroke, the same operator a
mouse drag runs, so they use the real brushes and their settings: radius,
strength, falloff, symmetry, Dyntopo or Remesh. A stroke is given as points
in world space; each point is projected into a 3D View region and the brush
raycasts the mesh from there, exactly as it would under the cursor.

Large form changes use Edit Mode with proportional editing (G with O), on
the vertex nearest a landmark.

The 3D View can be split into Quad View (Ctrl+Alt+Q). Strokes then go to the
pane that looks along the requested axis, so front and side work can be
mixed without turning the view.

    import sculpt_kit as K
    K.focus("CHR_Head", mode="SCULPT")
    K.quad(True); K.frame((0, 0, 1.68), 0.5)
    K.brush("Grab", radius=0.04, strength=1.0)
    K.stroke([(0.0, -0.09, 1.58), (0.0, -0.10, 1.57)], view="RIGHT")

Only works in the interactive Blender (it needs a 3D View).
"""

import contextlib

import bmesh
import bpy
from bpy_extras import view3d_utils
from mathutils import Vector

ESSENTIALS = "brushes/essentials_brushes-mesh_sculpt.blend/Brush/"
AXES = {"FRONT": Vector((0, 1, 0)), "BACK": Vector((0, -1, 0)), "RIGHT": Vector((-1, 0, 0)),
        "LEFT": Vector((1, 0, 0)), "TOP": Vector((0, 0, -1)), "BOTTOM": Vector((0, 0, 1))}


def area3d():
    win = bpy.context.window_manager.windows[0]
    area = max((a for a in win.screen.areas if a.type == "VIEW_3D"), key=lambda a: a.width * a.height)
    return win, area


def view_dir(region):
    return -(region.data.view_rotation @ Vector((0, 0, 1)))


def region_for(view=None):
    """The WINDOW region looking along `view` (an AXES key), or the largest
    one (the perspective pane in Quad View) when view is None."""
    win, area = area3d()
    regions = [r for r in area.regions if r.type == "WINDOW"]
    if view is None:
        return win, area, max(regions, key=lambda r: r.width * r.height)
    want = AXES[view]
    best = max(regions, key=lambda r: view_dir(r).dot(want))
    if view_dir(best).dot(want) < 0.999:
        raise RuntimeError("no pane looks along " + view)
    return win, area, best


def view3d():
    return region_for(None)


def override(view=None):
    if not bpy.context.window_manager.windows:
        return contextlib.nullcontext()     # background Blender: no 3D View needed
    win, area, region = region_for(view)
    return bpy.context.temp_override(window=win, area=area, region=region)


def quad(on=True):
    """Toggle Quad View (top, front, right and a free view)."""
    win, area = area3d()
    if bool(area.spaces.active.region_quadviews) != on:
        with override():
            bpy.ops.screen.region_quadview()


def frame(center, distance):
    """Centre and zoom every pane of the 3D View."""
    win, area = area3d()
    space = area.spaces.active
    for rv in list(space.region_quadviews) + [space.region_3d]:
        rv.view_location = Vector(center)
        rv.view_distance = distance


def mode(m):
    ob = bpy.context.view_layer.objects.active
    if ob and ob.mode != m:
        with override():
            bpy.ops.object.mode_set(mode=m)


def focus(obj_name, axis=None, mode="SCULPT", frame_it=False):
    """Make obj the only selected, active object and enter `mode`.
    axis turns a single (non-quad) view to FRONT, RIGHT, etc."""
    ob = bpy.data.objects[obj_name]
    with override():
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        if axis:
            bpy.ops.view3d.view_axis(type=axis)
        if frame_it:
            bpy.ops.view3d.view_selected()
        if mode != "OBJECT":
            bpy.ops.object.mode_set(mode=mode)
    return ob


def look(axis="FRONT", center=None, distance=None):
    """Turn the main view without changing mode; optionally frame a point."""
    win, area, region = view3d()
    with override():
        bpy.ops.view3d.view_axis(type=axis)
    r3d = region.data
    if center is not None:
        r3d.view_location = Vector(center)
    if distance is not None:
        r3d.view_distance = distance


def brush(name, radius=None, strength=None, direction=None):
    """Activate an Essentials sculpt brush. radius is in metres (scene
    space), so a stroke does the same thing whatever the zoom."""
    with override():
        bpy.ops.brush.asset_activate(asset_library_type="ESSENTIALS",
                                     relative_asset_identifier=ESSENTIALS + name)
    ts = bpy.context.tool_settings
    b = ts.sculpt.brush
    ups = ts.sculpt.unified_paint_settings
    if radius is not None:
        for s in (b, ups):
            s.use_locked_size = "SCENE"
            s.unprojected_size = 2.0 * radius      # Blender 5 stores the diameter
    if strength is not None:
        b.strength = strength
        ups.strength = strength
    if direction is not None and hasattr(b, "direction"):
        b.direction = direction
    return b


def stroke(points, pressure=1.0, mode="NORMAL", spacing=None, view=None):
    """Run one brush stroke through world-space points, seen from the pane
    looking along `view`. mode: NORMAL, INVERT (Ctrl, carve) or SMOOTH
    (Shift). spacing (metres) resamples the path so long segments still
    get dabs all the way along. pressure may be a function of 0..1."""
    win, area, region = region_for(view)
    r3d = region.data
    pts = [Vector(p) for p in points]
    if spacing and len(pts) > 1:
        dense = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            n = max(1, int((b - a).length / spacing))
            dense += [a.lerp(b, (i + 1) / n) for i in range(n)]
        pts = dense
    items = []
    for i, p in enumerate(pts):
        m = view3d_utils.location_3d_to_region_2d(region, r3d, p)
        if m is None:
            continue
        pr = pressure(i / max(1, len(pts) - 1)) if callable(pressure) else pressure
        items.append({"name": "", "location": p, "mouse": m, "mouse_event": m, "is_start": i == 0,
                      "pressure": pr, "size": 50.0, "x_tilt": 0.0, "y_tilt": 0.0, "time": float(i)})
    with bpy.context.temp_override(window=win, area=area, region=region):
        return bpy.ops.sculpt.brush_stroke(stroke=items, mode=mode)


def pmove(obj_name, target, delta, radius, falloff="SMOOTH"):
    """Edit Mode: select the vertex nearest `target` and move it by `delta`
    with proportional editing (radius in metres), mirrored across X."""
    ob = bpy.data.objects[obj_name]
    ob.data.use_mirror_x = True
    bm = bmesh.from_edit_mesh(ob.data)
    for v in bm.verts:
        v.select = False
    t = Vector(target)
    mw = ob.matrix_world
    v = min(bm.verts, key=lambda v: (mw @ v.co - t).length)
    v.select = True
    bm.select_flush(True)
    bmesh.update_edit_mesh(ob.data)
    with override():
        bpy.ops.transform.translate(value=delta, use_proportional_edit=True, proportional_edit_falloff=falloff,
                                    proportional_size=radius, use_proportional_connected=False, mirror=True)
    return tuple(round(c, 4) for c in mw @ v.co)


def op(fn, view=None, **kw):
    """Run any operator (bpy.ops.mesh.inset, ...) in the 3D View, as if
    invoked from its menu or hotkey there."""
    with override(view):
        return fn(**kw)


def sel(obj_name, pred, kind="VERT", extend=False):
    """Edit Mode select, like box or circle select: every vertex (or face,
    by its centre) whose world position satisfies pred. Returns the count."""
    ob = bpy.data.objects[obj_name]
    ts = bpy.context.tool_settings
    ts.mesh_select_mode = (kind == "VERT", kind == "EDGE", kind == "FACE")
    bm = bmesh.from_edit_mesh(ob.data)
    mw = ob.matrix_world
    if not extend:
        for f in bm.faces:
            f.select_set(False)
        for e in bm.edges:
            e.select_set(False)
        for v in bm.verts:
            v.select_set(False)
    if kind == "VERT":
        items = [(v, mw @ v.co) for v in bm.verts]
    elif kind == "EDGE":
        items = [(e, mw @ ((e.verts[0].co + e.verts[1].co) / 2)) for e in bm.edges]
    else:
        items = [(f, mw @ f.calc_center_median()) for f in bm.faces]
    n = 0
    for item, co in items:
        if pred(co):
            item.select_set(True)
            n += 1
    bm.select_flush_mode()
    bmesh.update_edit_mesh(ob.data)
    return n


def drag(obj_name, moves, proportional=0.0):
    """Edit Mode: grab (G) each vertex index to its world-space target, one
    at a time. With proportional > 0 the neighbours follow (O)."""
    ob = bpy.data.objects[obj_name]
    mw = ob.matrix_world
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    for i, target in moves:
        bm = bmesh.from_edit_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        for f in bm.faces:
            f.select_set(False)
        for v in bm.verts:
            v.select_set(False)
        v = bm.verts[i]
        v.select_set(True)
        d = Vector(target) - mw @ v.co
        bmesh.update_edit_mesh(ob.data)
        if d.length < 1e-6:
            continue
        with override():
            bpy.ops.transform.translate(value=tuple(d), use_proportional_edit=proportional > 0,
                                        proportional_edit_falloff="SMOOTH",
                                        proportional_size=max(proportional, 1e-4), use_proportional_connected=False)


def nearest(obj_name, point, among=None):
    """Index of the vertex nearest a world-space point (Edit Mode)."""
    ob = bpy.data.objects[obj_name]
    mw = ob.matrix_world
    bm = bmesh.from_edit_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    pool = bm.verts if among is None else [bm.verts[i] for i in among]
    p = Vector(point)
    return min(pool, key=lambda v: (mw @ v.co - p).length).index


def remesh(obj_name, voxel):
    """Sculpt Mode > Remesh (Ctrl+R) at the given voxel size."""
    ob = bpy.data.objects[obj_name]
    ob.data.remesh_voxel_size = voxel
    ob.data.use_remesh_fix_poles = True
    with override():
        bpy.ops.object.voxel_remesh()
    return len(ob.data.polygons)


def surface(obj_name, origin, direction):
    """Where a ray from origin along direction (world space) first hits the
    object's evaluated mesh, modifiers included. Works in Object Mode."""
    ob = bpy.data.objects[obj_name]
    mw = ob.matrix_world
    mi = mw.inverted()
    ok, loc, n, i = ob.ray_cast(mi @ Vector(origin), mi.to_3x3() @ Vector(direction))
    return tuple(round(c, 4) for c in mw @ loc) if ok else None
