# GLeNN character (in progress)

This is the original stylised racing driver for GLeNN. It is being built in Blender 5.2.2 LTS, in stages that each need sign-off: design approval, then blockout, then face and hair, then body, suit and helmet, then UVs, then the Rigify rig, and finally exports.

The working project lives on the artist's PC in `Documents/GLeNN_Character/`, using the folder layout from the brief:

- `project/`
- `scripts/`
- `renders/`
- `exports/`
- `textures/`
- `reports/`

This repository folder mirrors the scripts. Each script targets Blender 5.2.2 and can be re-run safely, because it only rebuilds the objects it owns.

| Script | Purpose |
| --- | --- |
| `glenn_common.py` | Paths, the collection layout, the palette and shared helpers |
| `sdf.py` | Signed distance fields in numpy and a surface-nets quad mesher (the sculpting layer) |
| `design_head.py` | The head and neck as a sculpted distance field |
| `build_head.py` | Head mesh, eyes, brows and the skin, eye and brow materials |
| `design_hair.py` | The hair as a sculpted distance field: a faded cap plus clumps |
| `build_hair.py` | Hair mesh and material, plus the strand tools the brows use |
| `design_body.py` | Suit, gloves and boots as distance fields, with the livery panel masks |
| `build_body.py` | Suit, glove and boot meshes and materials, including the printed decals |
| `make_decals.py` | Draws `textures/suit_decals.png` (number 61 and GLeNN RACING) with Pillow, outside Blender |
| `setup_scene.py` | Render settings, studio, lights and presentation cameras |
| `build_helmet.py` | The separate, removable helmet: a quad-grid shell, visor, trims and livery |
| `build_character.py` | Rebuilds everything into the master .blend in a background Blender, after a backup |
| `bg_render.py` | Renders views in a background Blender on the GPU, so the open Blender stays responsive |
| `render_views.py` | Beauty renders, plus clay and real-wireframe inspection renders |
| `validate_character.py` | Polygon counts and mesh checks, written to `reports/` |
| `shader_dsl.py` | A small helper that writes shader node graphs as arithmetic |
| `deform_tools.py` | Shape edits that keep the topology (kept for later retopology work, not used by the build) |

## Rebuilding

From the Blender that has `project/GLeNN_Character_Master.blend` open, with `scripts/` on `sys.path`:

```python
import build_character
build_character.launch()        # backs up the master, then rebuilds it in the background
build_character.status()        # "running", "done" or "failed", with the log tail
bpy.ops.wm.revert_mainfile()    # once done, load the rebuilt file
```

The sculpted meshes are dense surface-nets output, which is right for the design stage. Retopology, UVs and the rig come after the design is approved.
