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
| `setup_scene.py` | Render settings, studio, lights and presentation cameras |
| `build_helmet.py` | The separate, removable helmet: a quad-grid shell, visor, trims and livery |
| `build_hair.py` | Sculpted hair: a scalp cap plus clump meshes projected onto the head |
| `deform_tools.py` | Shape edits that keep the topology, used to restyle the base mesh |
| `render_views.py` | Beauty renders, plus clay and real-wireframe inspection renders |
| `validate_character.py` | Polygon counts and mesh checks, written to `reports/` |
| `shader_dsl.py` | A small helper that writes shader node graphs as arithmetic |
