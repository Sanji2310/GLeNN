# GLeNN driver: build report

Stage: design approval and first checkpoint (complete blockout, organized objects, working cameras, test render).
Detailed modeling, retopology, UVs and the rig wait for the design to be approved.

## Checkpoint

| Item | Value |
| --- | --- |
| Blender | 5.2.2 LTS (Windows 11, RTX 3050) |
| File | `project/GLeNN_Character_Master.blend`, 35.9 MB |
| Built by | `scripts/build_character.py` in a background Blender, 86.3 s, then saved |
| Backup before the build | `backup/GLeNN_Character_Master_20261011_163837.blend` |
| Reopen | Reverted in the open Blender after the build: file loads clean, not dirty, no missing images |
| Scene camera | `CAM_Hero` |

## Collections

```
Scene Collection
  GLN_Character
    GLN_Body      (empty: the body is never seen bare, so the suit is sculpted directly)
    GLN_Head      HEAD_Skin, HEAD_Eye_L, HEAD_Eye_R, HEAD_Brows
    GLN_Hair      HAIR_Sculpt
    GLN_Outfit    SUIT_Body, SUIT_Gloves, SUIT_Boots
  GLN_Helmet      HLM_Root, HLM_Shell, HLM_Visor, HLM_Gasket, HLM_Trim, HLM_Pivots,
                  HLM_Spoiler, HLM_Intake, HLM_IntakeMouth, HLM_Vents, HLM_Number
  GLN_Rig         (empty until the Rigify stage)
  GLN_Studio      STU_Backdrop, STU_HelmetStand, LGT_Key, LGT_Fill, LGT_RimL, LGT_RimR, LGT_Top
  GLN_Cameras     CAM_Front, CAM_Side, CAM_Rear (orthographic), CAM_ThreeQuarter, CAM_Hero,
                  CAM_Face, CAM_Helmet
```

## Face counts (as built)

| Object | Faces | Notes |
| --- | ---: | --- |
| HEAD_Skin | 230,824 | Sculpt mesh (surface nets, 1.1 mm voxels) |
| HEAD_Eye_L / HEAD_Eye_R | 1,152 each | Clean UV spheres |
| HEAD_Brows | 681 | Strand strips |
| HAIR_Sculpt | 131,662 | Sculpt mesh (1.3 mm voxels) |
| SUIT_Body | 235,692 | Sculpt mesh (3 mm voxels) |
| SUIT_Gloves | 111,696 | Sculpt mesh (1.6 mm voxels) |
| SUIT_Boots | 65,258 | Sculpt mesh (2.2 mm voxels) |
| HLM_Shell | 1,512 | Quad grid, solidified and subdivided at render |
| HLM_Visor | 512 | Parametric patch, solidified and subdivided at render |
| HLM_Pivots, Spoiler, Intake, IntakeMouth, Vents, Number | 68, 142, 288, 288, 480, 371 | |
| **Total** | **781,778** | Gasket and Trim are curves |

The sculpt meshes are dense on purpose: they are design-stage sculpts, not game meshes. They are not production-ready. After approval they get retopology to game budgets, UVs, baked detail and a fresh check of the face and the back of the head.

## Design record

- Original young adult driver, about 24, about 7.1 heads tall. Stylized, not a likeness of any real person.
- Fictional team "GLeNN Racing", number 61. Palette: indigo, cream, amber, charcoal. No real sponsor, team or brand graphics.
- Suit: indigo one-piece with a V-shaped cream shoulder yoke, cream side panels from armpit to ankle, a cream stripe along the top of each sleeve, amber pinstripes outside the cream, amber waistband and cuffs. "61" on the right chest and large on the back, "GLeNN RACING" on the left chest and across the back yoke.
- Gloves: indigo with a charcoal palm grip, cream knuckle strip and amber strap.
- Boots: indigo with a cream sole, amber heel tab and amber strap.
- Helmet (separate and removable, shown on a stand): cream shell, indigo crown band and chin wrap, amber pinstripes, dark visor, 61 on the chin.

## Known limits at this stage

- The nose is still slightly large and the upper lip thin. These are face refinements for the next pass.
- The decal image path is stored as an absolute path in the .blend. It will be made relative before export.
- No UVs, rig, deformation test or exports yet. Those stages come after approval, and they are what the completion proof (wireframes, deformation test, poly counts, reopen) is based on.
