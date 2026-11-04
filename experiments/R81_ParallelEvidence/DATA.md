# R81 PIE-Bench heldout inputs

Prepared 2026-10-30. Ready for parent package integration: **24 distinct sources × seeds [0,17] = 48 source-seed pairs per arm**. All image and prompt inputs are real dataset records. No generation or GPU evaluation was run.

## Availability and provenance

| Item | Exact count |
|---|---:|
| Pre-existing official mapping files in D:/科研论文资料 and /local/user/Downloads | 0 |
| Pre-existing images under annotation_images in those roots | 0 |
| ZIP archives whose member lists were searched | 353 (219 D:, 134 Downloads) |
| R80 reference source images read for content exclusion | 6 |
| Records in the publicly accessible mirror mapping | 700 |
| Source image members listed in the mirror input ZIP | 700 |
| Source images actually retrieved and retained | 24 |
| Local selected mapping records with readable source images | 24 |
| Distinct eligible and selected source contents | 24 |
| Supplied official mask records / decoded evaluation masks | 24 / 24 |
| Missing images / invalid records / unreadable images in selected subset | 0 / 0 / 0 |
| R80 content overlaps / duplicate contents among selected images | 0 / 0 |

Original bounded searches covered 01_ABC研究项目, 02_RF_Inversion完整资料, 实验交付 and Downloads, then expanded to both requested roots and ZIP member lists. A broad filename scan also found 586 historical image/result files containing 38 different 12-digit strings. Those include generated results, comparisons and unrelated blind-review IDs; they are not 38 verified PIE source images. Pre-existing usable official image+mapping pairs: **0**.

Actual retrieval source: [third-party PIE-Bench evaluation handoff mirror](https://huggingface.co/datasets/yanlinli/image-editing-evaluation-handoff). Mapping: `evaluation/PIE-Bench_DirectInversion/data/PIE-Bench_v1/mapping_file.json`; input archive: `archives/image_editing_piebench_inputs.zip` (70,968,291 bytes). HTTP Range requests read its central directory and selected compressed image members. The full ZIP and the full image dataset were not downloaded. The complete 700-record mapping was read into memory; only 24 selected records were saved locally. This is a mirror-derived subset, with schema and licensing checked against primary sources below.

## Exact integration schema

CSV ready for the parent's preparer: **heldout_data/manifest.csv**. UTF-8, one row per distinct source. Header:

```text
case,image,source_prompt,target_prompt,edit_rubric,target_category_id,target_category,seeds,independent_eval_annotation,independent_eval_mask
```

`case` is a 12-character string; preserve leading zeros. `image` is relative to the CSV's directory, e.g. `sources/000000000072.jpg`. `source_prompt` and `target_prompt` come from official `original_prompt` and `editing_prompt`; only bracket markers are removed, following official inference code. `edit_rubric` is the supplied `editing_instruction` (the tool falls back to the actual target prompt if the instruction is absent). `target_category_id` is the official editing_type_id string; `target_category` is its published category name. `seeds` is the literal CSV cell `0,17`; use global `--seeds 0,17` in the parent preparer.

The two `independent_eval_*` paths are relative to the CSV directory. They point to the raw official record and its decoded evaluation mask. **Do not map either field to `object_mask`, `pattern_roi`, `detail_roi`, route masks, method masks or method parameter selection.** They must be consumed only by evaluation after generation. No method ROI or object_mask is supplied. AUTO_DETAIL/COARSE_BOX must construct their masks through their independently defined source-image method. No arm templates, plan or engine were changed by this data task.

JSON: **heldout_tasks.json** is an object with `tasks` (24 records), `seeds`, `source_count`, `source_seed_pairs`, `counts`, `evaluation_annotations`, `evaluation_masks` and provenance. Task fields:

```text
case: str
key/task: "pie_" + case
source_image/image: relative path from heldout_tasks.json directory
source_prompt/target_prompt/edit_rubric: str
target_category_id/target_category: str
seeds: [0,17]
source_rgb_sha256: str
source_size: [width,height]
```

Evaluation annotations and masks are separate maps keyed by case, not method inputs. Raw official records retain brackets, blended_word, masks and any additional supplied fields. The JSON annotation records dataset provenance rather than newly collected human labels. No target image or newly invented evaluation rubric is supplied.

## Selection and evaluation

Selection uses SHA256(`R81-PIE-heldout-v1:` + case) ordering and category round-robin, one edit per source. Categories 0–3 have 3 sources each; categories 4–9 have 2 each. Extraction followed this order over the 700-record mapping, excluding the six R80 IDs, then comparing each retrieved source with all R80 reference contents and previously selected contents. No retrieved candidate required a content-overlap rejection.

R80 exclusions: `121000000003`, `321000000001`, `423000000000`, `522000000000`, `622000000002`, `723000000002`. Comparisons use EXIF-normalized RGB identity and a near-copy rejection rule (64-bit dHash distance <=8 AND 64×64 RGB mean absolute difference <=12). This covers identical pixels and similar resized/re-encoded copies; the actual selected images pass both comparisons. Source identities and excluded reference identities are recorded in heldout_tasks.json.

Official RLE is zero-based flattened row-major `(start,length)` pairs on a 512×512 mask. Decoded PNGs contain edited=255, unedited=0 and follow the official evaluator's policy of setting all four border edges to edited. Background preservation uses the complement. These labels remain independent of the method's own masks. Original records remain available for evaluation variants using the unmodified RLE.

Frozen selected data are already prepared; parent integration does not require network access. The preparation tool requires an explicit dataset root and Pillow, and never downloads data:

```powershell
python -B D:/科研论文资料/实验交付/R81_ParallelEvidence/prepare_pie.py --dataset-root D:/科研论文资料/实验交付/R81_ParallelEvidence/heldout_data/raw_pie --count 24 --inventory-only
```

For a fresh output location, copy prepare_pie.py there and run it with `--dataset-root <actual PIE root> --exclude-plan <R80 plan.json> --count 12` or `--count 24`. R80's six `data/<case>/source.png` files must accompany the exclusion plan. The tool refuses to overwrite existing selected outputs and refuses to produce a partial cohort. Raw PIE input subset is retained in heldout_data/raw_pie for reproducibility; it duplicates the same 24 source contents, not an additional 24 sources.

## Primary sources and license

- [Official repository and schema](https://github.com/cure-lab/PnPInversion): data/mapping_file.json plus data/annotation_images; record fields image_path, original_prompt, editing_prompt, editing_instruction, editing_type_id, blended_word and mask. The README's schema paragraph accidentally names the TI2I mapping; the inference implementation explicitly reads mapping_file.json for PIE.
- [Official inference implementation](https://github.com/cure-lab/PnPInversion/blob/main/run_editing_p2p.py): prompt bracket removal and image lookup.
- [Official evaluation decoder](https://github.com/cure-lab/PnPInversion/blob/main/evaluation/evaluate.py): mask encoding and border convention.
- [Official access form](https://forms.gle/hVMkTABb4uvZVjme9) links the [dataset license](https://docs.google.com/document/d/1K7pjAgGwS8-lsA-D6Me4jAKmYszKpcyydiAEyZuTm2I/edit), retrieved through its public text export. Annotations use **CC BY-NC-SA 4.0**: attribution, noncommercial use, share-alike derivatives and indication of changes. Image copyrights remain with the authors/original copyright holders. The form limits the dataset to noncommercial research. The project website's CC BY-SA notice concerns that website and is not the data license. Root LICENSE/LICENSE.md/LICENSE.txt were absent in the repository when checked.

Attribution: Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, Qiang Xu, *PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code*, ICLR 2024 (the data license uses the earlier title *Direct Inversion*). Changes in this subset: deterministic selection, bracket removal in method prompts, decoded evaluation-mask PNGs and packaging; original selected annotation records are preserved.
