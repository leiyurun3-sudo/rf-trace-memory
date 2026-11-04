# Experiment archive index

R74–R83 and R86–R87 historical workflows and actual results. R71–R73 and R84–R85 remain in their existing directories. No generation, evaluation or tests were run during this synchronization.

`delivery` is the local experiment-delivery directory; `Downloads` is the local Downloads directory. Each tree has `PROVENANCE.json` with original ZIP byte size, member paths/CRC32, inclusion scope and omissions. Tables use the public release asset names; byte sizes identify the historical source ZIPs, not repackaged public assets. Release assets are handled separately; no unpublished asset URL is asserted.

## Workflow sources (delivery)

| experiments/ tree | Public release asset name | Original source ZIP bytes | Native input images omitted |
|---|---|---:|---:|
| [R74_MethodStudy](../experiments/R74_MethodStudy) | `R74_MethodStudy_Ready.zip` | 1915099 | 6 |
| [R74_SplitDebt](../experiments/R74_SplitDebt) | `R74_SplitDebt_Update.zip` | 3727 | 0 |
| [R75_MatchedStrength](../experiments/R75_MatchedStrength) | `R75_MatchedStrength_Update.zip` | 4999 | 0 |
| [R75_TransportStudy](../experiments/R75_TransportStudy) | `R75_TransportStudy_Ready.zip` | 2868466 | 6 |
| [R76_EditRelease](../experiments/R76_EditRelease) | `R76_EditRelease_Ready.zip` | 3233355 | 6 |
| [R77_ReleaseRouting](../experiments/R77_ReleaseRouting) | `R77_ReleaseRouting_Ready.zip` | 2613662 | 6 |
| [R78_EvidenceBridge](../experiments/R78_EvidenceBridge) | `R78_EvidenceBridge_Ready.zip` | 1923861 | 6 |
| [R79_RegionMemory](../experiments/R79_RegionMemory) | `R79_RegionMemory_Ready.zip` | 1922893 | 6 |
| [R80_RegionBalance](../experiments/R80_RegionBalance) | `R80_RegionBalance_Ready.zip` | 1923613 | 6 |
| [R81_ParallelEvidence](../experiments/R81_ParallelEvidence) | `R81_ParallelEvidence_Ready.zip` | 16004201 | 78 |
| [R82_EditAwareSupport](../experiments/R82_EditAwareSupport) | `R82_EditAwareSupport_Ready.zip` | 5003450 | 10 |
| [R82_SoftSupport](../experiments/R82_SoftSupport) | `R82_SoftSupport_Ready.zip` | 3766530 | 8 |
| [R83_RequestMemory](../experiments/R83_RequestMemory) | `R83_RequestMemory_Ready.zip` | 5364361 | 10 |
| [R86_ActualDoseControls](../experiments/R86_ActualDoseControls) | `R86_ProfileFix_v2.zip` | 59233 | 0 |
| [R87_FreshDoseConfirmation](../experiments/R87_FreshDoseConfirmation) | `R87_FreshDoseConfirmation_Ready.zip` | 7865940 | 20 |

R82_EditAwareSupport overlays `R82_Reporting_Hotfix_v2.zip` (12,315 bytes); the earlier `R82_Reporting_Hotfix.zip` is 12,277 bytes. R86 uses the v2 repair; the superseded `R86_ActualDoseControls.zip` is 56,100 bytes. These historical variants remain asset sources rather than duplicate code trees.

## Actual result sources (Downloads)

| results/ tree | Public release asset name | Original source ZIP bytes | Completed / total | Step summaries | Float maps |
|---|---|---:|---:|---:|---:|
| [R74_MethodStudy](../results/R74_MethodStudy) | `R74_MethodStudy.zip` | 66265801 | 136 / 136 | 0 | 0 |
| [R74_SplitDebt](../results/R74_SplitDebt) | `R74_SplitDebt_Results.zip` | 1428549 | 16 / 32 | 0 | 0 |
| [R75_TransportStudy](../results/R75_TransportStudy) | `R75_TransportStudy.zip` | 5489003 | 87 / 87 | 0 | 0 |
| [R76_EditRelease](../results/R76_EditRelease) | `R76_EditRelease.zip` | 6282724 | 84 / 84 | 84 | 0 |
| [R77_ReleaseRouting](../results/R77_ReleaseRouting) | `R77_ReleaseRouting.zip` | 4855821 | 84 / 84 | 84 | 0 |
| [R78_EvidenceBridge](../results/R78_EvidenceBridge) | `R78_EvidenceBridge.zip` | 7803208 | 98 / 98 | 98 | 0 |
| [R79_RegionMemory](../results/R79_RegionMemory) | `R79_RegionMemory.zip` | 5295376 | 70 / 70 | 70 | 0 |
| [R80_RegionBalance](../results/R80_RegionBalance) | `R80_RegionBalance.zip` | 4624527 | 98 / 98 | 98 | 0 |
| [R81_ParallelEvidence](../results/R81_ParallelEvidence) | `R81_ParallelEvidence.zip` | 21385758 | 366 / 366 | 366 | 0 |
| [R82_EditAwareSupport](../results/R82_EditAwareSupport) | `R82_EditAwareSupport.zip` | 8308343 | 80 / 80 | 80 | 0 |
| [R83_RequestMemory](../results/R83_RequestMemory) | `R83_RequestMemory.zip` | 8205090 | 60 / 60 | 60 | 0 |
| [R86_ActualDoseControls](../results/R86_ActualDoseControls) | `R86_ActualDoseControls_pilot_v2.zip` | 5213404 | 40 / 40 | 40 | 40 |
| [R87_FreshDoseConfirmation](../results/R87_FreshDoseConfirmation) | `R87_FreshDoseConfirmation_results.zip` | 21507403 | 160 / 160 | 160 | 160 |

## Running a workflow

Use Linux and an existing compatible GPU environment with complete local FLUX weights. Requirements are recorded in each workflow or inherited from R81. Set `PYTHON` and `FLUX_MODEL_PATH` to actual existing paths. Restore exact native inputs from the matching Ready ZIP:

```bash
cd experiments/R87_FreshDoseConfirmation  # select the desired round
python restore_inputs.py /path/to/R87_FreshDoseConfirmation_Ready.zip
PYTHON=/path/to/env/bin/python FLUX_MODEL_PATH=/path/to/FLUX RUN_IDENTITY=0 bash run_all.sh
```

R74_MethodStudy and R75_TransportStudy–R83_RequestMemory use `bash run_all.sh` after input restoration. Mask PNGs, frozen reference thumbnails, plans/configs and code are included. `restore_inputs.py` extracts only the listed original input images; existing destination files are refused. Previews are not substituted for native inputs. Optional evaluation uses archived entrypoints and locally available evaluator weights.

For R74 specifically, run from a fresh checkout with its six omitted source images absent:

```bash
cd experiments/R74_MethodStudy
python restore_inputs.py /path/to/R74_MethodStudy_Ready.zip
PYTHON=/path/to/env/bin/python FLUX_MODEL_PATH=/path/to/FLUX bash run_all.sh
```

The Ready asset must contain one top-level `R74_MethodStudy/` directory with the six original, unmodified `data/<case>/source.png` members listed in `PROVENANCE.json`. The helper removes exactly one archive-root component and matches the relative paths, uncompressed sizes and CRC32 values; renaming the ZIP and its root is compatible. Repacking the Git tree alone omits those six images and cannot restore them. Keep source PNG bytes unchanged when preparing the public Ready asset. Already extracted inputs need no restoration; the helper refuses to overwrite them.

Both R74 plans use `name = output_name = R74_MethodStudy` and the same 136 job configurations. The runner resolves inputs under `experiments/R74_MethodStudy/data/<case>/`. With no `OUTPUT_NAME` override, it writes working results under that experiment's `results/`, a preview mirror at repository-root `R74_MethodStudy/`, and a generated result archive named `R74_MethodStudy.zip` at repository root. The historical browseable result export remains under `results/R74_MethodStudy/`.

R74_SplitDebt and R75_MatchedStrength retain their historical installers. They require the complete R73 or R74 base workflow in a separate staging directory. Invoke from that base directory; these historical installers copy/replace their named output directory. For the R73 base, use `R73_ResidualBudget16_Ready.zip` if code is absent in the existing checkout. R75_MatchedStrength searches for the neutral R74_MethodStudy name.

R86: `R85=/path/to/original/R85_IndependentComparison R86_PYTHON=/path/to/env/bin/python bash start.sh pilot`. The pilot reuses 10 R85 anchors and generates 40 controls. It requires the matching original R85 code, plan, inputs, `record.json`, full `steps.json`, native `raw.png` and anchor metrics. Compact summaries cannot satisfy this original provenance contract. The archive entrypoint contains its original synthetic-backend check; it was preserved without execution here.

R87 retains its modified `runtime_base` source once because the driver executes it; generated runtime mirrors, caches and native outputs are excluded. The optional identity workflow requires the original 40-source R85 inputs and local Qwen weights. `RUN_IDENTITY=0` runs the 20-fresh-source, 2-seed, 4-arm generation/evaluation workflow alone. Identity results are absent from the supplied R87 result ZIP.

## Scope and missing assets

- Preserved: all available result CSV/JSON summaries, metrics, archived manifests, per-step summaries/records, compact previews/detail crops, response/gating maps, and all 200 R86/R87 NPZ float maps. Local post-hoc analysis is separately attributed under `local_analysis/`; assistant visual screening is not blinded human evaluation.
- Original result assets retain full comparison-sheet grids and generated HTML interfaces. Git omits those, native PNG outputs, full source images, weights, runtime mirrors, caches and logs. Per-tree provenance lists exact omissions.
- R74_MethodStudy has 136 native `raw.png` outputs in its original ZIP, omitted from Git. All other supplied result ZIPs lack native `raw.png`. Native-resolution evaluation needs server originals.
- R74_SplitDebt is partial: 16 completed of 32 total, 16 pending. Its actual 32-job plan is also retained in experiments/R74_SplitDebt/plan.json; the installer narrative described a larger intended matrix.
- R75_TransportStudy lacks per-image records and step summaries. Other rounds preserve compact step summaries where available; full `steps.json` / native server records are absent unless explicitly listed in provenance.
- No result archive was found for R75_MatchedStrength or R82_SoftSupport. Their source workflows are retained without claiming completed runs.
- Historical manifests keep original run fingerprints. Neutral naming/path changes and the R82 repair overlay are documented separately; archived hashes describe the historical run, not edited public text.

## Attribution and naming

PIE-Bench annotations are CC BY-NC-SA 4.0; image copyrights remain with their original holders. Dataset terms and attribution are retained in `DATA_LICENSE.txt` and R81 `DATA.md`. Include these terms with released data. Model weights are absent. FlowEdit MIT and RF-Edit Apache notices are retained in `external_sources.md` / `BASELINE.md` for inherited adapters; they do not grant a blanket license to all images or the repository.

The unique historical R74 root with an extra label between `R74_` and `_MethodStudy` maps to `R74_MethodStudy`. Original source archives are identified by the byte sizes above and local filename locators in provenance; the public release names are `R74_MethodStudy_Ready.zip` and `R74_MethodStudy.zip`. Output defaults, historical plan references and source-selection history use the neutral mapping. Historical narrative READMEs are excluded. Personal Windows-home prefixes in this export are normalized to `/local/user`; generic configurable server defaults remain. Credential-like literals were checked while copying selected text; logs and environment files were excluded.
