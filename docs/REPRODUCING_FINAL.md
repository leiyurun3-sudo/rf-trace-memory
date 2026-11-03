# Reproducing finalized R84/R85

Download the intact [R84_R85_FinalNight_Ready.zip](https://github.com/leiyurun3-sudo/rf-trace-memory/releases/download/evaluated-r84-r85-20261102/R84_R85_FinalNight_Ready.zip) from the [evaluated release](https://github.com/leiyurun3-sudo/rf-trace-memory/releases/tag/evaluated-r84-r85-20261102). The same release supplies the compact final results and manuscript PDF.

## Install into the existing MINT project

Use the existing Linux `MINT_Flow_Jupyter_Ready` project and its shared base FLUX environment. The archive adds a nested `R84_R85_FinalNight/` directory; preserve both round directories and their relative paths. This repository's `experiments/R84_R85_FinalNight/` is a code/plan snapshot; restore data and frozen references from the intact Ready ZIP before running it. Do not flatten the snapshot or launch it directly from the repository's experiments directory.

The existing environment must provide CUDA PyTorch, diffusers 0.35.1, transformers 4.55.4, accelerate 1.10.1 and the remaining dependencies listed in each round's `requirements.txt` and `requirements_eval.txt`. Generation uses the cached base FLUX.1-dev revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`, via the historical cache location, `HF_HOME`, or `FLUX_MODEL_PATH`. Generation does not download FLUX weights. The launcher shares project queue locks and `evaluation_models/`; evaluator preparation loads CLIP/DINO/LPIPS on CPU and may download their weights. It can install missing LPIPS 0.1.4 with `--no-deps`, retaining the existing torch stack.

These are the actual packaged launch commands (for later execution on the prepared server):

```bash
cd /root/autodl-tmp/MINT_Flow_Jupyter_Ready
unzip -o /root/autodl-tmp/R84_R85_FinalNight_Ready.zip -d .
bash R84_R85_FinalNight/start_final_night.sh
```

Wait for evaluator preparation `DONE` and `Started final night` before disconnecting. Ctrl-C stops log viewing; the detached run continues. Repeat the same launcher to resume. With native generation outputs still present, retry evaluation using:

```bash
cd /root/autodl-tmp/MINT_Flow_Jupyter_Ready
bash R84_R85_FinalNight/start_final_night.sh --eval-only
```

The launcher calls `run_final_night.sh`, then each round's `runner.py` and `evaluate_all.sh`. It writes the compact `R84_R85_FinalNight_results.zip` in the MINT project root. Native PNGs stay in each nested round's `results/` on the server and are needed for reevaluation. Compact numeric files alone do not supply those images.

## Frozen primary method and result mapping

| Final record | Snapshot round | Role | Completed native outputs |
| --- | --- | --- | --- |
| `results/R84/` | `R84_RequestRelease/` | Development, 10 sources, seeds 0 and 17 | 130 |
| `results/R85/` | `R85_IndependentComparison/` | Independent predeclared comparison, 40 sources, seeds 0 and 17 | 520 |

The latest frozen primary is `RELEASE_CORR25`: request response from three native prompt-difference probes on the same reference (sigmas 0.25, 0.5, 0.75), `route_floor=0.25`, `corr_lambda=0.25`, `correction_budget_mode=local_coefficient`, `match_residual_rms=false`. `engine.py` applies the local candidate with gain 0.25 directly, with no global RMS renormalization. Source K/V memory uses window `[0,7)`. This maps to the direct local coefficient branch; the separate `online_same_state`/ONLINE budget branch is not the finalized primary. `RELEASE_BOTH25` is the fixed secondary candidate; `ENERGY_MATCH25` is the scalar energy control. R85 includes U25_KV, FLOWEDIT_N26, RFEDIT_I08 and ETA03; RFEDIT uses seed 0 only. The plans retain all per-arm settings.

## Included evidence and provenance

`results/R84/` and `results/R85/` contain the finalized top-level CSV/JSON exports, `evidence/` statistics and the accompanying numeric `analysis/` summaries. They come from `FinalNight_Evaluated_Review/R84_FinalNight` and `R85_FinalNight`, rather than the older results embedded in the runtime source. `analysis/` files come from the adjacent final-review analysis directories; the supplied R85 analysis script is preserved under the code snapshot's `analysis/` directory. Root summary/evaluation/statistics files are retained as supplied, including overlapping exports.

Both final generation summaries record zero failed native generations and automatic evaluation status `ok` for all 130/520 native outputs. Their legacy human-rating fields remain empty (`edit_judged=0`, rates null). The subsequent local factual VLM assessment is archived separately in `results/QwenFactualV2`: 520 records, 384 fully resolved and 136 partially unresolved, including 58 unresolved joint outcomes. Its corrected aggregation is reproduced with `python tools/factual_qwen_audit/recompute.py qwen_recomputed` from the repository root. VLM outcomes are distinct from the legacy human fields. R84 historical reference evaluation counts are separately recorded: 20 `not_run`, 40 `not_run_no_native_png_in_compact_return`, 20 `pipeline_error`; these do not belong to the 130 native outputs.

`experiments/R84_R85_FinalNight/PROVENANCE.json` identifies sources, omissions, inspected data sizes and copied runtime files. Original absolute `/root/...` strings in configs/plans/evidence denote historical server paths; relative `data/...` names denote public benchmark inputs installed by the Ready ZIP. Images, the raw benchmark archive, frozen reference assets, logs, caches, per-job traces and HTML galleries are stored in release packages rather than this code snapshot. Existing source data provenance/license notices are preserved.
