# Reproducing the experiments

This guide covers the published R71–R73 results, offline FLUX generation, and manual evaluation. The [research overview](../README.md) describes the method. Experiment code, plans, and inputs remain in `experiments/`; the published observations remain in `results/`. New runs write to separate directories.

```text
rf-trace-memory/
  requirements.txt
  experiments/
    .venv/bin/python          # interpreter selected by every run_all.sh
    R71_TraceMemory/          # run_all.sh; no start.sh
    R72_TimeBudget/           # run_all.sh and start.sh
    R73_ResidualBudget/       # run_all.sh and start.sh
  results/R71/               # published results
  results/R72/
  results/R73/
  resources/conference-template/            # original LaTeX template
  tools/summarize.py
  tools/build_report_assets.py
```

## 1. Inspect the published results

Run from the repository root with Python; this command uses only the standard library, reads the archived CSV files, and prints statistics:

```bash
python3 "tools/summarize.py"
```

| Experiment | Plan and source | Published gallery | New outputs | Embedded reference images |
|---|---|---|---:|---:|
| Source-memory and residual ablations | [R71](../experiments/R71_TraceMemory/plan.json) | [R71 gallery](../results/R71/index.html) | 90: 84 edits + 6 reconstructions | 0 |
| Correction and memory timing | [R72](../experiments/R72_TimeBudget/plan.json) | [R72 gallery](../results/R72/index.html) | 60 edits | 28 |
| Matched residual-RMS budgets | [R73](../experiments/R73_ResidualBudget/plan.json) | [R73 gallery](../results/R73/index.html) | 16 edits | 40 |

The three archives contain **166 new outputs, with zero generation failures**. Reference images reuse earlier job IDs and must be deduplicated when counting observations. R73 contains eight matched task/seed pairs: `cup_night` with seeds 10 and 17, and `printed_shirt_studio`, `rug_floor`, and `dog_rain` with seeds 0 and 17. Its `C50` comparison images come from R72. R72/R73 include the reference images and parameters needed for their comparisons, so the supplied plans can run independently of earlier rounds.

Expected R73 values from the archived CSVs:

| Quantity | Archived value |
|---|---:|
| Mean `pattern_ncc_shift6`, `BUDGET_EARLY` (8 images) | 0.378325704 |
| Mean `pattern_ncc_shift6`, R72 `C50` (8 references) | 0.350724800 |
| Mean `pattern_ncc_shift6`, `BUDGET_LATE` (8 images) | 0.334312572 |
| Maximum relative residual-budget error | approximately 2.583e-8 |
| Actual new transformer NFE | 1120 |
| Sum of standalone NFE | 1344 |
| Sum of recorded per-job time | approximately 678.634 seconds |

The summary script prints the new-output aggregates, budget error, costs, and timings. The three-way comparison is in [R73 comparison_metrics.csv](../results/R73/comparison_metrics.csv). Per-job time excludes model loading and the packaging performed after each job; it is not total run wall time. Human ratings in the published summaries are pending, so semantic and joint success rates remain unset.

## 2. Environment and dependencies

Generation commands below target **Linux or WSL with NVIDIA CUDA**, Bash, and a CUDA-capable PyTorch installation. The background launchers also use `flock`, `nohup`, and GNU `tail --pid`. Use a Python version supported by the chosen PyTorch build and the pinned libraries; the original Python version is not recorded in the published results.

| Component | Required setting or evidence |
|---|---|
| Diffusers | **0.35.1**, enforced by the loaders |
| Transformers | **4.55.4**, enforced by the loaders |
| Accelerate | **1.10.1**, enforced by the loaders |
| Model | `black-forest-labs/FLUX.1-dev`, base model with transformer `in_channels=64` |
| Model revision | **`3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`**, in all three plans |
| PyTorch / CUDA | Select a CUDA build compatible with the GPU, driver, and Python; no exact R71–R73 version is recorded in the published archive |
| Other generation dependencies | NumPy, Pillow, safetensors, SentencePiece, protobuf; historical versions are unrecorded |
| Report-asset dependency | Matplotlib for `tools/build_report_assets.py`; no historical version pin |

[requirements.txt](../requirements.txt) covers direct third-party imports, Accelerate offload, safetensors weight loading, T5 tokenizer dependencies, and the CPU report-asset builder. `PIL` is provided by Pillow. SentencePiece and protobuf support the T5 tokenizer/conversion path used by the pinned FLUX pipeline ([T5 source](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/t5/tokenization_t5.py), [FLUX source](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/flux/pipeline_flux.py)). Hugging Face Hub, tokenizers, and the remaining library dependencies are installed transitively. Other imports are Python standard-library modules or files in the experiment directory. No torchvision, torchaudio, OpenCV, SciPy, pandas, xformers, or evaluation model is directly required by these experiments.

The three exact library pins matter because the attention implementation imports Diffusers internals. The requirements file is a minimal dependency specification, not a complete historical environment lock. The published result packages omit `environment.json` and `model_inventory.json`: the [packaging allowlist](../experiments/R73_ResidualBudget/reporting.py) excludes both. Fresh runs create them under the experiment's own `results/` directory; retain these files separately when archiving a reproduction.

### Set up the interpreter

First change into the repository root, quoting the actual path. On WSL, a Windows D: checkout is typically under `/mnt/d/`. Create the environment **inside `experiments/`**, because every [run_all.sh](../experiments/R73_ResidualBudget/run_all.sh) resolves its interpreter as `experiments/.venv/bin/python`:

```bash
cd "/absolute/path/to/rf-trace-memory" &&
python3 -m venv "experiments/.venv"
```

Select the CUDA wheel index for your system using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), and export it as `TORCH_INDEX_URL`. Then, from the repository root:

```bash
(
  set -euo pipefail
  REPO="$(pwd -P)"
  PY="$REPO/experiments/.venv/bin/python"
  : "${TORCH_INDEX_URL:?Set TORCH_INDEX_URL to the CUDA wheel index for your GPU/driver}"
  "$PY" -m pip install torch --index-url "$TORCH_INDEX_URL"
  "$PY" -m pip install -r "$REPO/requirements.txt"
)
```

If the environment already has the intended CUDA-enabled PyTorch build, run only the requirements installation. Use this environment's interpreter for every Python command associated with generation; activation alone does not change the interpreter selected by `run_all.sh`.

### Prepare the pinned model cache

Set `HF_HOME` to an absolute cache directory in the same shell used to launch generation. The loader checks these locations, in this order:

1. `/root/autodl-tmp/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/<revision>`
2. `$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots/<revision>`

Here `<revision>` is the exact commit above. An arbitrary model directory, `HF_HUB_CACHE` alone, or an unset default Hugging Face cache is not searched by this loader. If both supported locations exist, the first one wins.

Obtain access to the model through Hugging Face and prepare its complete Diffusers snapshot before an offline run. For an online preparation step, after configuring Hugging Face authentication and exporting `HF_HOME`, the following command fetches that exact revision:

```bash
(
  set -euo pipefail
  REPO="$(pwd -P)"
  : "${HF_HOME:?Export HF_HOME as an absolute model-cache directory}"
  case "$HF_HOME" in /*) ;; *) printf '%s\n' 'HF_HOME must be absolute' >&2; exit 1 ;; esac
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    "$REPO/experiments/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
    cache_dir=str(Path(os.environ["HF_HOME"]) / "hub"),
)
PY
)
```

For a transferred cache, preserve the snapshot's symlink targets in `blobs/`, or materialize the linked files. Retain `model_index.json`, scheduler and tokenizer files, model configs, shard indices, and all weights for `transformer`, `text_encoder`, `text_encoder_2`, and `vae`. The runner forces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `local_files_only=True`; it does not fetch missing files. The loader checks safetensors headers and exact file lengths, and records header hashes. It does not compute full weight-content hashes.

### GPU and numerical settings

The [loader](../experiments/R73_ResidualBudget/model_loader.py) requires at least **10 GiB free GPU memory** at startup. This is its admission threshold; peak demand also depends on the GPU, software stack, and offload path. It selects model CPU offload at **27 GiB or more free**, and sequential CPU offload below that threshold. Sequential offload increases CPU/GPU transfers and runtime. Provide sufficient host RAM for the unquantized model and cached tensors; the archive records no host-RAM minimum or peak-memory measurement.

All supplied plans use a BF16 network, FP32 integration and projection, VAE tiling/slicing, 512 × 512 generation, 28 steps, and a maximum text length of 512. The loader disables CUDA matmul TF32 and cuDNN benchmarking. Use a GPU/PyTorch combination that supports BF16. The saved seeds and revision define the protocol; a different GPU or numerical stack can change output pixels.

## 3. Inputs and fixed protocol

Each experiment contains its required `data/<case>/source.png` and `outer.png`. All supplied jobs use `global_update=true` and box sampling. The update mask is constructed as all ones; `outer.png` is used for object/background measurements. The legacy `edit_mask: full_mask.png` field does not require an additional file for these plans.

The plans contain the exact source/target prompts, task rubrics, seed lists, 512-coordinate pattern/detail ROIs, and per-arm parameters. Inversion uses empty text, guidance embedding 0, `gamma=0.5`, and one substep per interval; editing uses guidance 3.5. Preserve the supplied job order when comparing amortized costs.

For `dog_rain`, the exact specification is:

> **Source:** A white French bulldog walking toward the camera on green grass, head lowered, daylight.
>
> **Target:** The same white French bulldog in the same walking pose, in rain, wet fur, overcast light, same dog, same pose.
>
> **Rubric:** Visible rain AND wet fur with overcast illumination. Preserve one dog and walking pose. A dry dog or purple deformation fails.

The requested changes are rain, wet fur, and overcast illumination; paved-road replacement is outside this prompt and rubric.

R73's `corr_weights`, `budget_h`, `budget_full_residual_rms`, and `correction_budget_target` are supplied in its plan, derived from the paired R72 `C50` records. R73 compares `BUDGET_EARLY` and `BUDGET_LATE` against those frozen budgets. Its engine checks interval sizes against `budget_h` with a 1e-7 tolerance and reports measured budget matching at a 1% relative-error threshold. Changing resolution, scheduler, prompts, or source images requires a newly calibrated experiment rather than reuse of the frozen R73 profiles.

## 4. Run and monitor

Run these blocks from the repository root after preparing the interpreter and model cache. Keep `HF_HOME` exported if using a custom cache. The blocks quote the current directory and paths, so spaces in the checkout path are supported.

### R73: start/resume and follow progress in one command

```bash
(
  set -euo pipefail
  REPO="$(pwd -P)"
  bash "$REPO/experiments/R73_ResidualBudget/start.sh"
)
```

The launcher starts a background run, or attaches to its existing log. It clears `PAUSE` when starting a new process. **Ctrl+C exits this log viewer while generation continues.** Progress messages report the current job, stage, step, completed count, and failures.

### R72: start/resume and follow progress

```bash
(
  set -euo pipefail
  REPO="$(pwd -P)"
  bash "$REPO/experiments/R72_TimeBudget/start.sh"
)
```

### R71: foreground run with live progress

R71 has `run_all.sh` and no `start.sh`. This block clears its pause marker and displays progress directly:

```bash
(
  set -euo pipefail
  REPO="$(pwd -P)"
  rm -f -- "$REPO/experiments/R71_TraceMemory/PAUSE"
  bash "$REPO/experiments/R71_TraceMemory/run_all.sh"
)
```

Ctrl+C requests a graceful pause of this foreground runner after the current image and packaging finish. Each round shares `experiments/.ab_queue.lock` and `experiments/.mint_runner.lock`; competing runs print `WAIT` and serialize. These locks cover this checkout's runners, not unrelated GPU programs. Launch one round at a time for straightforward cost accounting.

### Inspect or pause an existing R73 run

From the repository root, inspect the latest progress snapshot without attaching a GPU:

```bash
python3 -m json.tool "experiments/R73_ResidualBudget/results/progress.json"
```

The file appears after the runner acquires its queue locks. For an already-started background run, follow its log until its recorded PID exits:

```bash
(
  set -euo pipefail
  REPO="$(pwd -P)"
  PID="$(cat "$REPO/experiments/R73_ResidualBudget/.runner.pid")"
  tail --pid="$PID" -n 25 -F "$REPO/R73_ResidualBudget.log"
)
```

Pause after the current image:

```bash
touch "experiments/R73_ResidualBudget/PAUSE"
```

Wait for `status: paused` before starting again; the launcher reattaches while the old process is still active. Resume with the same `start.sh` block. A plan's 172800-second generation deadline also pauses between jobs; model loading occurs before that deadline starts. Completed records are skipped on resume after their artifact hashes are checked. An interrupted image without a completed record is rerun. Per-job errors are saved and the loop can continue, so read `progress.json` and `summary.json` for `failed` counts or `completed_with_errors`, even if the launcher exits normally.

## 5. Output locations, provenance, and packaging

For experiment directory `experiments/<experiment>/`, fresh output locations are:

| Path relative to repository root | Contents |
|---|---|
| `experiments/<experiment>/results/` | Working results, records, progress, environment, model inventory |
| `<experiment>_Images/` | Published-style copy of compact results, gallery, review page, and comparison sheets |
| `<experiment>_Results.zip` | Compact result archive refreshed after each image |
| `<experiment>.log` | Background launcher log for R72/R73 |

`<experiment>` is exactly `R71_TraceMemory`, `R72_TimeBudget`, or `R73_ResidualBudget`. The repository's archived [R71](../results/R71/), [R72](../results/R72/), and [R73](../results/R73/) directories are separate from these destinations and remain unchanged by the normal commands.

Each completed job writes `preview.jpg` (384 × 384, JPEG quality 90), `pattern.jpg` (ROI cropped from the generated 512 × 512 image, JPEG quality 92), `steps.json` (scalar diagnostics), and `record.json` (parameters, metrics, timings, provenance, and artifact hashes). **The native-resolution pattern crop is JPEG-compressed.** Full-size generated images, latents, and trajectory arrays are not exported. Per-round artifacts include source previews, contact sheets, `metrics.csv`, `arm_summary.csv`, `summary.json`, `manifest.json`, `plan.json`, and the offline `review/index.html`; R72/R73 also include comparison CSVs with references.

The [R71](../results/R71/manifest.json), [R72](../results/R72/manifest.json), and [R73](../results/R73/manifest.json) manifests record hashes of the original plan bytes, experiment `.py`/`.sh` files, and inputs. Resume checks the working and public output fingerprints. Preserve source files and line endings for an existing run; changes require a separate reproduction checkout/output tree. The runner has no `--output`, `--plan`, or single-job CLI option. Do not place the published archives into the working output path simply to start a new run.

To refresh a fresh run's gallery, review page, CSVs, and ZIP after generation has stopped, use the CPU-only packaging mode:

```bash
(
  set -euo pipefail
  REPO="$(pwd -P)"
  "$REPO/experiments/.venv/bin/python" \
    "$REPO/experiments/R73_ResidualBudget/runner.py" --pack
)
```

It uses Pillow/NumPy and takes the same queue locks. It reads the experiment's working results, not the repository's published `results/R73/`, and does not regenerate image metrics from JPEGs.

## 6. Interpret NFE and caches

NFE counts transformer evaluations in [FluxBackend.field](../experiments/R73_ResidualBudget/mint_backend.py); jobs use batch size one. It excludes text encoding, VAE encoding/decoding, manual attention operations, tensor transfers, and packaging.

| Configuration | Inversion | Target edit | Source edit | Standalone total |
|---|---:|---:|---:|---:|
| `RAW`, `ETA03` | 28 | 28 | 0 | 56 |
| R71 `KV100` | 28 | 28 | 7 | 63 |
| Residual-correction arms, including R73 | 28 | 28 | 28 | 84 |

The runner caches one inversion trajectory in host memory, up to two encoded source images, and prompt embeddings during a process. The inversion key includes source hash, case, resolution, seed, gamma, inversion substeps, actual sigma grid, schedule-reference setting, and empty conditioning. Adjacent compatible arms can reuse the trajectory; process restarts or eviction require inversion again. Trajectories are not saved to disk. Source-conditioned edit velocities and KV features are recomputed for each arm; the implementation does not provide a cross-arm source-feature cache.

`nfe_total` measures newly executed evaluations for a job, so a trajectory-cache hit has `nfe_inversion=0`. `standalone_inversion_nfe` preserves the inversion cost. R72/R73 also record `nfe_standalone`; for R71 compute it as `standalone_inversion_nfe + nfe_edit`. Compare standalone totals for per-image algorithm cost and actual totals for a particular ordered run. The archived R73 sequence performs eight inversions and sixteen edits: `8 × 28 + 16 × 56 = 1120` actual NFE, versus `16 × 84 = 1344` standalone NFE. Restarting midway can change actual totals and timing while keeping the standalone protocol unchanged.

## 7. Measurements and human review

The archived MAE/NCC measurements use saved **384 × 384 JPEG previews**, resized to 512 × 512 with Lanczos. They do not use the native pattern crop. `object_mae` and `bg_mae` use `outer.png` and its complement. Pattern NCC uses luminance in the supplied ROI; `pattern_ncc_shift6` searches integer offsets from −6 to +6 pixels in each axis. The stored `(dx, dy)` shifts the output ROI relative to the source ROI. Empty masks and flat ROIs receive explicit missing-value statuses.

For a new R73 run, open `experiments/R73_ResidualBudget/results/review/index.html` locally. It hides arm labels and includes the sixteen new images plus forty references. Score `edit_ok`, `subject_ok`, `pattern_ok`, and `artifact` as 0/1 using each task's exact rubric, export `R73_review.json`, and import it after the run has paused or completed:

```bash
(
  set -euo pipefail
  REPO="$(pwd -P)"
  "$REPO/experiments/.venv/bin/python" \
    "$REPO/experiments/R73_ResidualBudget/review.py" import \
    "/absolute/path/to/R73_review.json"
)
```

The importer checks the review schema, experiment fingerprint, job ID, and preview hash, then saves `judge.json` and refreshes the compact outputs. Use the review page belonging to that run: hashes from published images will not identify newly generated previews. R71 imports three fields (`edit_ok`, `subject_ok`, `artifact`); R72 adds `pattern_ok`. Their importers accept new jobs only. R73 accepts new jobs and embedded references.

The archived summary definitions differ: R71/R72 use completed, fully rated edits as their rate denominators; R73 uses all planned non-diagnostic jobs, counting generation failures as unsuccessful and leaving rates pending until remaining completed images are rated. R73 joint success requires `edit_ok=1`, `subject_ok=1`, `pattern_ok=1`, and `artifact=0`. Keep reconstruction diagnostics separate from editing outcomes and state the denominator when comparing rounds.

## 8. Rebuild derived report assets

The implemented [report-asset builder](../tools/build_report_assets.py) reads the frozen CSVs, records, native pattern JPEGs, and source PNGs. It requires NumPy, Pillow, and Matplotlib, uses the noninteractive `Agg` backend, and writes derived assets under `docs/`. Run from the repository root with the environment prepared above:

```bash
(
  set -euo pipefail
  REPO="$(pwd -P)"
  "$REPO/experiments/.venv/bin/python" "$REPO/tools/build_report_assets.py"
)
```

A separate CPU environment containing those three libraries can run the same script. Expected outputs are nine CSV tables under `docs/tables/`: `r71_task_means.csv`, `r72_task_means.csv`, `r73_task_means.csv`, `r73_comparison_task_means.csv`, `r73_compute.csv`, `r73_paired_budget.csv`, `budget_timing.csv`, `native_crop_sensitivity.csv`, and `native_crop_summary.csv`. The script also writes `docs/figures/budget_timing.svg` and its PNG preview, and prints a JSON summary. It preserves the experiment sources and archived results.

The native-crop sensitivity calculation compares each saved quality-92 `pattern.jpg` against the corresponding crop of the original source PNG, resized to the recorded generated dimensions. It computes fixed-coordinate luminance NCC without shift search. This is a separate measurement from the archived preview NCC: it changes both the image representation and the reference encoding. Its CSVs preserve the archived preview value alongside the new native-crop value.

## 9. Paper template

The original LaTeX template is under [resources/conference-template](../resources/conference-template/). With `pdflatex` installed, its supplied build script runs three passes:

```bash
bash "resources/conference-template/build.sh"
```

The output is `resources/conference-template/build/main.pdf`. Template compilation is independent of GPU reproduction.
