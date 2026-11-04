# External baselines

FlowEdit FLUX is adapted from the official [FlowEdit_utils.py](https://github.com/fallenshock/FlowEdit/blob/2620a9364f0f9c21368c36266e6200ff250273ec/FlowEdit_utils.py) at commit `2620a9364f0f9c21368c36266e6200ff250273ec`; [official FLUX config](https://github.com/fallenshock/FlowEdit/blob/2620a9364f0f9c21368c36266e6200ff250273ec/FLUX_exp.yaml) specifies 28 steps, n_avg=1, source guidance=1.5, target guidance=5.5, n_min=0, n_max=24, seed=10. Thus recommend 28/24 as the official default; 50/33 is a separate configuration, not the official checked-in FLUX default. At batch size 1 official 28/24 consumes 48 field evaluations, including 24 source and 24 target, with zero inversion NFE. A 50/33 run consumes 66. Match backbone, image size and prompts for comparison; record actual NFE and schedule. `b.steps` selects steps; set `flowedit_n_max=33` for 50/33. Use full-strength schedule (`b.experiment_strength=1`); the adapter records the actual schedule. `src` and `tgt` are precomputed embedding tuples. Optional keys: `flowedit_n_avg`, `flowedit_n_min`, `flowedit_src_guidance`, `flowedit_tgt_guidance`, `seed`. Noise is seeded on the backend device. FP32 latent integration is inherited from the host backend, while official code returns predictions and casts to model dtype; backend prompt length and VAE mode also follow the shared pipeline. No downloads or inversion are required at execution.

The [FlowEdit paper](https://arxiv.org/abs/2412.08629) integrates target-minus-source velocities on noisy source samples and an evolving clean edit state. It does not use source inversion or attention sharing. An actual per-method NFE budget may differ from other methods; report it rather than equating step counts.

[RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit/tree/7ac380143379b656eca5398157f41f97ef1ef2f9), commit `7ac380143379b656eca5398157f41f97ef1ef2f9`, implements RF-Edit via inversion plus attention feature sharing, alongside a two-evaluation second-order solver. Its FLUX image entrypoint is [edit.py](https://github.com/wangjiangshan0725/RF-Solver-Edit/blob/7ac380143379b656eca5398157f41f97ef1ef2f9/FLUX_Image_Edit/src/edit.py); [sampling.py](https://github.com/wangjiangshan0725/RF-Solver-Edit/blob/7ac380143379b656eca5398157f41f97ef1ef2f9/FLUX_Image_Edit/src/flux/sampling.py) passes inversion, timestep and injection state through its own Flux model. Integration requires adapting its transformer attention feature capture/injection to Diffusers, solver schedule and state handling, then checking equivalence with official examples; `b.field` alone cannot expose the requisite attention features. This is a separate substantial implementation, not included here. See [paper](https://arxiv.org/abs/2411.04746) and [example settings](https://github.com/wangjiangshan0725/RF-Solver-Edit/blob/7ac380143379b656eca5398157f41f97ef1ef2f9/FLUX_Image_Edit/src/run_horse.sh).

FlowEdit license (MIT):

Copyright (c) 2024 Vladimir Kulikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## DirectEdit: first priority for next baseline stage

Official repository commit `26b04bf576bfb83204bea619ae411d075faec99e`: [README](https://github.com/Tr1stesse/DirectEdit/blob/26b04bf576bfb83204bea619ae411d075faec99e/README.md), [single-image FLUX entrypoint](https://github.com/Tr1stesse/DirectEdit/blob/26b04bf576bfb83204bea619ae411d075faec99e/edit_real_flux_singleimg.py), [inversion and correction](https://github.com/Tr1stesse/DirectEdit/blob/26b04bf576bfb83204bea619ae411d075faec99e/inversion/flow_direct_correction_inv_flux.py), [attention controller](https://github.com/Tr1stesse/DirectEdit/blob/26b04bf576bfb83204bea619ae411d075faec99e/controller/attn_norm_ctrl_flux.py), [environment](https://github.com/Tr1stesse/DirectEdit/blob/26b04bf576bfb83204bea619ae411d075faec99e/environment.yml), [license](https://github.com/Tr1stesse/DirectEdit/blob/26b04bf576bfb83204bea619ae411d075faec99e/LICENSE). The code runs Euler source inversion storing all states and deltas, then uses stored deltas to align source/target inputs, restores the recorded source state, and shares attention values during the configured steps. This directly tests a closely related source-trajectory correction and sharing mechanism. An optional mask composites source outside the masked region; record mask usage separately.

Manual Linux/GPU commands from the official entrypoint, for a future run (not executed here):

```bash
git clone https://github.com/Tr1stesse/DirectEdit.git
cd DirectEdit
git checkout 26b04bf576bfb83204bea619ae411d075faec99e
conda env create -f environment.yml
conda activate directedit
mkdir -p outputs
python edit_real_flux_singleimg.py --num_steps 30 --inv_cfg 1 --recov_cfg 3 --attn_ratio 0.1 --src_path examples/1.jpg --src_prompt 'a cup of coffee with a drawing of a tulip put on the wooden table.' --tar_prompt 'a cup of coffee with a drawing of a lion put on the wooden table.' --saved_path outputs --seed 2024
```

The image entrypoint loads `black-forest-labs/FLUX.1-dev` via `from_pretrained` with FP16; all checkpoint components, tokenizer and text encoders must be present locally or obtainable with model access approved in Hugging Face. The single-image path does not load SD3.5 or SAM2; those are listed by the README for the broader project/demo and masks. The supplied environment includes Python 3.10, PyTorch 2.6.0, diffusers 0.32.2, CUDA 12 components, transformers and flash-attn; its pinned Linux-specific builds and flash-attn can require a suitable CUDA toolchain. The entrypoint defaults to CUDA, 30 steps, inversion guidance 1, recovery guidance 2, attention ratio 0.15, layer ratio 0 and seed 2024; the command uses the README single-image example's recovery 3 and attention 0.1. For a custom mask append `--mask_path /path/to/mask.png`. The checkpoint is referenced by Hub ID, so pre-populate the Hugging Face cache and use `HF_HUB_OFFLINE=1` if an entirely offline future run is required. These commands and dependencies are source-inspected, not runtime-verified.

## FlowEdit adapter source audit

With a full-strength schedule and matching scheduler configuration, the backend constructs the same `np.linspace(1, 1/T, T)` inputs and sequence-length shift as the official FLUX implementation, with the scheduler's terminal zero appended. Each skipped high-sigma step, paired source/target evaluation and Euler update, n_avg mean, n_min transition to target-only sampling, and final return choice follows the official algorithm. `b.field` uses sigma as transformer time; for the standard FlowMatchEuler schedule this corresponds to the official `t/1000`. Custom scheduler configurations should be checked for that correspondence. The adapter requires `n_max <=` the active schedule length, so use full strength and adjust n_max when changing step count.

Numerical and RNG differences: the official runner loads FLUX in FP16 and uses the global device RNG seeded once before iterating over all edits; its `randn_like` uses the packed latent's device and dtype. This adapter uses the host BF16 transformer, FP32 source/latent/noise and FP32 integration, and a new explicitly seeded device generator for each job. Consequently even matching seeds need not reproduce official samples, especially if the official runner has already consumed random numbers. CPU and CUDA generators also produce different random sequences for the same seed; the adapter uses `b.device` (the supplied backend sets CUDA), not CPU RNG. The official script uses the prompt encoder default max sequence length (512); the supplied backend defaults to 128 unless the parent changes it. Both use VAE posterior mode, but image preprocessing and precision are controlled by their respective pipelines. These are source-level observations; numerical equivalence and image quality have not been tested.

## Attention attribution and R78 definition

Inherited attention.py credits ReFlex, https://arxiv.org/abs/2507.01496, Sections 4.1-4.3 and B.2 for map/residual variants. R78 U25_SRC_ONLY uses the pre-existing attn_src branch: current target image queries with paired source-image K/V only, softmax normalized over source image tokens. At gate1, modified image-query output has no direct target-text value contribution; text queries retain native output. This is an ablation of the local implementation, not an official ReFlex baseline. No GAIN algorithm changes are included.
