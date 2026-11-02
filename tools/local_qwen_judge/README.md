# Local Qwen-VL task evaluation

The evaluator runs Qwen2.5-VL-7B-Instruct on the experiment server. Images stay local. The first launch downloads a pinned model snapshot; inference then uses offline mode.

```bash
bash start_qwen_judge.sh
```

The launcher detects the existing R85 package and reuses `/root/autodl-tmp/MINT_Flow_Jupyter_Ready/.venv/bin/python`. Overrides: `R85_ROOT`, `QWEN_MODEL_DIR`, `QWEN_PYTHON`, `QWEN_JUDGE_OUT`. An optional model-download mirror is selected explicitly with `HF_ENDPOINT`. It does not upgrade the generation environment. Keep the GPU free for the evaluator.

The bundled plan fixes 520 outputs. Source and output native PNGs are required; compact previews are excluded. A source-only pass freezes one task rubric and subject/pattern applicability for all methods. Output scoring receives image roles, source/target descriptions, and the rubric, with method names and numerical metrics hidden. The four fields are edit, subject, pattern, and artifact, each containing value, applicable, and reason. Requested identity or appearance changes are edit objectives. Unknown ratings remain unresolved.

Greedy inference uses 768 maximum new tokens and processor max_pixels=512*512. Model revision, protocol, input identity, rubrics, prompts, raw responses, and errors are saved. Forty source-as-output controls and a fixed 20-output repeat audit are separate from the main denominator. Repeated greedy scoring checks consistency, not independent evaluator agreement.

Outputs include ratings.csv, arm_summary.csv, paired_summary.csv, summary.json, raw responses, applicability counts, unresolved bounds, and source-bootstrap intervals. Paired effects compare CORR25 with every other method using available paired source means. Missing scores are reported explicitly. RF-Edit contributes its single deterministic seed. The completed output folder is archived to `/root/autodl-tmp/A_Qwen_LocalJudge_results.zip`.

No VLM ratings have been generated in the local manuscript preparation environment. Published numerical tables contain the completed CLIP/DINO/LPIPS study; model-judged task outcomes are a separate evaluation layer. See the [official model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and [technical report](https://arxiv.org/abs/2502.13923).
