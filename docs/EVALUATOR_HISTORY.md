# Local factual evaluator: versions and evidence

All inference used local Qwen2.5-VL weights and native experiment images. Model weights are not redistributed. These directories preserve the evaluation workflow, including unsuccessful diagnostic stages.

| Directory | Role | Use in reported results |
|---|---|---|
| `tools/local_qwen_judge` | Historical v6 rubric-and-rating pipeline, schema handling, progress and recovery | Excluded from the current factual comparison |
| `tools/qwen_diagnostic` | Synthetic-color, image-order, caption and prompt diagnostics | Diagnostic observations only |
| `tools/qwen_evidence_calibration` | Factual observation schema and known-condition checks | Diagnostic observations only |
| `tools/factual_qwen_v1` | Earlier factual evaluator and pair-control gate | Historical failed calibration; not a benchmark ranking |
| `tools/factual_qwen_v2` | Frozen factual rules, independent observations, ordered-pair/clean controls, resumable inference | Produces the v2 raw archive |
| `tools/qwen_offline_audit` | Offline parsing and dependency recovery | Retains original model responses and records recovered fields |
| `tools/factual_qwen_audit` | Recomputes corrected summaries and recovery sensitivity from archived records | Current reported exploratory factual proxy |

The runnable v2 package uses the existing R85 native outputs and local model. Its `RUN.txt` gives the original launch commands. On the prepared Linux server, `bash start.sh` launches the detached process and logs progress. The process runs its known-image checks before the 520-output benchmark. Each stage preserves errors and raw answers instead of converting unresolved observations to passes.

The compact v2 raw return contains 520 benchmark records. Offline recovery updates 51 dependent records after recovering 48 observations without model inference. All 435 previously resolved joint labels remain unchanged. The current edit-and-preserve event has 104 passes, 358 failures and 58 unknowns. Its missing-label bounds are not sampling confidence intervals.

`J_EP = E AND P` uses three-valued logic: any known failure makes the conjunction fail, all required conditions passing makes it pass, and remaining cases are unknown. Artifact calls were uniformly negative and lacked positive-artifact calibration, so artifact absence is not included as validated success evidence. Three task rules have known scope mismatches; the manuscript and audit exports include their exclusion sensitivity.

The R87 source-as-output identity-control launcher is a separate workflow. Its result is absent from the received fresh-source archive. Reusing the same frozen source observation on both sides is an algebraic consistency check, not an empirical estimate of judge specificity.

## Recompute without GPU inference

From the repository root:

```bash
python tools/factual_qwen_audit/recompute.py qwen_recomputed
python tools/factual_qwen_audit/recovery_sensitivity.py /path/to/A_Qwen_FactualJudge_v2_results.zip sensitivity_recomputed
```

The [research release](https://github.com/leiyurun3-sudo/rf-trace-memory/releases/tag/research-evidence-r87-20261104) includes the original diagnostic/result packages and fixes. Archive names identify historical versions; a successful program completion is distinct from a validated evaluator or a successful image edit.
