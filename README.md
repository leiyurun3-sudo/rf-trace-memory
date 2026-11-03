# TraceMemory

**Spatial source correction and memory for training-free rectified-flow image editing**

Updated **3 November 2026** with corrected factual-judge evidence for R85.

[Manuscript PDF](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Development controls](paper/development_dose.tex) · [Final evaluation](paper/final_evaluation.tex) · [Run instructions](docs/REPRODUCING_FINAL.md) · [Download packages](https://github.com/leiyurun3-sudo/rf-trace-memory/releases/tag/evaluated-r84-r85-20261102)

## Research question and current method

Text-guided editing must change the requested attributes while preserving unaffected content. TraceMemory investigates how the **location, magnitude, and memory coverage** of source correction affect this balance.

The frozen primary configuration is **TraceMemory (CORR25)**, recorded as `RELEASE_CORR25` in the plans and results. Three source/target prompt-response probes on an inverted reference define an edit-response map. Its complement supplies a soft protection weight. The method applies the protected residual directly, with coefficient 0.25, and uses paired source K/V memory during the first seven editing steps.

For reference states $r_i$, edited state $z_i$, and positive generation interval $h_i$:

$$
p_i=\frac{r_{i+1}-r_i}{h_i},\qquad c_i=p_i-u_i^s(z_i),\qquad P=1-0.75E.
$$

$$
z_{i+1}=z_i+h_i\left[u_i^{t,M}(z_i)+0.25P\odot c_i\right].
$$

The response $E$ is estimated from native source/target field differences at three reference nodes. It uses source and target prompts; benchmark edit masks enter evaluation only. The target velocity acts over the full image. Memory preserves the native target-text contribution and image-attention mass at each modified layer. The primary correction uses **direct local scaling without global RMS restoration**. Earlier `ONLINE` configurations are retained as historical controls.

The `ENERGY_MATCH25` control applies

$$
\widehat c_i^{\rm energy}=0.25\frac{\mathrm{RMS}(P\odot c_i)}{\mathrm{RMS}(c_i)}c_i.
$$

This matches the local rule's correction RMS at a common state while keeping its global direction. Each arm follows its own trajectory. The paired experiment tests spatial allocation beyond instantaneous scalar weakening.

`RELEASE_BOTH25` is the preregistered secondary configuration: it additionally weights memory queries by protection support. Primary and secondary roles were frozen before the independent comparison.

## Final evaluated results

**650 outputs completed, zero generation failures, and automatic evaluation completed for all outputs.** R84 contains 130 development outputs on ten sources. R85 contains 520 outputs on 40 new sources, balanced across ten PIE-Bench categories. Six methods use two seeds per source; deterministic RF-Edit uses one run per source.

The table reports equal-source means. Preservation LPIPS uses the 33 sources with nonempty preservation masks; other metrics use all 40 sources. CLIP and DINO are cosine-based automatic scores. NFE counts standalone transformer calls, including inversion, memory capture, and response probes.

| Method | CLIP target ↑ | Directional CLIP ↑ | DINO ↑ | LPIPS full ↓ | LPIPS preserved ↓ | Preserved MAE ↓ | NFE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TraceMemory (CORR25)** | 0.2629 | 0.0863 | 0.5904 | 0.5419 | 0.5142 | 0.1169 | 97 |
| BOTH25, secondary | 0.2733 | 0.1069 | 0.5579 | 0.5620 | 0.5269 | 0.1221 | 97 |
| Same-state energy control | 0.2622 | 0.0881 | 0.5856 | 0.5483 | 0.5226 | 0.1202 | 97 |
| Uniform correction, U25 | 0.2632 | 0.0908 | 0.6111 | 0.4971 | 0.4743 | 0.0972 | 91 |
| FlowEdit N26 | 0.2712 | 0.1177 | 0.5737 | 0.4437 | 0.4281 | 0.1207 | 52 |
| RF-Inversion ETA03 | 0.2694 | 0.1065 | 0.4670 | 0.5431 | 0.5322 | 0.1062 | 56 |
| RF-Edit I08 adapter | 0.2453 | 0.0383 | 0.7875 | 0.2778 | 0.2688 | 0.0455 | 100 |

![Final metrics and source-bootstrap intervals](paper/figures/final_metrics.png)

### Findings

1. **Spatial allocation improves perceptual preservation against the same-state energy control.** CORR25 reduces preservation LPIPS by **0.0084**, with a 95% source-bootstrap interval **[−0.0143, −0.0022]** and Holm-adjusted **p = 0.0238**. Full-image LPIPS decreases by 0.0064. The directional CLIP difference is −0.0018 with an interval spanning zero.
2. **Memory attenuation increases semantic editing and source departure together.** BOTH25 increases directional CLIP by **0.0207** over CORR25, while preservation LPIPS increases by **0.0127** and DINO decreases by 0.0325.
3. **FlowEdit N26 achieves stronger directional alignment and lower perceptual preservation error at lower cost.** Its directional CLIP exceeds CORR25 by 0.0315, and its preservation LPIPS is lower by 0.0861, using 52 versus 97 NFE. Uniform U25 also has lower preservation LPIPS than CORR25. CORR25 improves DINO over ETA03 by 0.1234.
4. **The development dose study shows a preservation response.** Increasing the direct correction coefficient from 0.15 to 0.25 to 0.35 changes preservation LPIPS from **0.5544 → 0.4983 → 0.4561**. Semantic-score differences have broad intervals on this ten-source development set.

These results establish a modest controlled localization benefit and quantify the memory trade-off. The independent comparison identifies FlowEdit and uniform correction as strong reference configurations for further algorithm development.

### Statistical and evaluation protocol

Seeds are averaged within each source before inference. Reported paired intervals use source-bootstrap resampling; tests use source sign flips and Holm correction across the final six-comparison analysis family within each metric. The frozen plan specifies CORR25 versus FlowEdit as the primary comparison and assigns the added ETA03 comparison to a separate secondary family; the final exporter groups all six CORR25 comparisons within each metric. The manuscript appendix documents this grouping difference and broader-family sensitivity. Full numeric records retain the comparison family, sample counts, and unrounded estimates.

LPIPS preservation is a spatial-feature-map average weighted by the official edit-mask complement, with receptive-field spill across its boundary. Empty complements are excluded from that metric and counted explicitly. CLIP-T, directional CLIP, CLIP-I, DINOv2, LPIPS, PSNR, and SSIM exports are included. Task-specific evaluation uses frozen task rules, separate factual observations of source and output, and paired preservation judgments with anonymized method identities. Corrected factual-judge results are reported below; the automatic R84/R85 metrics above are unchanged.

### Corrected factual-judge evaluation (v2)

The local Qwen-VL factual judge evaluated all 520 R85 outputs. Edit requires all frozen target conditions and a source-to-target change; protected means applicable source requirements are preserved. Joint pass requires edit and protected pass and no severe artifact. Counts use all planned outputs; unknowns remain unknown.

| Method | Outputs | Edit pass / unknown | Protected pass / unknown | Joint pass / unknown |
|---|---:|---:|---:|---:|
| TraceMemory (CORR25) | 80 | 36 / 1 | 39 / 20 | 16 / 8 |
| BOTH25 | 80 | 42 / 0 | 38 / 24 | 19 / 13 |
| Same-state energy control | 80 | 35 / 0 | 37 / 24 | 14 / 9 |
| Uniform correction, U25 | 80 | 40 / 0 | 41 / 23 | 20 / 9 |
| FlowEdit N26 | 80 | 41 / 0 | 41 / 18 | 16 / 12 |
| RF-Inversion ETA03 | 80 | 42 / 0 | 34 / 18 | 16 / 6 |
| RF-Edit I08 adapter | 40 | 4 / 0 | 25 / 9 | 3 / 1 |

Across all outputs, protected is unknown for 136, edit for 1, and joint for 58. A known failure can determine joint failure despite another unknown. All 520 severe-artifact judgments were negative. All 15 synthetic controls passed (six factual, six paired, three clean-artifact); there were no positive artifact controls, so artifact sensitivity remains unmeasured.

Offline parser recovery extracted explicit answers from 48 previously rejected observations, affecting 51 records; one conflicting observation remains uncertain. This fixes response formatting, while semantic unknowns remain in the corrected records. The frozen task rules have specific coverage limits: the giant rule checks presence rather than lake position; the watercolor rule checks brushwork rather than Chinese style; and the sofa rule specifies rusty iron and allows color changes. The older v6 judge is excluded from ranking; this table uses the corrected v2 judge.

The [archived records and summaries](results/QwenFactualV2) include pointwise, unadjusted bootstrap intervals on complete sources and paired complete-source intersections. Those intervals describe the selected complete sources, not the full 40-source population, and do not establish confirmatory superiority. Recompute the aggregation with Python 3:

```bash
python tools/factual_qwen_audit/recompute.py qwen_recomputed
```

The [recovery implementation](tools/factual_qwen_audit/recover.py) documents the original format repair; recomputation above starts from archived corrected records and requires no model or images.

### Qualitative examples

![Source, CORR25, BOTH25, energy control, and FlowEdit on four seed-0 examples](paper/figures/final_qualitative.jpg)

Columns: source, CORR25, BOTH25, energy control, FlowEdit N26. Rows: grass-to-water, wall-to-red, parrots-kissing, and camera-to-phone. The wall example illustrates localized attribute editing; FlowEdit more clearly achieves the requested parrot interaction. Dog and person outputs show identity or appearance changes. Population-level comparisons use the complete declared source set.

## Manuscript and local AI evaluation

The extended manuscript contains the method, experiments, statistical analysis, and appendices. It accompanies the executable evaluation code and archived numerical evidence.

The current factual-judge comparison is reproduced from the corrected records in `results/QwenFactualV2` using `tools/factual_qwen_audit/recompute.py`. Inference ran locally with Qwen2.5-VL-7B-Instruct; images were not uploaded. The earlier runner under `tools/local_qwen_judge` is historical, and its v6 scores are excluded from the current comparison.

## Files and reproduction

| Location | Contents |
|---|---|
| [paper/main.pdf](paper/main.pdf) | Updated extended manuscript and appendices |
| [paper/main.tex](paper/main.tex) | Manuscript source incorporating final evaluated results |
| [results/QwenFactualV2](results/QwenFactualV2) | Corrected factual-judge records, recovery audit, and summaries |
| [results/R85](results/R85) | Independent comparison: evaluation CSV, metrics, summaries, source-paired evidence |
| [results/R84](results/R84) | Development controls and dose response |
| [experiments/R84_R85_FinalNight](experiments/R84_R85_FinalNight) | Frozen code, plans, selectors, launchers, evaluators, and statistics |
| [docs/REPRODUCING_FINAL.md](docs/REPRODUCING_FINAL.md) | Environment, generation, evaluation-only restart, and package layout |
| [Release assets](https://github.com/leiyurun3-sudo/rf-trace-memory/releases/tag/evaluated-r84-r85-20261102) | Frozen 2 November package, numerical results, and manuscript snapshot; current PDF is linked above |
| [README_R71_R73.md](README_R71_R73.md) | Archived early timing/budget study; original code and results remain available |

The runnable ZIP contains benchmark inputs and frozen references needed by the nested launchers. Native generated PNGs remain on the experiment server for image-metric reevaluation; compact return packages provide thumbnails, comparisons, numerical results, and small logs. Historical absolute paths in frozen records preserve run provenance. The reproduction guide identifies the runtime root and restore procedure.

Rebuild the final figure from checked-in numerical evidence:

```bash
python paper/scripts/build_final_metrics.py \
  --summary results/R85/evidence/overnight_arm_summary.csv \
  --out paper/figures/final_metrics.pdf
```

Compile the manuscript from its directory with Tectonic:

```bash
cd paper
tectonic main.tex
```

Third-party model, dataset, and adapted-method terms remain applicable. Dataset provenance and original notices accompany the experiment snapshot and runnable package. Model weights are not redistributed here.
