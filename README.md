# TraceMemory

**Spatial source correction and memory for training-free rectified-flow image editing**

Updated **2 November 2026** with completed R84/R85 generation, automatic evaluation, and source-paired analysis.

[Manuscript PDF](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Final evaluation](paper/final_evaluation.tex) · [Run instructions](docs/REPRODUCING_FINAL.md) · [Download packages](https://github.com/leiyurun3-sudo/rf-trace-memory/releases/tag/evaluated-r84-r85-20261102)

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

| Method | CLIP target ↑ | Directional CLIP ↑ | DINO ↑ | LPIPS full ↓ | LPIPS preserved ↓ | NFE |
|---|---:|---:|---:|---:|---:|---:|
| **TraceMemory (CORR25)** | 0.2629 | 0.0863 | 0.5904 | 0.5419 | 0.5142 | 97 |
| BOTH25, secondary | 0.2733 | 0.1069 | 0.5579 | 0.5620 | 0.5269 | 97 |
| Same-state energy control | 0.2622 | 0.0881 | 0.5856 | 0.5483 | 0.5226 | 97 |
| Uniform correction, U25 | 0.2632 | 0.0908 | 0.6111 | 0.4971 | 0.4743 | 91 |
| FlowEdit N26 | 0.2712 | 0.1177 | 0.5737 | 0.4437 | 0.4281 | 52 |
| RF-Inversion ETA03 | 0.2694 | 0.1065 | 0.4670 | 0.5431 | 0.5322 | 56 |
| RF-Edit I08 adapter | 0.2453 | 0.0383 | 0.7875 | 0.2778 | 0.2688 | 100 |

![Final metrics and source-bootstrap intervals](paper/figures/final_metrics.png)

### Findings

1. **Spatial allocation improves perceptual preservation against the same-state energy control.** CORR25 reduces preservation LPIPS by **0.0084**, with a 95% source-bootstrap interval **[−0.0143, −0.0022]** and Holm-adjusted **p = 0.0238**. Full-image LPIPS decreases by 0.0064. The directional CLIP difference is −0.0018 with an interval spanning zero.
2. **Memory attenuation increases semantic editing and source departure together.** BOTH25 increases directional CLIP by **0.0207** over CORR25, while preservation LPIPS increases by **0.0127** and DINO decreases by 0.0325.
3. **FlowEdit N26 achieves stronger directional alignment and lower perceptual preservation error at lower cost.** Its directional CLIP exceeds CORR25 by 0.0315, and its preservation LPIPS is lower by 0.0861, using 52 versus 97 NFE. Uniform U25 also has lower preservation LPIPS than CORR25. CORR25 improves DINO over ETA03 by 0.1234.
4. **The development dose study shows a preservation response.** Increasing the direct correction coefficient from 0.15 to 0.25 to 0.35 changes preservation LPIPS from **0.5544 → 0.4983 → 0.4561**. Semantic-score differences have broad intervals on this ten-source development set.

These results establish a modest controlled localization benefit and quantify the memory trade-off. The independent comparison identifies FlowEdit and uniform correction as strong reference configurations for further algorithm development.

### Statistical and evaluation protocol

Seeds are averaged within each source before inference. Reported paired intervals use source-bootstrap resampling; tests use source sign flips and Holm correction across six planned comparisons within each metric. The full numeric records retain the comparison family, sample counts, and unrounded estimates. The manuscript appendix includes broader-family sensitivity analysis.

LPIPS preservation is a spatial-feature-map average weighted by the official edit-mask complement, with receptive-field spill across its boundary. Empty complements are excluded from that metric and counted explicitly. CLIP-T, directional CLIP, CLIP-I, DINOv2, LPIPS, PSNR, and SSIM exports are included. **Human edit/subject/pattern/artifact judgments remain unfilled, so human edit and joint-success rates are unreported.**

### Qualitative examples

![Source, TraceMemory, energy control, and FlowEdit on four seed-0 examples](paper/figures/final_qualitative.jpg)

Columns: source, CORR25, energy control, FlowEdit N26. Rows: grass-to-water, wall-to-red, parrots-kissing, and camera-to-phone. These examples show attribute changes alongside relational failures and identity changes. Population-level comparisons use the complete declared source set.

## Files and reproduction

| Location | Contents |
|---|---|
| [paper/main.pdf](paper/main.pdf) | Updated extended manuscript and appendices |
| [paper/main.tex](paper/main.tex) | Manuscript source incorporating final evaluated results |
| [results/R85](results/R85) | Independent comparison: evaluation CSV, metrics, summaries, source-paired evidence |
| [results/R84](results/R84) | Development controls and dose response |
| [experiments/R84_R85_FinalNight](experiments/R84_R85_FinalNight) | Frozen code, plans, selectors, launchers, evaluators, and statistics |
| [docs/REPRODUCING_FINAL.md](docs/REPRODUCING_FINAL.md) | Environment, generation, evaluation-only restart, and package layout |
| [Release assets](https://github.com/leiyurun3-sudo/rf-trace-memory/releases/tag/evaluated-r84-r85-20261102) | Intact runnable package, compact evaluated results, and manuscript PDF |
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
