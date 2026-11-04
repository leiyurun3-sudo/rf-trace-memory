# Claims, contrasts and decision rules

This map uses the existing main.tex draft and R78/R79 reported results as development evidence. R80 outcomes are pending. R81 provides a fixed independent-source pilot and targeted development controls.

| Research claim | Existing observation | R81 evidence |
|---|---|---|
| Spatial correction allocation improves preservation during a successful edit | R78 routed NCC exceeds full-image uniform NCC; shifted support loses correlation | AUTO_DETAIL vs U25_KV on 24 new sources: joint success, edit success, requested-change-aware preservation and source-cluster intervals |
| Useful routing can be obtained without evaluation annotations | R79 ONLINE25 is manual ROI with online normalization | AUTO_DETAIL reads only source pixels; PIE edit masks are evaluation-only. Report category-specific success and failures |
| Budget normalization is more useful than direct local scaling | R79 changes both NCC and correction impulse | BUDGET15/25/35 dose-response with R80 ONLINE25 at 0.25; report actual RMS/gain and quality jointly |
| Native text contribution and image-attention mass matter | Nearby methods already use source features and attention adaptation | PLAIN_KV vs R80 ONLINE25: identical route, budget, timing and prompts, changed readout |
| Source memory improves the regional correction method | R79 LOCAL on/off changes correlation | R80 ONLINE_OFF supplies the matched online off-control; R81 KV_ONLY supplies correction-off. GATE050 and EARLY3 measure write strength and cost |
| Routing tolerates imprecise region support | Center/shift controls already exist | Erode/dilate route support while keeping evaluation regions fixed; report gains because normalization changes local dose |
| The combined method advances over external approaches | Previous FlowEdit working points illustrate a trade-off | Fixed FlowEdit N26 and pinned RF-Edit on all 24 new sources, shared source/prompt/resolution, native solver settings, total compute |

## Outcomes

Primary outcome: `edit_ok AND subject_ok AND pattern_ok AND NOT severe_artifact`. Every planned output is in the denominator. Requested object/texture changes are evaluated against the instruction: preserve only details that should remain. A criterion is satisfied when no additional unintended damage is present; reviewers should not penalize a requested pattern replacement as preservation failure.

Use two independent blinded reviews and report agreement before adjudication. Review interfaces and templates are generated with outputs. Model-based metrics assist review and give reproducible comparisons; they do not create human labels. Report failed, pending and fully judged counts separately.

Secondary outcomes: target/directional CLIP, CLIP-I, DINO, LPIPS when local weights are provided, and PSNR/SSIM in unchanged regions. Native PNG is the metric input. Whole-image NCC on the new set is an auxiliary correlation score, not exact pattern identity. Dev pattern ROIs and heldout whole-image ROIs are separate scopes.

Report all tasks and categories. Treat the dog edits as one source cluster. A deterministic external method's duplicate seed slots contribute one source-level observation after aggregation. Source-level paired bootstrap intervals and sign-flip tests summarize differences; Holm adjustment covers reported arm pairs within each cohort and metric. Report effect sizes alongside intervals. The pilot's small category counts support exploration; freeze a configuration before the subsequent independent larger test.

## Selection and paper structure

Select using joint success, then edit success, preservation, artifacts and measured cost. Compare preservation at comparable semantic editing performance using operating points selected on development data. Do not equalize success by dropping unfavorable samples. MAE is a source-deviation measure.

The candidate paper mechanism is regional allocation of an explicit correction budget, with separately controlled source-memory writes. Generic masking, residual correction and source KV are established by the primary literature. The contribution to establish is a repeatable editing/preservation advantage of this allocation and an implementation usable without the evaluation annotations.

The draft can describe equations, reconstruction identity, development results, RMS bookkeeping, computational cost and these hypotheses now. Final superiority statements, final configuration and generalization results come from the new comparison and semantic review. Existing manuscript formatting is IEEEtran; old comments describing SPIE are historical. No manuscript edits are included in this package.

After this pilot: freeze one global setting, add the closest remaining comparator DirectEdit or ReFlex, expand a disjoint public benchmark, and complete a blinded preference study. These steps depend on measured pilot performance and are not replaced by an arbitrary NCC threshold.
