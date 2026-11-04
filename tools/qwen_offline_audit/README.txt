A_Qwen_v2: offline recovery and audit

Completed input: 520 outputs, 40 sources. All 15 synthetic checks passed.
No GPU inference, image generation, downloads, or human ratings were performed
by this recovery program. The input archive is preserved in original/.

48 rejected factual observations (45 outputs, 3 sources) were recovered from
explicit answers already present in saved raw responses. This updated 51
dependent records. Valid observations, pair judgments, and artifact judgments
remain unchanged. One conflict between attempts stays uncertain. Target values
are not available to the syntax canonicalizer. Full schema option lists are
never interpreted as selected answers.

Results: 520 records; 384 fully resolved, 136 with some unresolved fields;
zero remaining schema-error records. Joint: 104 positive, 358 negative,
58 unknown. A resolved failure in one required channel can establish joint=0
even when another channel is unknown. Unknown labels are not replaced by zero.

Important measurement findings:
* Preservation remains unknown in 136 outputs (26.2%); target condition in 1.
* All 520 artifact calls returned false. Only clean-image artifact controls
  were present, so sensitivity to actual severe artifacts is unvalidated.
* edit is satisfaction of the frozen COMPLETE target conditions plus evidence
  of source-to-target change, not an isolated measure of the changed attribute.
* The giant task checks presence, not location in the middle of the lake.
* The Chinese-watercolor task checks watercolor/brushwork, not the Chinese style.
* The ferruginous-sofa rule narrows the material to rusty iron and permits color
  changes even though the target text specifies a black sofa.
* No task rule was changed after seeing method results. Any future rule revision
  needs a distinct protocol and the same treatment of every method.

paired_summary.csv uses complete source intersections and 5,000 paired source
bootstrap resamples. CIs are pointwise and unadjusted; missingness varies by arm.
Report counts, coverage, and the bounds in arm_summary.csv alongside estimates.
These data are exploratory local-VLM judgments, not a validated final ranking.

Reproduce on any computer with Python 3 (standard library only):
python recover.py original/A_Qwen_FactualJudge_v2_results.zip reproduced_results

Files:
results/ratings.csv              repaired per-output labels
results/arm_summary.csv          counts, unknowns, and rate bounds
results/paired_summary.csv       complete-source paired joint comparisons
results/recovery_audit.json      every failed raw response and normalization
results/changed_records.json     before/after labels of all affected records
results/original_summary.json    untouched input summary
results/followup_manifest.json   unresolved channels; not a request to rerun all
audit_examples.jpg              preview-only visual checks of three tasks

No manuscript, GitHub repository, or experiment images were modified.
