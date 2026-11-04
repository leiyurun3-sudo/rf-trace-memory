"""Benchmark-only aggregation, with no model imports or joint-label imputation.

summarize(plan, jobs, records, out) uses jobs as the planned denominator. Missing
records/labels remain unknown. Valid partial labels survive an unresolved record
status; the runner owns the edit/protected/artifact/joint decisions. Extra record
IDs are ignored. Explicit control, repeat and calibration jobs are excluded.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile


CRITERIA = ("edit", "protected", "artifact", "joint")
PRIMARY_ARM = "RELEASE_CORR25"
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 85006


def _auxiliary(job):
    for key in ("audit_repeat", "diagnostic_only", "is_control", "control", "repeat", "calibration"):
        value = job.get(key)
        if value is True or (isinstance(value, str) and value.lower() in ("true", "1")):
            return True
    names = {str(job.get(key, "")).lower() for key in ("arm", "role", "kind")}
    return bool(names & {"source_control", "control", "repeat", "calibration"})


def _mean(values):
    return sum(values) / len(values) if values else None


def _percentile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap(values, namespace):
    # Stable per-arm/contrast streams, independent of input dictionary ordering.
    if len(values) < 2:
        return None
    seed = int.from_bytes(hashlib.sha256(
        json.dumps([BOOTSTRAP_SEED, namespace], ensure_ascii=False).encode("utf-8")
    ).digest()[:16], "big")
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(BOOTSTRAP_RESAMPLES))
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)  # csv writes None as an empty cell, including labels.
    return stream.getvalue()


def _write_atomic(out, documents):
    """Stage every file before replacement; replace summary.json last."""
    out.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for name, text in documents:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                             prefix="." + name + ".", suffix=".tmp",
                                             dir=out, delete=False) as stream:
                path = Path(stream.name)
                staged.append((path, out / name))
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def summarize(plan, jobs, records, out):
    """Save ratings/arm/paired CSVs and summary JSON; return the summary dict.

    Labels must be actual integers 0/1 or None (no bool/float coercion). A missing
    field is unknown. Joint labels are used as supplied, never recomputed here.
    Bounds and resolved rates concern the event value=1; for artifact that means
    artifact presence, rather than artifact-free success. Bootstrap CIs require
    at least two complete sources and are pointwise, without hypothesis tests.
    """
    if not isinstance(records, dict):
        raise ValueError("records must map job IDs to record dictionaries")
    benchmark, excluded = [], []
    seen_ids, seen_seeds = set(), set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Every planned job must be a dictionary")
        if _auxiliary(job):
            excluded.append(job.get("id"))
            continue
        for key in ("id", "case", "arm", "seed"):
            if key not in job or job[key] is None:
                raise ValueError("Planned job missing " + key)
        if any(not isinstance(job[key], str) or not job[key].strip() for key in ("id", "case", "arm")):
            raise ValueError("Job id, case and arm must be nonempty strings")
        if type(job["seed"]) not in (int, str):
            raise ValueError("Seed must be an integer or string")
        jid = job["id"]
        cluster_seed = (job["arm"], job["case"], str(job["seed"]))
        if jid in seen_ids or cluster_seed in seen_seeds:
            raise ValueError("Duplicate planned job or arm/source/seed: " + jid)
        seen_ids.add(jid)
        seen_seeds.add(cluster_seed)
        benchmark.append(job)

    ratings, grouped = [], {}
    for job in benchmark:
        jid = job["id"]
        missing = jid not in records
        record = {} if missing else records[jid]
        if not isinstance(record, dict):
            raise ValueError("Record must be a dictionary: " + jid)
        status = "missing" if missing else record.get("status", "unresolved")
        if status not in ("ok", "unresolved", "error", "missing"):
            raise ValueError("Invalid record status: " + jid)
        row = {key: job[key] for key in ("id", "case", "arm", "seed")}
        row["status"] = status
        for criterion in CRITERIA:
            value = record.get(criterion)
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError(jid + "." + criterion + " must be integer 0/1 or None")
            row[criterion] = value
        row["detail_json"] = json.dumps(record.get("detail"), ensure_ascii=False,
                                        sort_keys=True, allow_nan=False)
        ratings.append(row)
        grouped.setdefault(job["arm"], {}).setdefault(job["case"], []).append(row)

    arms, arm_rows, complete_means = {}, [], {}
    for arm, sources in sorted(grouped.items()):
        rows = [row for cluster in sources.values() for row in cluster]
        complete = {case: _mean([row["joint"] for row in cluster])
                    for case, cluster in sorted(sources.items())
                    if all(row["joint"] is not None for row in cluster)}
        complete_means[arm] = complete
        source_values = list(complete.values())
        source_ci = _bootstrap(source_values, ["joint", arm])
        result = {"planned_outputs": len(rows), "planned_sources": len(sources),
                  "record_status_counts": {status: sum(row["status"] == status for row in rows)
                                           for status in ("ok", "unresolved", "error", "missing")},
                  "joint_source_estimate": {
                      "complete_sources": len(complete),
                      "incomplete_sources": len(sources) - len(complete),
                      "mean_source_seed_average": _mean(source_values),
                      "source_bootstrap_95ci": source_ci}, "criteria": {}}
        for criterion in CRITERIA:
            n = len(rows)
            positive = sum(row[criterion] == 1 for row in rows)
            negative = sum(row[criterion] == 0 for row in rows)
            resolved = positive + negative
            unknown = n - resolved
            stats = {"planned_outputs": n, "positive_outputs": positive,
                     "negative_outputs": negative, "unknown_outputs": unknown,
                     "resolved_outputs": resolved,
                     "resolved_rate": positive / resolved if resolved else None,
                     "resolved_rate_denominator": resolved,
                     "output_rate_lower": positive / n,
                     "output_rate_upper": (positive + unknown) / n}
            result["criteria"][criterion] = stats
            arm_rows.append({"arm": arm, "criterion": criterion, **stats,
                             "planned_sources": len(sources),
                             "complete_joint_sources": len(complete) if criterion == "joint" else None,
                             "joint_source_mean": _mean(source_values) if criterion == "joint" else None,
                             "joint_source_ci95_low": source_ci[0] if criterion == "joint" and source_ci else None,
                             "joint_source_ci95_high": source_ci[1] if criterion == "joint" and source_ci else None})
        arms[arm] = result

    paired = []
    primary = complete_means.get(PRIMARY_ARM)
    if primary is not None:
        for other, comparator in sorted(complete_means.items()):
            if other == PRIMARY_ARM:
                continue
            shared = sorted(primary.keys() & comparator.keys())
            differences = [primary[case] - comparator[case] for case in shared]
            interval = _bootstrap(differences, ["paired_joint", PRIMARY_ARM, other])
            paired.append({"arm": PRIMARY_ARM, "reference": other, "criterion": "joint",
                           "paired_complete_sources": len(shared),
                           "arm_complete_sources": len(primary),
                           "reference_complete_sources": len(comparator),
                           "difference": _mean(differences),
                           "ci95_low": interval[0] if interval else None,
                           "ci95_high": interval[1] if interval else None})

    plan_count = len(plan["jobs"]) if isinstance(plan, dict) and isinstance(plan.get("jobs"), list) else None
    summary = {
        "aggregation_version": "factual-judge-benchmark-aggregate-v1",
        "scope": "Benchmark jobs only; calibration, identity controls and repeat rows are separate.",
        "planned_outputs": len(benchmark), "plan_file_job_count": plan_count,
        "planned_source_count": len({job["case"] for job in benchmark}),
        "excluded_auxiliary_jobs": len(excluded), "excluded_auxiliary_job_ids": excluded,
        "ignored_record_ids": sorted(str(jid) for jid in records if jid not in seen_ids),
        "label_policy": "Runner-supplied 0/1/null only; no joint recomputation. Missing labels remain unknown; valid partial labels survive unresolved status.",
        "rate_policy": "Event value=1. Bounds: positive/planned to (positive+unknown)/planned. resolved_rate: positive/resolved, with its denominator reported. Artifact value=1 means artifact presence.",
        "source_policy": "Joint only. A source is included only when every planned seed in that arm has a resolved joint label. Average seeds within source, then weight sources equally. Paired differences use the intersection of complete sources, CORR25 minus comparator.",
        "inference": {"method": "Percentile source bootstrap; paired bootstrap resamples source differences",
                      "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "rng_stream": "SHA256-derived seed per arm/contrast",
                      "percentiles": "2.5 and 97.5, linear interpolation",
                      "minimum_sources_for_ci": 2, "intervals": "pointwise, unadjusted",
                      "signflip_tests": False, "holm_adjustment": False},
        "primary_arm": PRIMARY_ARM, "primary_arm_present": primary is not None,
        "arms": arms, "paired_joint": paired,
    }
    # Serialization is completed before creating or replacing any output file.
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    rating_fields = ["id", "case", "arm", "seed", "status", *CRITERIA, "detail_json"]
    arm_fields = ["arm", "criterion", "planned_outputs", "positive_outputs", "negative_outputs",
                  "unknown_outputs", "resolved_outputs", "resolved_rate", "resolved_rate_denominator",
                  "output_rate_lower", "output_rate_upper", "planned_sources", "complete_joint_sources",
                  "joint_source_mean", "joint_source_ci95_low", "joint_source_ci95_high"]
    paired_fields = ["arm", "reference", "criterion", "paired_complete_sources", "arm_complete_sources",
                     "reference_complete_sources", "difference", "ci95_low", "ci95_high"]
    _write_atomic(Path(out), [("ratings.csv", _csv_text(ratings, rating_fields)),
                              ("arm_summary.csv", _csv_text(arm_rows, arm_fields)),
                              ("paired_summary.csv", _csv_text(paired, paired_fields)),
                              ("summary.json", summary_json)])
    return summary
