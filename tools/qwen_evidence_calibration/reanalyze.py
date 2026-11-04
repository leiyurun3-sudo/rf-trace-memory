"""Standard-library reanalysis: python -B reanalyze.py --results ARCHIVE_DIR.

Read raw responses, not archived parsed labels/status, and write only NEW files
under audit/. No inference, model imports, source rewrites, or expected filling.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from calibrate import (FIELDS, VERSION, decode_json, known_fact_expected,
                       narrative_evidence, read, summarize, validate)


def analyze(root):
    plan = read(root / "plan_private.json")
    tasks, groups = plan["tasks"], plan["groups"]
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate observation IDs in plan")
    records, observations, traces = [], [], []
    original_statuses, original_errors = Counter(), Counter()
    original_byid = {}
    for task in tasks:
        tid = task["id"]
        if not isinstance(tid, str) or not tid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tid):
            raise ValueError("Unsafe observation ID")
        path = root / "responses" / (tid + ".json")
        row = {"id": tid, "group": task["group"], "role": task["role"],
               "kind": task["kind"], "original_status": "missing",
               "json_syntax_ok": False, "factual_schema_ok": False,
               "narrative_evidence_present": False, "narrative_evidence_valid": None,
               "fully_known_expected": known_fact_expected(task.get("expected")),
               "known_facts_exact_match": None, "facts_json": "", "uncertain_fields": "", "error": ""}
        record = {"id": tid, "status": "failed", "parsed": None}
        trace = {"id": tid, "record_file": "responses/" + tid + ".json"}
        try:
            saved = read(path)
            if saved.get("id") != tid or saved.get("kind") != task["kind"]:
                raise ValueError("Record ID/kind differs from plan")
            original_byid[tid] = saved
            row["original_status"] = saved.get("status")
            original_statuses[str(saved.get("status"))] += 1
            if saved.get("error"):
                original_errors[saved["error"]] += 1
            trace["raw_response"] = saved.get("raw_response")
            parsed = decode_json(saved.get("raw_response"))
            row["json_syntax_ok"] = True
            narrative = narrative_evidence(parsed)
            row["narrative_evidence_present"] = narrative["present"]
            row["narrative_evidence_valid"] = narrative["valid"]
            record["narrative_evidence"] = narrative
            facts = validate(parsed, task["kind"])
            record.update(status="ok", parsed=facts)
            row["factual_schema_ok"] = True
            row["facts_json"] = json.dumps(facts, ensure_ascii=False, sort_keys=True)
            row["uncertain_fields"] = json.dumps([key for key, value in facts.items()
                                                    if value is None or value == "uncertain"])
            if row["fully_known_expected"]:
                expected = task["expected"]
                # Typed matching prevents numeric 0/1 from becoming boolean truth.
                row["known_facts_exact_match"] = all(key in facts and type(facts[key]) is type(value)
                                                     and facts[key] == value for key, value in expected.items())
            trace["facts"] = facts
        except (ValueError, TypeError, OSError) as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
        records.append(record)
        observations.append(row)
        traces.append(trace)

    summary = summarize(tasks, groups, records)
    # Attribute accuracy independently recomputed above from validated raw facts.
    known = [row for row in observations if row["fully_known_expected"]]
    summary["known_attribute_correct"] = sum(row["known_facts_exact_match"] is True for row in known)
    summary["known_attribute_total"] = len(known)
    comparisons = summary.pop("comparisons")
    for comparison in comparisons:
        source = original_byid.get(comparison["source_id"], {})
        candidate = original_byid.get(comparison["candidate_id"], {})
        a, b = source.get("image_sha256"), candidate.get("image_sha256")
        comparison["same_image_sha256"] = a == b if isinstance(a, str) and isinstance(b, str) else None
        if comparison["role"] == "identity" and comparison["same_image_sha256"] is not True:
            comparison["known_expected"] = None
            comparison["known_correct"] = None
    known_comparisons = [row for row in comparisons if row["known_expected"] is not None]
    summary["known_edit_correct"] = sum(row["known_correct"] is True for row in known_comparisons)
    summary["known_edit_total"] = len(known_comparisons)
    summary["gate"] = ("small_calibration_passed" if len(records) == len(tasks)
                       and all(row["factual_schema_ok"] for row in observations)
                       and known and all(row["known_facts_exact_match"] is True for row in known)
                       and known_comparisons and all(row["known_correct"] is True for row in known_comparisons)
                       else "needs_inspection")

    def controls(rows):
        return {"total": len(rows), "resolved": sum(row["edit_rule"] is not None for row in rows),
                "known_expected": sum(row["known_expected"] is not None for row in rows),
                "known_correct": sum(row["known_correct"] is True for row in rows)}
    identity = [row for row in comparisons if row["role"] == "identity"]
    synthetic_names = {g["name"] for g in groups if g["kind"] == "synthetic"}
    synthetic = [row for row in comparisons if row["group"] in synthetic_names]
    originals = read(root / "summary.json")
    audit = {
        "audit_version": "factual-raw-reanalysis-v2", "factual_protocol": VERSION,
        "results_directory": str(root), "original_summary": originals,
        "original_record_status_counts": dict(original_statuses),
        "original_error_counts": dict(original_errors),
        "summary": summary,
        "synthetic_rule_controls": controls(synthetic),
        "identity_controls": controls(identity),
        "unknown_expected_comparisons": sum(row["known_expected"] is None for row in comparisons),
        "real_candidate_accuracy": None,
        "narrative_policy": "Optional and measured separately; missing prose is never invented or treated as a factual schema failure.",
        "scope": "Independent parsing of archived raw replies; expected labels used only after factual validation, never to populate observations. "
                 "Schema validity is not visual accuracy. Only the small known controls have accuracy labels; "
                 "real candidates need visual review. No method ranking or 40-source validation.",
        "observations": observations, "raw_response_traces": traces,
        "comparisons": comparisons,
    }
    return audit, observations, comparisons


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    audit, observations, comparisons = analyze(root)
    destination = root / "audit"
    destination.mkdir(exist_ok=False)
    with (destination / "factual_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    write_csv(destination / "factual_observations.csv", observations)
    write_csv(destination / "factual_comparisons.csv", comparisons)
    print(json.dumps({"original_schema_ok": audit["original_summary"].get("schema_ok"),
                      "original_record_status_counts": audit["original_record_status_counts"],
                      "factual_summary": audit["summary"],
                      "synthetic_rule_controls": audit["synthetic_rule_controls"],
                      "identity_controls": audit["identity_controls"],
                      "unknown_expected_comparisons": audit["unknown_expected_comparisons"]},
                     ensure_ascii=False, indent=2))
    print("Wrote new audit files:", destination)


if __name__ == "__main__":
    main()
