"""24 local VLM calls: factual observation first, fixed task rules afterwards."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import time
import zipfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
HERE = Path(__file__).resolve().parent
VERSION = "factual-calibration-v2"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decode_json(raw):
    """Parse saved text independently; reject duplicate keys and non-JSON values."""
    if not isinstance(raw, str):
        raise ValueError("Response must be text")
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonstandard JSON constant: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def parse(raw):
    try:
        return decode_json(raw)
    except (ValueError, TypeError):
        return None


FIELDS = {
    "shapes": {
        "circle_present": "boolean", "circle_color": ["blue", "red", "other", "absent", "uncertain"],
        "square_present": "boolean", "square_color": ["green", "red", "other", "absent", "uncertain"],
        "triangle_present": "boolean",
    },
    "wall": {
        "wall_color": ["white", "red", "pink", "other", "uncertain"],
        "floor_color": ["dark", "red", "other", "uncertain"], "hanging_plant_present": "boolean",
    },
    "moon": {"moon_present": "boolean", "tree_present": "boolean"},
    "kiwi": {"kiwi_present": "boolean", "smoothie_glass_present": "boolean"},
}


def prompt(kind):
    subjects = {
        "shapes": "Inspect the circle, square, and triangle, if visible, and their colors.",
        "wall": "Inspect the wall, the floor, and the hanging plant separately. Distinguish wall color from floor color.",
        "moon": "Inspect whether a moon and a tree are visible, including small bright objects in the sky.",
        "kiwi": "Inspect whether any kiwi fruit or slice, and a smoothie glass, are visible.",
    }
    return (
        "Describe factual visual attributes of this ONE image. "
        "You are not given an editing goal. Do not infer a before/after transformation. "
        + subjects[kind]
        + " Return one JSON object containing only these factual fields using the listed allowed values: "
        + json.dumps(FIELDS[kind], ensure_ascii=False)
        + ". Boolean fields accept true, false, or null ONLY if visually undecidable. "
        "Color fields use absent only when that shape is absent; use uncertain if undecidable. "
        "If shape presence is null, its color must be uncertain. "
        "No evidence or narrative field is requested. Report observations without assuming a task."
    )


def validate(value, kind):
    """Return factual fields only; schema validity does not establish accuracy."""
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    for key, allowed in FIELDS[kind].items():
        if key not in value:
            raise ValueError("Missing field: " + key)
        x = value[key]
        if allowed == "boolean":
            if x is not None and type(x) is not bool:
                raise ValueError("Expected bool/null: " + key)
        elif not isinstance(x, str) or x not in allowed:
            raise ValueError("Invalid enum: " + key)
    if kind == "shapes":
        for name in ("circle", "square"):
            present, color = value[name + "_present"], value[name + "_color"]
            if present is False and color != "absent":
                raise ValueError("Absent shape must have absent color: " + name)
            if present is True and color == "absent":
                raise ValueError("Present shape cannot have absent color: " + name)
            if present is None and color != "uncertain":
                raise ValueError("Unknown shape presence requires uncertain color: " + name)
    return {key: value[key] for key in FIELDS[kind]}


def narrative_evidence(value):
    """Record optional narrative coverage without generating or requiring prose."""
    present = isinstance(value, dict) and "evidence" in value
    return {"present": present,
            "valid": (isinstance(value["evidence"], str) and bool(value["evidence"].strip()))
            if present else None}


def known_fact_expected(expected):
    """Exact-observation accuracy uses only fully known, nonempty expectations."""
    return (isinstance(expected, dict) and bool(expected)
            and all(value is not None and value != "uncertain" for value in expected.values()))


def build(root, out):
    from PIL import Image, ImageDraw
    tasks, groups = [], []

    def add(path, kind, role, group, expected=None):
        task = {
            "id": "obs_%03d" % (len(tasks) + 1), "image": str(path), "kind": kind,
            "role": role, "group": group, "prompt": prompt(kind), "expected": expected,
        }
        tasks.append(task)
        return task["id"]

    def scene(name, circle, square, triangle=False):
        im = Image.new("RGB", (448, 448), "white")
        d = ImageDraw.Draw(im)
        colors = {"red": (220, 20, 30), "blue": (20, 40, 220), "green": (20, 150, 40)}
        if circle:
            d.ellipse((35, 130, 185, 280), fill=colors[circle])
        if square:
            d.rectangle((270, 130, 410, 270), fill=colors[square])
        if triangle:
            d.polygon([(110, 120), (30, 280), (190, 280)], fill=colors["blue"])
        p = out / "stimuli" / (name + ".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p)
        state = {
            "circle_present": circle is not None, "circle_color": circle or "absent",
            "square_present": square is not None, "square_color": square or "absent",
            "triangle_present": triangle,
        }
        return p, state

    specs = [
        ("color", ("blue", "green", False), ("red", "green", False), ("blue", "red", False)),
        ("remove", ("blue", "green", False), (None, "green", False), ("blue", None, False)),
        ("add", (None, "green", False), ("blue", "green", False), (None, "green", True)),
    ]
    for family, src_spec, positive_spec, wrong_spec in specs:
        ids = {}
        src, expected = scene(family + "_source", *src_spec)
        ids["source"] = add(src, "shapes", "source", family, expected)
        # Independent call: no cached or hard-coded identity score.
        ids["identity"] = add(src, "shapes", "identity", family, expected)
        for role, spec in [("positive", positive_spec), ("wrong_object", wrong_spec)]:
            p, state = scene(family + "_" + role, *spec)
            ids[role] = add(p, "shapes", role, family, state)
        groups.append({"name": family, "kind": "synthetic", "ids": ids})
    cases = read(HERE / "cases.json")
    for case, kind in zip(cases, ("wall", "moon", "kiwi")):
        src = root / case["source"]
        if not src.is_file() or digest(src) != case["source_sha256"]:
            raise ValueError("Native source missing or changed: " + str(src))
        ids = {"source": add(src, kind, "source", kind), "identity": add(src, kind, "identity", kind)}
        for i, output in enumerate(case["real_outputs"], 1):
            p = root / output["path"]
            if not p.is_file() or digest(p) != output["sha256"]:
                raise ValueError("Native output missing or changed: " + str(p))
            ids["candidate_" + str(i)] = add(p, kind, "candidate_" + str(i), kind)
        groups.append({"name": kind, "kind": "real", "ids": ids})
    return tasks, groups


def edit_rule(family, source, candidate):
    if source is None or candidate is None:
        return None
    if family in ("color", "wall"):
        key = "circle_color" if family == "color" else "wall_color"
        if source[key] in ("uncertain", "absent", "red") or candidate[key] in ("uncertain", "absent"):
            return None
        return int(candidate[key] == "red")
    key = {"remove": "circle_present", "add": "circle_present",
           "moon": "moon_present", "kiwi": "kiwi_present"}[family]
    start, end = source[key], candidate[key]
    if start is None or end is None:
        return None
    expected_start = family != "add"
    if start is not expected_start:
        return None
    return int(end is (not expected_start))


def summarize(tasks, groups, records):
    byid = {r["id"]: r for r in records}
    states = {i: r["parsed"] if r["status"] == "ok" else None for i, r in byid.items()}
    comparisons = []
    for group in groups:
        source = states.get(group["ids"]["source"])
        for role, tid in group["ids"].items():
            if role == "source":
                continue
            value = edit_rule(group["name"], source, states.get(tid))
            expected = int(role == "positive") if group["kind"] == "synthetic" else (0 if role == "identity" else None)
            comparisons.append({
                "group": group["name"], "role": role, "edit_rule": value,
                "source_id": group["ids"]["source"], "candidate_id": tid,
                "known_expected": expected, "known_correct": value == expected if expected is not None else None,
                "status": "unresolved" if value is None else "resolved",
            })
    known = [r for r in comparisons if r["known_expected"] is not None]
    attribute_checks = []
    for task in tasks:
        if known_fact_expected(task.get("expected")):
            state = states.get(task["id"])
            attribute_checks.append(state is not None and all(state.get(k) == v for k, v in task["expected"].items()))
    passed = (
        len(records) == len(tasks) and all(r["status"] == "ok" for r in records)
        and all(attribute_checks) and all(r["known_correct"] for r in known)
    )
    return {
        "protocol": VERSION, "completed": len(records), "planned": len(tasks),
        "schema_ok": sum(r["status"] == "ok" for r in records),
        "narrative_evidence_present": sum(r.get("narrative_evidence", {}).get("present") is True for r in records),
        "narrative_evidence_valid": sum(r.get("narrative_evidence", {}).get("valid") is True for r in records),
        "known_attribute_correct": sum(attribute_checks), "known_attribute_total": len(attribute_checks),
        "known_edit_correct": sum(r["known_correct"] for r in known), "known_edit_total": len(known),
        "gate": "small_calibration_passed" if passed else "needs_inspection",
        "comparisons": comparisons,
        "scope": "Calibration only. Real candidates have no assumed success label. "
                 "No joint-success or method ranking; no old scores imported. "
                 "Passing these controls does not establish reliability on 40 benchmark tasks.",
    }


def package(out):
    target = out.with_suffix(".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, Path(out.name) / p.relative_to(out))
    print("Result ZIP:", target, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        tasks, groups = build(a.root, a.out)
        save(a.out / "plan_private.json", {"tasks": tasks, "groups": groups})
        print("24 factual-observation calls; no downloads, no full evaluation restart.", flush=True)
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU fallback disabled")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(a.model), local_files_only=True, device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        model.eval()
        model.generation_config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            setattr(model.generation_config, name, None)
        processor = AutoProcessor.from_pretrained(str(a.model), local_files_only=True,
                                                  use_fast=True, max_pixels=262144)
        save(a.out / "runtime.json", {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "model": str(a.model), "processor": type(processor.image_processor).__name__,
            "protocol": VERSION, "model_revision": read(a.model / "judge_model_revision.json")
            if (a.model / "judge_model_revision.json").is_file() else None})
        started = time.monotonic()
        for i, task in enumerate(tasks, 1):
            record = {"id": task["id"], "kind": task["kind"], "prompt": task["prompt"],
                      "image_sha256": digest(task["image"])}
            try:
                with Image.open(task["image"]) as im:
                    im = im.convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": task["prompt"]}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[im], padding=True, return_tensors="pt")
                grid, pixels = inputs["image_grid_thw"], inputs["pixel_values"]
                merge = int(processor.image_processor.merge_size)
                expected = int((grid.prod(dim=1) // (merge * merge)).sum())
                actual = int((inputs["input_ids"] == model.config.image_token_id).sum())
                record["input_audit"] = {"grid": grid.tolist(), "expected_image_tokens": expected,
                                         "actual_image_tokens": actual, "finite": bool(torch.isfinite(pixels).all())}
                if len(grid) != 1 or expected != actual or not record["input_audit"]["finite"]:
                    raise ValueError("Vision input check failed")
                inputs = inputs.to(model.device)
                tic = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=384)
                raw = processor.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True)[0]
                record.update(raw_response=raw, parsed=parse(raw), seconds=time.monotonic() - tic)
                record["narrative_evidence"] = narrative_evidence(record["parsed"])
                record["facts"] = validate(record["parsed"], task["kind"])
                record["status"] = "ok"
            except Exception as exc:
                record.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
            records.append(record)
            save(a.out / "responses" / (task["id"] + ".json"), record)
            save(a.out / "progress.json", {"completed": i, "planned": len(tasks),
                                          "elapsed_seconds": time.monotonic() - started})
            print(i, "/", len(tasks), record["status"], record.get("error", ""), flush=True)
        summary = summarize(tasks, groups, records)
        save(a.out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        save(a.out / "fatal_error.json", {"error": type(exc).__name__ + ": " + str(exc)})
        raise
    finally:
        package(a.out)


if __name__ == "__main__":
    main()
