#!/usr/bin/env python3
"""Convert an explicitly supplied official PIE-Bench root; never download data.

Requires Pillow. Evaluation annotations are isolated from method inputs.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

from PIL import Image, ImageOps


CATEGORIES = {
    "0": "random editing", "1": "change object", "2": "add object",
    "3": "delete object", "4": "change attribute content",
    "5": "change attribute pose", "6": "change attribute color",
    "7": "change attribute material", "8": "change background",
    "9": "change style",
}
REPO = "https://github.com/cure-lab/PnPInversion"
LICENSE = "https://docs.google.com/document/d/1K7pjAgGwS8-lsA-D6Me4jAKmYszKpcyydiAEyZuTm2I/edit"
SELECTION = "R81-PIE-heldout-v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(path):
    # RGB content is independent of filename, metadata, and lossless encoding.
    with Image.open(path) as opened:
        im = ImageOps.exif_transpose(opened).convert("RGB")
        identity = hashlib.sha256(
            f"{im.width}x{im.height}:RGB:".encode() + im.tobytes()
        ).hexdigest()
        thumb = im.resize((64, 64), Image.Resampling.LANCZOS).tobytes()
        gray = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
        dhash = 0
        for y in range(8):
            for x in range(8):
                dhash = (dhash << 1) | int(gray[y * 9 + x] > gray[y * 9 + x + 1])
        return {"rgb_sha256": identity, "thumb": thumb, "dhash": dhash,
                "size": [im.width, im.height]}


def same_source(a, b):
    if a["rgb_sha256"] == b["rgb_sha256"]:
        return True
    # Conservative rejection of re-encoded/resized near copies as well.
    if bin(a["dhash"] ^ b["dhash"]).count("1") > 8:
        return False
    return sum(abs(x - y) for x, y in zip(a["thumb"], b["thumb"])) / len(a["thumb"]) <= 12.0


def inside(root, relative):
    rel = PurePosixPath(str(relative).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or ":" in str(rel):
        raise ValueError(f"Unsafe mapping image path: {relative}")
    result = (root / Path(*rel.parts)).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes dataset: {relative}")
    return result


def exclusion_sources(plan_path):
    plan = read_json(plan_path)
    ids = sorted({str(t["case"]) for t in plan["tasks"]})
    if len(ids) != 6:
        raise ValueError(f"Expected the six R80 sources; got {len(ids)}")
    refs = []
    for case in ids:
        path = plan_path.parent / "data" / case / "source.png"
        if not path.is_file():
            raise ValueError(f"R80 source content required for exclusion: {path}")
        refs.append((case, fingerprint(path)))
    return refs


def valid_annotation(item):
    if "mask" not in item or item["mask"] is None:
        return True
    mask = item["mask"]
    # Official format is zero-based flattened row-major (start, length) pairs.
    return (isinstance(mask, list) and len(mask) % 2 == 0
            and all(type(v) is int and v >= 0 for v in mask)
            and all(mask[i] < 512 * 512 for i in range(0, len(mask), 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="Directory containing mapping_file.json and annotation_images/")
    parser.add_argument("--count", type=int, choices=(12, 24), default=12)
    parser.add_argument("--data-origin", default="user-supplied local dataset",
                        help="Provenance of the supplied data; does not fetch anything")
    parser.add_argument("--exclude-plan", type=Path, default=Path(__file__).resolve().parent.parent / "R80_RegionBalance" / "plan.json")
    parser.add_argument("--inventory-only", action="store_true", help="Print real counts without writing data")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    mapping_path = root / "mapping_file.json"
    images_root = root / "annotation_images"
    if not mapping_path.is_file() or not images_root.is_dir():
        parser.error("dataset-root must contain actual mapping_file.json AND annotation_images/; no download performed")
    mapping = read_json(mapping_path)
    if not isinstance(mapping, dict) or not mapping:
        parser.error("Official mapping must be a nonempty JSON object keyed by image ID")
    exclusions = exclusion_sources(args.exclude_plan.resolve())
    counters = Counter(mapping_records=len(mapping), excluded_reference_sources=len(exclusions))
    rejected = {name: [] for name in ("invalid_records", "missing_images", "unreadable_images", "excluded_r80")}
    candidates = []
    for case, item in sorted(mapping.items()):
        if (not re.fullmatch(r"\d{12}", case) or not isinstance(item, dict)
                or not all(isinstance(item.get(k), str) and item[k].strip()
                           for k in ("image_path", "original_prompt", "editing_prompt"))
                or str(item.get("editing_type_id")) not in CATEGORIES
                or not valid_annotation(item)):
            rejected["invalid_records"].append(case)
            continue
        image = inside(images_root, item["image_path"])
        if not image.is_file():
            rejected["missing_images"].append(case)
            continue
        try:
            fp = fingerprint(image)
        except (OSError, ValueError, Image.DecompressionBombError):
            rejected["unreadable_images"].append(case)
            continue
        counters["available_image_mapping_pairs"] += 1
        if case in {c for c, _ in exclusions} or any(same_source(fp, e) for _, e in exclusions):
            rejected["excluded_r80"].append(case)
            continue
        candidates.append({"case": case, "item": item, "image": image, "fp": fp})

    # Stable order independent of filesystem and JSON insertion order. Category
    # round-robin improves coverage without inventing target prompts or labels.
    candidates.sort(key=lambda c: hashlib.sha256((SELECTION + ":" + c["case"]).encode()).hexdigest())
    unique = []
    duplicate_ids = []
    for c in candidates:
        if any(same_source(c["fp"], other["fp"]) for other in unique):
            duplicate_ids.append(c["case"])
        else:
            unique.append(c)
    buckets = {k: [c for c in unique if str(c["item"]["editing_type_id"]) == k] for k in CATEGORIES}
    selected = []
    while len(selected) < args.count and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(selected) < args.count:
                selected.append(bucket.pop(0))
    counts = dict(counters)
    counts.update({k: len(v) for k, v in rejected.items()})
    counts.update(duplicate_source_records=len(duplicate_ids),
                  eligible_distinct_sources=len(unique), requested_sources=args.count,
                  selected_sources=len(selected),
                  selected_with_official_mask=sum(c["item"].get("mask") is not None for c in selected),
                  eligible_by_category=dict(Counter(str(c["item"]["editing_type_id"]) for c in unique)))
    audit = {"dataset_root": str(root), "counts": counts, "rejected_ids": rejected,
             "duplicate_source_ids": duplicate_ids, "selected_ids": [c["case"] for c in selected]}
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.inventory_only:
        return 0
    if len(selected) != args.count:
        print(f"Need {args.count} distinct non-R80 sources with actual official prompts; no outputs written.", file=sys.stderr)
        return 2

    output = Path(__file__).resolve().parent
    data = output / "heldout_data"
    manifest_path = output / "heldout_tasks.json"
    if (data / "sources").exists() or (data / "evaluation_only").exists() or manifest_path.exists():
        parser.error("Prepared sources/evaluation_only or heldout_tasks.json already exists; refusing to overwrite selections")
    # All selection and content checks precede output creation.
    (data / "sources").mkdir(parents=True)
    (data / "evaluation_only").mkdir()
    tasks = []
    for c in selected:
        case, item, fp = c["case"], c["item"], c["fp"]
        extension = c["image"].suffix.lower()
        image_rel = f"heldout_data/sources/{case}{extension}"
        annotation_rel = f"heldout_data/evaluation_only/{case}.json"
        shutil.copyfile(c["image"], output / image_rel)
        # Entire official record preserved ONLY in the evaluation directory,
        # including mask/blended_word/additional supplied independent annotations.
        write_json(output / annotation_rel, {
            "case": case, "usage": "evaluation_only_after_generation",
            "official_record": item, "mask_encoding": "zero_based_start_length_pairs_row_major_512x512",
            "official_evaluator_boundary_policy": "set four border edges to edited=1",
            "independence": "supplied dataset annotation; not a new human annotation or ground-truth edited image",
        })
        if item.get("mask") is not None:
            pixels = bytearray(512 * 512)
            encoded = item["mask"]
            for i in range(0, len(encoded), 2):
                start, length = encoded[i:i+2]
                length = min(length, len(pixels) - start)
                pixels[start:start+length] = b"\xff" * length
            # Match the official evaluation decoder's boundary convention.
            pixels[:512] = b"\xff" * 512
            pixels[-512:] = b"\xff" * 512
            for y in range(512):
                pixels[y*512] = pixels[y*512+511] = 255
            Image.frombytes("L", (512, 512), bytes(pixels)).save(data / "evaluation_only" / f"{case}_mask.png")
        tasks.append({
            "case": case, "key": f"pie_{case}", "source_image": image_rel,
            "image": image_rel, "task": f"pie_{case}", "seeds": [0, 17],
            "source_prompt": item["original_prompt"].replace("[", "").replace("]", ""),
            "target_prompt": item["editing_prompt"].replace("[", "").replace("]", ""),
            "target_category_id": str(item["editing_type_id"]),
            "target_category": CATEGORIES[str(item["editing_type_id"])],
            "edit_rubric": item.get("editing_instruction", item["editing_prompt"]),
            "source_rgb_sha256": fp["rgb_sha256"], "source_size": fp["size"],
        })
    manifest = {
        "schema_version": 1, "dataset": "PIE-Bench", "source_repository": REPO,
        "data_origin": args.data_origin,
        "annotation_license": "CC-BY-NC-SA-4.0", "data_license_url": LICENSE,
        "image_rights": "retained by original authors/copyright holders",
        "selection_rule": SELECTION + "; SHA256 order; category round-robin; one task per source",
        "content_exclusion": "RGB identity OR (64-bit dHash distance<=8 AND 64x64 RGB MAE<=12); also reject R80 IDs",
        "excluded_source_ids": [case for case, _ in exclusions],
        "excluded_source_rgb_sha256": {case: fp["rgb_sha256"] for case, fp in exclusions},
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "seeds": [0, 17], "source_count": len(tasks),
        "source_seed_pairs": len(tasks) * 2,
        "evaluation_annotations": {c["case"]: f"heldout_data/evaluation_only/{c['case']}.json" for c in selected},
        "evaluation_masks": {c["case"]: f"heldout_data/evaluation_only/{c['case']}_mask.png"
                             for c in selected if c["item"].get("mask") is not None},
        "annotation_policy": {
            "method_inputs": "tasks contains only images, prompts, categories and source identity metadata",
            "evaluation_only_directory": "heldout_data/evaluation_only",
            "method_mask": "Generate independently from source image / allowed prompts. Never read official mask, blended_word, or evaluation annotations for method masks, tuning or selection.",
            "enforcement": "Manifest separation; runtime consumer must enforce access. This preparer does not change the engine.",
        },
        "counts": counts, "tasks": tasks,
    }
    write_json(manifest_path, manifest)
    # Adapter for the parent's CSV preparer. No object_mask/pattern_roi is
    # provided: official labels must not become the method's routing mask.
    fields = ["case", "image", "source_prompt", "target_prompt", "edit_rubric",
              "target_category_id", "target_category", "seeds",
              "independent_eval_annotation", "independent_eval_mask"]
    with (data / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            case = task["case"]
            row = {key: task[key] for key in fields if key in task}
            row.update(image=task["image"].removeprefix("heldout_data/"), seeds="0,17",
                       independent_eval_annotation=f"evaluation_only/{case}.json",
                       independent_eval_mask=f"evaluation_only/{case}_mask.png"
                       if case in manifest["evaluation_masks"] else "")
            writer.writerow(row)
    print(f"Prepared exactly {len(tasks)} distinct sources, {len(tasks)*2} source-seed pairs: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        sys.exit(2)
