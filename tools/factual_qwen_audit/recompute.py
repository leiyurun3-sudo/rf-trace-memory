"""Recompute benchmark aggregation from the archived corrected records."""
import argparse
import json
from pathlib import Path
from aggregate import summarize

base = Path(__file__).resolve().parents[2] / "results" / "QwenFactualV2"
parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path, help="new output directory")
args = parser.parse_args()
if args.out.exists():
    parser.error("output directory already exists")
jobs = json.loads((base / "jobs.json").read_text(encoding="utf-8"))
records = json.loads((base / "records.json").read_text(encoding="utf-8"))
summarize({"jobs": jobs}, jobs, records, args.out)
