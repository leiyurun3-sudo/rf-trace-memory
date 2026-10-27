"""Import manual review JSON on CPU: python review.py import R71_review.json."""
import argparse
from pathlib import Path
import time

import reporting as report

ROOT = Path(__file__).resolve().parent


def import_reviews(path, root=ROOT):
    root = Path(root)
    plan, _ = report.load_plan(root)
    provenance = report.fingerprint(root)
    report.guard_existing(root, provenance)
    data = report.read(path)
    if data.get('schema') != 'r71-human-review-v1' or data.get('fingerprint') != provenance['fingerprint']:
        raise ValueError('Review schema/fingerprint mismatch; export from the current review page.')
    jobs = {j['id']: j for j in plan['jobs']}
    seen = set()
    pending = []
    out, _, _ = report.locations(root)
    # Validate the entire import before writing any judgments.
    if not isinstance(data.get('reviews'), list):
        raise ValueError('reviews must be a list')
    for item in data['reviews']:
        jobid = item.get('jobid')
        if jobid not in jobs or jobid in seen:
            raise ValueError('Unknown or duplicate jobid: ' + str(jobid))
        seen.add(jobid)
        record = report.completed(root, jobs[jobid], provenance['fingerprint'])
        if not record or item.get('preview_sha256') != record['preview_sha256']:
            raise ValueError('Preview SHA mismatch or incomplete job: ' + jobid)
        ratings = {}
        for name in ('edit_ok', 'subject_ok', 'artifact'):
            if name not in item or (item[name] is not None and (type(item[name]) is not int or item[name] not in (0, 1))):
                raise ValueError('Rating must be null, 0 or 1: ' + jobid + '/' + name)
            ratings[name] = item[name]
        # Blank exports never erase previously supplied human judgments.
        old_path = out / jobid / 'judge.json'
        old = report.read(old_path) if old_path.exists() else {}
        if old and (old.get('preview_sha256') != record['preview_sha256'] or old.get('fingerprint') != provenance['fingerprint'] or old.get('jobid') != jobid):
            raise ValueError('Stale existing judgment: ' + jobid)
        for name in ratings:
            if ratings[name] is None:
                ratings[name] = old.get(name)
        if all(v is None for v in ratings.values()):
            continue
        pending.append((old_path, dict(jobid=jobid, preview_sha256=record['preview_sha256'],
            fingerprint=provenance['fingerprint'], **ratings,
            judge_status='done' if all(v is not None for v in ratings.values()) else 'pending',
            evaluation_method='human', imported_at=time.time(), import_sha256=report.sha(path))))
    for destination, judgment in pending:
        report.save(destination, judgment)
    report.package(root, plan, provenance)
    print(f'Imported {len(pending)} human judgments; reconstruction excluded from edit success rate.', flush=True)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('import')
    command.add_argument('json_file', type=Path)
    args = parser.parse_args()
    # Serialize with runner/--pack, without importing torch or loading a model.
    import contextlib
    with contextlib.ExitStack() as stack:
        for name in ('.ab_queue.lock', '.mint_runner.lock'):
            stack.enter_context(report.file_lock(ROOT.parent / name))
        import_reviews(args.json_file)


if __name__ == '__main__':
    main()
