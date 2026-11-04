"""CPU evidence tables. No generation or inferred human ratings.
python evidence_stats.py --results /path/to/results --baseline U25_KV
Use native-result evaluation.csv if available; metrics remain reference diagnostics.
"""
import argparse, csv, itertools, json, math, random, statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    if not path.exists(): return []
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))


def number(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError): return None


def write_csv(path, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys or ["status"]); w.writeheader(); w.writerows(rows)


def quantile(xs,p):
    xs=sorted(xs); q=(len(xs)-1)*p; i=int(q)
    return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(q-i)


def inference(values, rng, repeats):
    # One value per source: average all its tasks/seeds before inference.
    n=len(values); mean=statistics.mean(values)
    if n<2: return mean,None,None,None
    draws=[statistics.mean(rng.choices(values,k=n)) for _ in range(repeats)]
    obs=abs(mean)
    if n<=16:
        dist=[abs(sum(v*s for v,s in zip(values,signs))/n) for signs in itertools.product((-1,1),repeat=n)]
        pv=sum(x>=obs-1e-12 for x in dist)/len(dist)
    else:
        hits=sum(abs(sum(v*rng.choice((-1,1)) for v in values)/n)>=obs-1e-12 for _ in range(repeats))
        pv=(hits+1)/(repeats+1)
    return mean,quantile(draws,.025),quantile(draws,.975),pv


def key(r):
    return (r.get("case"),r.get("task"),str(r.get("seed")))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--baseline",default="U25_KV")
    p.add_argument("--replicates",type=int,default=10000)
    args=p.parse_args(); root=args.results; rng=random.Random(20261029)
    rows=read_csv(root/"metrics.csv")
    extra={r["id"]:r for r in read_csv(root/"evaluation.csv")}
    for r in rows:
        if r["id"] in extra:
            # Preserve newest imported judgments in metrics.csv.
            r.update({k:v for k,v in extra[r["id"]].items() if k.startswith(("clip_","dino_","lpips_","pixel_","eval_"))})
        failed=r.get("status") in ("failed","error")
        flags=[number(r.get(k)) for k in ("edit_ok","subject_ok","pattern_ok","artifact")]
        r["joint_success"]=0 if failed else (int(flags[:3]==[1,1,1] and flags[3]==0) if all(v in (0,1) for v in flags) else None)
        if failed: r["edit_ok"]=0
    out=root/"evidence";out.mkdir(exist_ok=True)
    eligible=[r for r in rows if str(r.get("diagnostic_only",False)).lower() not in ("true","1")]
    byarm=defaultdict(list)
    for r in eligible: byarm[r["arm"]].append(r)
    summaries=[]
    for arm,rr in byarm.items():
        n=len(rr); joint=[number(r.get("joint_success")) for r in rr]; edits=[number(r.get("edit_ok")) for r in rr]
        summaries.append(dict(arm=arm,planned=n,completed=sum(r.get("status")=="completed" for r in rr),
            joint_judged=sum(v is not None for v in joint),edit_judged=sum(v is not None for v in edits),
            joint_success_rate=sum(joint)/n if all(v is not None for v in joint) else None,
            edit_success_rate=sum(edits)/n if all(v is not None for v in edits) else None,
            joint_lower_bound=sum(v==1 for v in joint)/n,joint_upper_bound=sum(v is None or v==1 for v in joint)/n))
    write_csv(out/"success_coverage.csv",summaries)
    metrics=["joint_success","edit_ok","pattern_ncc_shift6","mae_vs_source","object_mae","bg_mae",
             "clip_target_similarity","clip_directional_similarity","clip_image_similarity","dino_full_feature_similarity","lpips_full"]
    base={key(r):r for r in byarm.get(args.baseline,[])}
    if not base: raise SystemExit("Baseline absent: "+args.baseline)
    pairs=[]; comparisons=[]
    for arm,rr in byarm.items():
        if arm==args.baseline: continue
        for metric in metrics:
            grouped=defaultdict(list);expected=0;missing=0
            for r in rr:
                b=base.get(key(r))
                if b is None: continue
                expected+=1; va,vb=number(r.get(metric)),number(b.get(metric))
                if va is None or vb is None: missing+=1;continue
                d=va-vb;grouped[r["case"]].append(d)
                pairs.append(dict(arm=arm,baseline=args.baseline,metric=metric,case=r["case"],task=r.get("task"),seed=r.get("seed"),difference=d))
            if not grouped: continue
            # Success inference requires every planned matched pair to be judged.
            if metric in ("joint_success","edit_ok") and missing: continue
            means=[statistics.mean(ds) for ds in grouped.values()]
            mean,lo,hi,pv=inference(means,rng,args.replicates)
            comparisons.append(dict(arm=arm,baseline=args.baseline,metric=metric,source_n=len(means),
                pair_n=sum(map(len,grouped.values())),missing_pairs=missing,expected_pairs=expected,
                source_weighted_difference=mean,ci95_low=lo,ci95_high=hi,p_two_sided_signflip=pv,
                scope="development_exploratory",estimator="mean within source, then mean across sources"))
    # Holm correction within each metric across arm comparisons.
    for metric in metrics:
        family=sorted([r for r in comparisons if r["metric"]==metric and r["p_two_sided_signflip"] is not None],key=lambda r:r["p_two_sided_signflip"])
        previous=0
        for i,r in enumerate(family):
            previous=max(previous,min(1,(len(family)-i)*r["p_two_sided_signflip"]))
            r["p_holm_within_metric"]=previous
    write_csv(out/"paired_differences.csv",pairs);write_csv(out/"source_clustered_comparisons.csv",comparisons)
    # Arm points retain all planned rows. No post-hoc subset matching by outcome.
    points=[]
    for s in summaries:
        rr=byarm[s["arm"]];vals=[number(r.get("pattern_ncc_shift6")) for r in rr]
        points.append(dict(**s,pattern_ncc_shift6=statistics.mean(v for v in vals if v is not None) if any(v is not None for v in vals) else None,
                           pattern_available=sum(v is not None for v in vals),pareto_status="pending_semantics" if s["edit_success_rate"] is None else "arm_level_point"))
    write_csv(out/"editing_preservation_points.csv",points)
    print("Evidence tables:",out,"; human labels remain pending unless imported.")


if __name__=="__main__": main()
