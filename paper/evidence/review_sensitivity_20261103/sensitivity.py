import csv, json, zipfile
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
import os
ROOT = Path(os.environ['QWEN_AUDIT_ROOT'])
RESULTS = ROOT / 'results'
ZIP = ROOT / 'original/A_Qwen_FactualJudge_v2_results.zip'
EXCLUDED = {'114000000002': 'giant lake placement', '913000000004': 'Chinese watercolor style', '713000000002': 'ferruginous black sofa'}
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
def conjunction(v):
    v=list(v)
    if 0 in v: return 0
    return 1 if v and all(x == 1 for x in v) else None
def target(facts, goal):
    if facts is None: return None
    return conjunction(None if facts.get(k) in (None, 'uncertain') else int(facts[k] == v) for k,v in goal.items())
def self_edit(facts, goal):
    v=target(facts,goal)
    return 0 if v == 0 else None

def main():
    records=read(RESULTS/'records.json'); tasks=read(RESULTS/'tasks.json'); source=read(RESULTS/'source_facts.json')
    with zipfile.ZipFile(ZIP) as z:
        prefix='A_Qwen_FactualJudge_v2_results/'
        plan=json.loads(z.read(prefix+'code/r85_plan.json'))
        control=json.loads(z.read(prefix+'control_audit.json'))
        names=z.namelist()
        source_observations=[x for x in names if '/observations/source_' in x and x.endswith('.json')]
        identity_names=[x for x in names if any(k in x.lower() for k in ('source_control','source_as','self_control','identity_control'))]
    jobs=plan['jobs']; assert len(jobs)==520 and len(records)==520 and len(tasks)==40
    assert set(j['id'] for j in jobs)==set(records)
    assert set(EXCLUDED)<=set(t['case'] for t in tasks)
    cases={t['case'] for t in tasks}; assert len(cases)==40
    arms=sorted(set(j['arm'] for j in jobs)); rows=[]; detailed=[]
    for j in jobs:
        rec=records[j['id']]; edit=rec.get('edit'); protected=rec.get('protected'); ep=conjunction([edit,protected])
        assert edit in (0,1,None) and protected in (0,1,None)
        assert rec.get('artifact') == 0 and ep==rec.get('joint')
        detailed.append({'id':j['id'],'case':j['case'],'arm':j['arm'],'seed':j['seed'],'excluded':int(j['case'] in EXCLUDED),'edit':edit,'protected':protected,'edit_and_preserve':ep})
    for cohort in ('all_40','exclude_3_37'):
        for arm in arms:
            group=[d for d in detailed if d['arm']==arm and (cohort=='all_40' or not d['excluded'])]
            for metric in ('edit','edit_and_preserve'):
                c=Counter(d[metric] for d in group); n=len(group)
                rows.append({'cohort':cohort,'arm':arm,'metric':metric,'sources':len(set(d['case'] for d in group)),'outputs':n,'pass':c[1],'fail':c[0],'unknown':c[None], 'lower':c[1]/n,'upper':(c[1]+c[None])/n})
    control_rows=[]
    for t in tasks:
        facts=source.get('source_'+t['case']); v=target(facts,t['target_all'])
        control_rows.append({'case':t['case'],'target_match':v,'same_observation_self_edit':self_edit(facts,t['target_all'])})
    ctrl=Counter(r['same_observation_self_edit'] for r in control_rows)
    out={'excluded_sources':EXCLUDED,'scope':'Frozen recovered v2 labels. E AND P uses three-valued conjunction; no artifact criterion. Exclusion is post hoc and changes task mix. Bounds are deterministic missing-label bounds.', 'provenance':{'plan_and_control_archive':str(ZIP),'recovered_records':str(RESULTS/'records.json'),'tasks':str(RESULTS/'tasks.json'),'recovered_source_facts':str(RESULTS/'source_facts.json'),'supplement':'archive/manuscript/supplement.tex'},'source_control_audit':{'actual_v2_40_independent_source_as_output_calls_found':False,'evidence':'Archived runner observes each source once, then observes planned outputs separately; no source-as-output control jobs or observation filenames found. Deterministic self comparison reuses the same observation, not independent inference.', 'source_observation_files':len(source_observations),'identity_control_filenames':identity_names,'logical_recomputation':{'total':len(control_rows),'pass':ctrl[1],'fail':ctrl[0],'unknown':ctrl[None]},'known_image_controls':{'facts':f"{control['correct']}/{control['total']}",'ordered_pairs':f"{control['pair_correct']}/{control['pair_total']}",'clean_artifact':f"{control['artifact_correct']}/{control['artifact_total']}"},'rows':control_rows},'counts':rows}
    for filename,data in [('counts.csv',rows),('output_labels.csv',detailed),('logical_self_comparison.csv',control_rows)]:
        with (HERE/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    (HERE/'results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    label={'RELEASE_CORR25':'CORR25','RELEASE_BOTH25':'BOTH25','ENERGY_MATCH25':'ENERGY','U25_KV':'U25','FLOWEDIT_N26':'FlowEdit','ETA03':'ETA03','RFEDIT_I08':'RF-Edit'}
    lines=[r'\begin{tabular}{lrrrr}',r'\hline',r'Method & $N$ & Edit (P/F/U) & $E\wedge P$ (P/F/U) \\',r'\hline']
    for cohort in ('all_40','exclude_3_37'):
        lines.append(r'\multicolumn{4}{l}{'+('All 40 sources' if cohort=='all_40' else '37 sources after exclusion')+r'} \\')
        for arm in arms:
            x=[r for r in rows if r['cohort']==cohort and r['arm']==arm]; e=next(r for r in x if r['metric']=='edit'); p=next(r for r in x if r['metric']=='edit_and_preserve')
            def fmt(r): return f"{r['pass']}/{r['fail']}/{r['unknown']}"
            lines.append(f"{label.get(arm,arm.replace('_',r'\_'))} & {e['outputs']} & {fmt(e)} & {fmt(p)} " + r'\\')
    lines += [r'\hline',r'\end{tabular}',r'% P/F/U = pass/fail/unknown; exclude three rule-mismatch sources post hoc. Artifact is not part of the conjunction.']
    (HERE/'table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'counts':rows,'logical_control':out['source_control_audit']|{'rows':'omitted'}},indent=2))
if __name__=='__main__': main()
