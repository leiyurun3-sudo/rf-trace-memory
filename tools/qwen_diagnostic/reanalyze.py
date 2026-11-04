"""Zero-GPU audit of saved diagnostic records; write one NEW audit JSON only.

Usage: python -B reanalyze.py ARCHIVE_DIRECTORY
Uses recorded tensor diagnostics, not a tensor recomputation. No model imports.
"""
import argparse
import json
import math
from pathlib import Path

from probe import _decode_json, assess_colors


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest_string(value):
    return (isinstance(value,str) and len(value)==64
            and all(c in '0123456789abcdef' for c in value))


def input_audit(task,record):
    images=record.get('images',[])
    audit=record.get('input_audit',{})
    grid=audit.get('image_grid_thw',[])
    hashes=[image.get('sha256') for image in images]
    tensors=audit.get('per_image_tensor_sha256',[])
    expected=audit.get('expected_image_tokens')
    actual=audit.get('actual_image_tokens')
    checks={
        'inference_complete':record.get('status')=='inference_complete',
        'image_count_matches_plan':len(images)==len(task['images']),
        'image_paths_match_plan':[image.get('path') for image in images]==task['images'],
        'rendered_slots_match_images':audit.get('rendered_image_slots')==len(images),
        'prompt_slots_match_images':record.get('rendered_prompt','').count('<|image_pad|>')==len(images),
        'grid_count_matches_images':len(grid)==len(images),
        'image_tokens_match':type(expected) is int and expected>0 and type(actual) is int and actual==expected,
        'pixels_finite_recorded':audit.get('pixels_finite') is True,
        'pixel_extrema_finite':all(type(audit.get(k)) in (int,float) and math.isfinite(audit[k]) for k in ('pixel_min','pixel_max')),
        'file_hashes_present':bool(images) and all(digest_string(h) for h in hashes),
        'tensor_hashes_present':len(tensors)==len(images) and all(digest_string(h) for h in tensors),
    }
    grid_valid=bool(grid) and all(isinstance(g,list) and len(g)==3 and all(type(v) is int and v>0 for v in g) for g in grid)
    shape=audit.get('pixel_values_shape',[])
    checks['grid_rows_match_pixel_tensor']=(grid_valid and isinstance(shape,list) and len(shape)==2
                                            and shape[0]==sum(math.prod(g) for g in grid))
    same_file=hashes[0]==hashes[1] if len(hashes)==2 and checks['file_hashes_present'] else None
    same_tensor=tensors[0]==tensors[1] if len(tensors)==2 and checks['tensor_hashes_present'] else None
    if same_file is True:
        checks['identity_pair_tensor_hashes_equal']=same_tensor is True
    return {'id':task['id'],'kind':task.get('kind','synthetic'),
            'record_file':'responses/'+task['id']+'.json',
            'image_count':len(images),'rendered_image_slots':audit.get('rendered_image_slots'),
            'expected_image_tokens':expected,'actual_image_tokens':actual,
            'pixels_finite':audit.get('pixels_finite'),'image_grid_thw':grid,
            'file_sha256':hashes,'per_image_tensor_sha256':tensors,
            'same_file_hash':same_file,'same_tensor_hash':same_tensor,
            'checks':checks,'all_checks_ok':all(checks.values())}


def response_audit(record,kind):
    result={'id':record['id'],'record_file':'responses/'+record['id']+'.json',
            'raw_response':record.get('raw_response'),'json_syntax_ok':False,'schema_ok':False}
    try: data=_decode_json(record.get('raw_response'))
    except ValueError as exc:
        result['syntax_error']=str(exc);return result
    result.update(json_syntax_ok=True,reported_data=data)
    try:
        if not isinstance(data,dict): raise ValueError('Expected a JSON object')
        if kind=='pair_observation':
            required=('image_1','image_2','differences','same_image')
            if any(k not in data for k in required): raise ValueError('Missing neutral-pair field')
            for key in ('image_1','image_2','differences'):
                if not isinstance(data[key],str) or not data[key].strip():
                    raise ValueError(key+' must be a nonempty string under the original prompt schema')
            if data['same_image'] is not None and type(data['same_image']) is not bool:
                raise ValueError('same_image must be boolean or null')
        else:
            if kind=='old_rating':
                for key in ('edit','artifact'):
                    if key not in data or not isinstance(data[key],dict): raise ValueError('Missing nested '+key)
                    item=data[key]
                    if 'value' not in item or 'reason' not in item: raise ValueError('Missing value/reason in '+key)
                    if item['value'] is not None and (type(item['value']) is not int or item['value'] not in (0,1)):
                        raise ValueError(key+' value must be integer 0/1 or null')
                    if not isinstance(item['reason'],str) or not item['reason'].strip(): raise ValueError('Missing '+key+' reason')
                value=data['edit']['value']
            else:
                for key in ('observed_source','observed_candidate','reason','edit'):
                    if key not in data: raise ValueError('Missing '+key)
                for key in ('observed_source','observed_candidate','reason'):
                    if not isinstance(data[key],str) or not data[key].strip(): raise ValueError(key+' must be a nonempty string')
                value=data['edit']
                if value is not None and (type(value) is not int or value not in (0,1)):
                    raise ValueError('edit must be integer 0/1 or null')
            result['validated_edit']=value
        result['schema_ok']=True
    except ValueError as exc:
        result['schema_error']=str(exc)
    return result


def analyze(root):
    plan=load(root/'probe_plan.json')
    if len({t['id'] for t in plan})!=len(plan): raise ValueError('Duplicate task IDs in plan')
    records={}
    missing=[]
    for task in plan:
        if Path(task['id']).name!=task['id'] or '/' in task['id'] or '\\' in task['id']:
            raise ValueError('Unsafe task ID')
        path=root/'responses'/(task['id']+'.json')
        if path.is_file():
            record=load(path)
            if record.get('id')!=task['id']: raise ValueError('Record ID mismatch: '+str(path))
            records[task['id']]=record
        else: missing.append(str(path.relative_to(root)))
    inputs=[input_audit(t,records[t['id']]) for t in plan if t['id'] in records]
    colors=[]
    for task in plan:
        if task.get('expected_colors') is not None and task['id'] in records:
            record=records[task['id']]
            colors.append({'id':task['id'],'expected_colors':task['expected_colors'],
                           'raw_response':record.get('raw_response'),
                           'archived_known_answer_match':record.get('known_answer_match'),
                           **assess_colors(record.get('raw_response'),task['expected_colors'])})
    comparison=[]
    neutral=[]
    for task in plan:
        if task.get('kind')!='old_rating': continue
        case=task['id'].removesuffix('_old_rating')
        entry={'case':case}
        for kind in ('old_rating','evidence_rating','pair_observation'):
            record=records.get(case+'_'+kind)
            audit=response_audit(record,kind) if record is not None else {'missing':True,'schema_ok':False}
            matching=next((x for x in inputs if x['id']==case+'_'+kind),None)
            audit['identity_file_hashes_equal']=matching['same_file_hash'] if matching else None
            audit['identity_tensor_hashes_equal']=matching['same_tensor_hash'] if matching else None
            if kind=='pair_observation':
                neutral.append({'case':case,**audit})
            else: entry[kind]=audit
        comparison.append(entry)
    def rating_summary(kind):
        rows=[x[kind] for x in comparison]
        valid=[x for x in rows if x.get('schema_ok') is True]
        return {'total':len(rows),'schema_ok':len(valid),
                'edit_yes':sum(x['validated_edit']==1 for x in valid),
                'edit_no':sum(x['validated_edit']==0 for x in valid),
                'edit_null':sum(x['validated_edit'] is None for x in valid)}
    # Compare recorded default-processor tensor hashes under image-order reversal.
    forward=records.get('synthetic_red_blue',{}).get('input_audit',{}).get('per_image_tensor_sha256')
    reverse=records.get('synthetic_blue_red',{}).get('input_audit',{}).get('per_image_tensor_sha256')
    order_ok=(forward==list(reversed(reverse))) if isinstance(forward,list) and isinstance(reverse,list) and len(forward)==len(reverse)==2 else None
    identity=[x for x in inputs if x['same_file_hash'] is True]
    return {'audit_version':'saved-diagnostic-audit-v1','archive_directory':str(root),
            'scope':'Zero-GPU reanalysis of raw replies and recorded input diagnostics; no method ranking or repaired labels.',
            'tensor_evidence':'Recorded checksums/shapes/finite flags only; tensors are not stored or recomputed.',
            'original_summary':load(root/'summary.json'),
            'input_summary':{'planned_calls':len(plan),'records_found':len(records),'missing_records':missing,
                             'image_occurrences':sum(x['image_count'] for x in inputs),
                             'calls_all_checks_ok':sum(x['all_checks_ok'] for x in inputs),
                             'identity_pairs':len(identity),
                             'identity_tensor_equal':sum(x['same_tensor_hash'] is True for x in identity),
                             'default_red_blue_tensor_order_reversed_correctly':order_ok},
            'per_call_input_audit':inputs,
            'colors_summary':{'total':len(colors),
                              'json_syntax_ok':sum(x['json_syntax_ok'] for x in colors),
                              'accepted_schema_ok':sum(x['colors_schema_ok'] for x in colors),
                              'semantic_exact_colors_match':sum(x['semantic_exact_colors_match'] is True for x in colors)},
            'colors':colors,
            'identity_ratings_summary':{'old_rating':rating_summary('old_rating'),
                                        'evidence_rating':rating_summary('evidence_rating')},
            'identity_rating_comparisons':comparison,
            'neutral_pair_summary':{'total':len(neutral),'schema_ok':sum(x['schema_ok'] for x in neutral),
                                    'reported_same_image_true':sum(isinstance(x.get('reported_data'),dict) and x['reported_data'].get('same_image') is True for x in neutral),
                                    'note':'Reported fields remain visible even when the full prompt schema fails; no list-to-string coercion.'},
            'neutral_pairs':neutral}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive_directory',type=Path)
    args=parser.parse_args()
    root=args.archive_directory.resolve()
    destination=root/'diagnostic_audit.json'
    if destination.exists(): raise FileExistsError('Refusing to overwrite existing audit: '+str(destination))
    audit=analyze(root)
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(audit,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:audit[key] for key in ('input_summary','colors_summary','identity_ratings_summary','neutral_pair_summary')},ensure_ascii=False,indent=2))
    print('Wrote NEW audit:',destination)


if __name__=='__main__':
    main()
