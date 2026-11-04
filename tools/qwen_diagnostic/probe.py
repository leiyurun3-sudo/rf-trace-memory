"""Offline diagnostic, 26 calls; no edits to existing ratings or experiment images."""
import argparse
import hashlib
import json
import os
import platform
import re
import time
import zipfile
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
HERE=Path(__file__).resolve().parent

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _decode_json(text):
    if not isinstance(text,str): raise ValueError('Response must be text')
    s=text.strip()
    fence=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',s,re.DOTALL|re.IGNORECASE)
    if fence: s=fence.group(1)
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    def constant(value): raise ValueError('Nonstandard JSON constant: '+value)
    return json.loads(s,object_pairs_hook=pairs,parse_constant=constant)

def parse(text):
    try: return _decode_json(text)
    except ValueError: return None

def assess_colors(text,expected):
    """Separate syntax, accepted structure, and exact ordered color content.

    Accept a string list or an object with a colors string list. Preserve case,
    order and labels; never infer, lowercase, complete or replace an answer.
    Invalid syntax/schema leaves semantic matching unresolved (None).
    """
    result={'json_syntax_ok':False,'colors_schema_ok':False,
            'semantic_exact_colors_match':None}
    try: data=_decode_json(text)
    except ValueError as exc:
        result['syntax_error']=str(exc);return result
    result['json_syntax_ok']=True
    colors=data.get('colors') if isinstance(data,dict) else data
    if not isinstance(colors,list) or not colors or any(not isinstance(c,str) or not c.strip() for c in colors):
        result['schema_error']='Expected a nonempty list of nonempty strings, or an object containing that list under colors.'
        return result
    result.update(colors_schema_ok=True,colors=list(colors),
                  response_form='object' if isinstance(data,dict) else 'list',
                  semantic_exact_colors_match=colors==expected)
    return result

OBSERVATION='Describe only the visible objects, colors, and their locations. Do not infer any requested edit. Keep the description under 80 words.'
PAIR_OBSERVATION='Describe image 1 and image 2 separately, then list visible differences. Images may be identical. Do not assume an editing operation occurred. Return JSON with string fields image_1, image_2, differences, and a same_image boolean (or null if uncertain).'
NEW_RATING='''Image 1 is the source; image 2 is the candidate. Requested change: {edit}
Inspect the images, not a description of what the candidate ought to show. They may be identical, partially edited, or incorrectly edited.
First describe visible evidence in each image. Then decide whether the requested change is achieved in image 2. Return one JSON object with string fields observed_source, observed_candidate, reason, and edit (integer 1 if achieved, 0 if not achieved, null only if not visually decidable). No example answer is supplied.'''
NEW_RUBRIC='''Inspect this source image for the requested change: {edit}
List what should change and what unrelated content should remain. A color, pose or style change does not automatically remove all identity and unrelated-content requirements. Do not require preservation of the entity explicitly requested for removal or replacement.
Independently decide whether subject preservation and distinctive pattern preservation have a meaningful criterion for this task. Return JSON: requested_edit and protected_requirements (arrays of strings), subject_applicable and pattern_applicable (boolean or null only if visually undecidable), subject_reason and pattern_reason (strings). No example answers are supplied. This is a diagnostic response, not a rule used to score other outputs.'''

def plan(root,out):
    from PIL import Image,ImageDraw
    cases=load(HERE/'cases.json')
    stimuli=out/'stimuli';stimuli.mkdir(parents=True,exist_ok=True)
    for name,rgb in [('red',(220,20,30)),('blue',(20,40,220))]:
        p=stimuli/(name+'.png'); im=Image.new('RGB',(448,448),'white')
        ImageDraw.Draw(im).rectangle((64,64,384,384),fill=rgb); im.save(p)
    tasks=[]
    cp='Name the color of the large central square in each image, in image order. Return JSON with colors, an array of lowercase English color names.'
    for names in [('red',),('blue',),('red','blue'),('blue','red'),('red','red')]:
        tasks.append(dict(id='synthetic_'+'_'.join(names),images=[str(stimuli/(s+'.png')) for s in names],prompt=cp,expected_colors=list(names),processor='default'))
    for case in cases:
        src=root/case['source']
        if not src.is_file(): raise FileNotFoundError(src)
        if sha(src)!=case['source_sha256']: raise ValueError('Source differs from archived evaluation: '+str(src))
        cid=case['case'];edit=case['edit']
        for suffix,ims,prompt in [
            ('caption',[src],OBSERVATION),
            ('pair_observation',[src,src],PAIR_OBSERVATION),
            ('old_rating',[src,src],case['old_rating_prompt']),
            ('evidence_rating',[src,src],NEW_RATING.format(edit=edit)),
            ('old_rubric',[src],case['old_rubric_prompt']),
            ('evidence_rubric',[src],NEW_RUBRIC.format(edit=edit)),
        ]:
            tasks.append(dict(id=cid+'_'+suffix,images=[str(p) for p in ims],prompt=prompt,processor='default',kind=suffix))
    # Two genuinely different source/output pairs check the same pipeline without an identity control.
    for k,real in enumerate(cases[0]['real_outputs']):
        src=root/cases[0]['source']; op=root/real['path']
        if not op.is_file() or sha(op)!=real['sha256']: raise ValueError('Missing/changed native output: '+str(op))
        tasks.append(dict(id='real_pair_'+str(k+1),images=[str(src),str(op)],prompt=NEW_RATING.format(edit=cases[0]['edit']),processor='default',kind='real_pair'))
    tasks.append(dict(id='synthetic_red_blue_slow',images=[str(stimuli/'red.png'),str(stimuli/'blue.png')],prompt=cp,expected_colors=['red','blue'],processor='slow'))
    save(out/'probe_plan.json',tasks)
    return tasks

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--plan-only',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    tasks=plan(a.root,a.out)
    print('Planned',len(tasks),'diagnostic calls. Existing ratings are read-only.',flush=True)
    if a.plan_only: return
    import torch,transformers,PIL
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback.')
    conf=load(a.model/'config.json')
    if conf.get('model_type')!='qwen2_5_vl': raise ValueError('Expected Qwen2.5-VL checkpoint')
    idx=load(a.model/'model.safetensors.index.json')
    missing=[x for x in set(idx['weight_map'].values()) if not (a.model/x).is_file()]
    if missing: raise FileNotFoundError('Missing local shards: '+str(missing))
    meta={'torch':torch.__version__,'transformers':transformers.__version__,'pillow':PIL.__version__,
          'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'model':str(a.model),
          'config_sha256':sha(a.model/'config.json'),'mode':'offline diagnostic; no score migration',
          'model_revision':load(a.model/'judge_model_revision.json') if (a.model/'judge_model_revision.json').is_file() else None}
    save(a.out/'runtime.json',meta)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(a.model),local_files_only=True,device_map='auto',torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model.eval();model.generation_config.do_sample=False
    for k in ('temperature','top_p','top_k'): setattr(model.generation_config,k,None)
    processors={}; results=[];started=time.monotonic()
    for i,t in enumerate(tasks,1):
        record={'id':t['id'],'prompt':t['prompt'],'images':[], 'kind':t.get('kind','synthetic'),'expected_colors':t.get('expected_colors')}
        try:
            mode=t['processor']
            if mode not in processors:
                kw={'use_fast':False} if mode=='slow' else {}
                processors[mode]=AutoProcessor.from_pretrained(str(a.model),local_files_only=True,max_pixels=262144,**kw)
            processor=processors[mode];ims=[]
            for p in t['images']:
                with Image.open(p) as im: im=im.convert('RGB');ims.append(im)
                record['images'].append({'path':p,'sha256':sha(p),'size':list(im.size)})
            messages=[{'role':'user','content':[{'type':'image'} for _ in ims]+[{'type':'text','text':t['prompt']}]}]
            rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[rendered],images=ims,padding=True,return_tensors='pt')
            grid=inputs.get('image_grid_thw');pixels=inputs.get('pixel_values')
            if grid is None or pixels is None or len(grid)!=len(ims): raise RuntimeError('Missing/mismatched vision tensors')
            merge=int(getattr(processor.image_processor,'merge_size',2))
            expected=int((grid.prod(dim=1)//(merge*merge)).sum().item())
            itoken=getattr(model.config,'image_token_id',None)
            actual=int((inputs['input_ids']==itoken).sum().item()) if itoken is not None else None
            record['input_audit']={'processor':type(processor).__name__,'image_processor':type(processor.image_processor).__name__,
                'use_fast_requested':mode,'rendered_image_slots':rendered.count('<|image_pad|>'),
                'image_grid_thw':grid.tolist(),'pixel_values_shape':list(pixels.shape),
                'pixel_values_dtype':str(pixels.dtype),'pixels_finite':bool(torch.isfinite(pixels).all()),
                'pixel_min':float(pixels.min()),'pixel_max':float(pixels.max()),
                'expected_image_tokens':expected,'actual_image_tokens':actual,
                'input_ids_shape':list(inputs['input_ids'].shape)}
            # Save per-image feature checksums to confirm that swapping image order swaps tensors.
            offset=0;checks=[]
            for g in grid:
                length=int(g.prod().item());v=pixels[offset:offset+length].float().contiguous().numpy()
                checks.append(hashlib.sha256(v.tobytes()).hexdigest());offset+=length
            record['input_audit']['per_image_tensor_sha256']=checks
            if actual!=expected or not record['input_audit']['pixels_finite']: raise RuntimeError('Vision token or finite-value check failed')
            record['rendered_prompt']=rendered
            inputs=inputs.to(model.device)
            tic=time.monotonic()
            with torch.inference_mode(): generated=model.generate(**inputs,max_new_tokens=768,do_sample=False)
            raw=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            record.update(status='inference_complete',seconds=time.monotonic()-tic,raw_response=raw,parsed=parse(raw),
                          generated_tokens=int(generated.shape[1]-inputs.input_ids.shape[1]))
            if t.get('expected_colors'):
                record['color_audit']=assess_colors(raw,t['expected_colors'])
                record['known_answer_match']=record['color_audit']['semantic_exact_colors_match']
        except Exception as e:
            record.update(status='error',error=type(e).__name__+': '+str(e))
        save(a.out/'responses'/(t['id']+'.json'),record);results.append(record)
        elapsed=time.monotonic()-started
        save(a.out/'progress.json',{'completed':i,'planned':len(tasks),'errors':sum(r['status']=='error' for r in results),'elapsed_seconds':elapsed})
        print(i,'/',len(tasks),t['id'],record['status'],record.get('error',''),flush=True)
    synthetic=[r for r in results if r.get('expected_colors')]
    summary={'calls':len(results),'errors':sum(r['status']=='error' for r in results),
             'known_color_syntax_pass':sum(r.get('color_audit',{}).get('json_syntax_ok') is True for r in synthetic),
             'known_color_schema_pass':sum(r.get('color_audit',{}).get('colors_schema_ok') is True for r in synthetic),
             'known_color_pass':sum(r.get('known_answer_match') is True for r in synthetic),'known_color_total':len(synthetic),
             'note':'Diagnostic only. Inspect real-case observations, old/new prompts, and frozen applicability; no method ranking or repaired labels.'}
    save(a.out/'summary.json',summary)
    dest=a.out.with_suffix('.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(a.out.rglob('*')):
            if p.is_file(): z.write(p,Path(a.out.name)/p.relative_to(a.out))
    print(json.dumps(summary,ensure_ascii=False),flush=True);print('Result ZIP:',dest,flush=True)

if __name__=='__main__': main()
