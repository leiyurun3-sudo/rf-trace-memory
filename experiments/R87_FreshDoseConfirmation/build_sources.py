"""Build the frozen fresh-source cohort from an existing local PIE archive."""
import argparse, hashlib, io, json, re, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
SALT = 'R87-confirm-20261104'
CATS = ['random editing','change object','add object','delete object',
        'change attribute content','change attribute pose','change attribute color',
        'change attribute material','change background','change style']

def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def fingerprint(im):
    im = ImageOps.exif_transpose(im).convert('RGB')
    thumb = np.asarray(im.resize((64,64), Image.Resampling.LANCZOS), dtype=np.int16)
    gray = np.asarray(im.convert('L').resize((9,8), Image.Resampling.LANCZOS))
    bits = (gray[:,:-1] > gray[:,1:]).reshape(-1)
    return dict(rgb=sha(f'{im.width}x{im.height}:RGB:'.encode()+im.tobytes()),
                thumb=thumb, dhash=int(''.join('1' if b else '0' for b in bits),2))
def same(a,b):
    return a['rgb']==b['rgb'] or ((a['dhash']^b['dhash']).bit_count()<=8 and float(np.abs(a['thumb']-b['thumb']).mean())<=12)
def mask_of(item):
    values=item.get('mask')
    if not isinstance(values,list) or len(values)%2: return None
    pixels=np.zeros(512*512, dtype=np.uint8)
    for start,length in zip(values[::2],values[1::2]):
        if type(start) is not int or type(length) is not int or start<0 or length<0 or start+length>pixels.size: return None
        pixels[start:start+length]=255
    if not np.any(pixels) or np.all(pixels): return None
    pixels=pixels.reshape(512,512);pixels[[0,-1],:]=255;pixels[:,[0,-1]]=255
    return pixels if np.any(pixels==0) else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);a=p.parse_args()
    if (ROOT/'manifest.json').exists(): raise FileExistsError('Frozen manifest already exists; do not resample after results.')
    mapping=read(a.raw/'mapping_file.json');prior_cases=set();fps={};plans=[];missing=set()
    for path in sorted(a.history.glob('R*/plan.json')):
        if not re.match(r'R(?:7[1-9]|8[0-6])_',path.parent.name): continue
        obj=read(path);records=obj.get('tasks',[])+obj.get('jobs',[])
        cases={str(t['case']) for t in records if 'case' in t};prior_cases.update(cases)
        plans.append(dict(directory=path.parent.name,case_count=len(cases)))
        for case in sorted(cases):
            source=path.parent/'data'/case/'source.png'
            if not source.is_file(): missing.add(case);continue
            with Image.open(source) as im: f=fingerprint(im)
            fps.setdefault(case,f)
    missing-=set(fps)
    if missing: raise RuntimeError('History source images missing for '+repr(sorted(missing)))
    if len(prior_cases)<70: raise RuntimeError('History exclusion incomplete: expected at least known 70 case IDs.')
    chosen=[];rejected=Counter();selected=[]
    with zipfile.ZipFile(a.raw/'piebench_inputs.zip') as z:
        members={}
        for name in z.namelist():
            if name.lower().endswith(('.png','.jpg','.jpeg','.webp')): members.setdefault(Path(name).stem,[]).append(name)
        for cat in range(10):
            candidates=sorted((sha((SALT+':'+case).encode()),case,item) for case,item in mapping.items() if str(item.get('editing_type_id'))==str(cat))
            n=0
            for order,case,item in candidates:
                if n==2: break
                if case in prior_cases: rejected['historical_id']+=1;continue
                mask=mask_of(item)
                if mask is None: rejected['empty_or_invalid_evaluation_region']+=1;continue
                if not all(isinstance(item.get(k),str) and item[k].strip() for k in ('original_prompt','editing_prompt','image_path')): rejected['missing_prompt_or_path']+=1;continue
                names=members.get(Path(item['image_path'].replace('\\','/')).stem,[])
                if len(names)!=1: rejected['missing_or_ambiguous_image']+=1;continue
                raw=z.read(names[0])
                with Image.open(io.BytesIO(raw)) as im: image=ImageOps.exif_transpose(im).convert('RGB')
                info=fingerprint(image)
                if any(same(info,old) for old in fps.values()): rejected['historical_content']+=1;continue
                if any(same(info,old) for old in selected): rejected['within_cohort_content']+=1;continue
                selected.append(info);n+=1;chosen.append((case,item,image.copy(),mask,info,order,names[0]))
            if n!=2: raise RuntimeError('Insufficient eligible new sources in category '+str(cat))
    tasks=[];annotations={}
    for case,item,image,mask,info,order,member in chosen:
        folder=ROOT/'data'/case;folder.mkdir(parents=True,exist_ok=True)
        image.resize((512,512),Image.Resampling.LANCZOS).save(folder/'source.png')
        Image.fromarray(mask).save(folder/'outer.png')
        Image.new('L',(512,512),255).save(folder/'full_mask.png');Image.new('L',(512,512),0).save(folder/'core.png')
        cat=str(item['editing_type_id'])
        tasks.append(dict(case=case,key='pie_'+case,task='pie_'+case,image=f'data/{case}/source.png',source_image=f'data/{case}/source.png',
            source_prompt=item['original_prompt'].replace('[','').replace(']',''),target_prompt=item['editing_prompt'].replace('[','').replace(']',''),
            target_category_id=cat,target_category=CATS[int(cat)],edit_rubric=item.get('editing_instruction') or item['editing_prompt'],seeds=[0,17],
            pattern_roi=[0,0,512,512],detail_roi=[0,0,512,512],eval_pixel_region='background',diagnostic_only=False,
            roi_scope='whole_image_auxiliary',cohort='fresh_confirmation',source_rgb_sha256=info['rgb'],source_size=list(image.size)))
        annotations[case]=dict(official_record=item,archive_member=member,selection_sha256=order,edit_fraction=float((mask>0).mean()),evaluation_only=True)
    write(ROOT/'manifest.json',dict(dataset='PIE-Bench',tasks=tasks,seeds=[0,17],source_count=20,category_counts={str(i):2 for i in range(10)},
        selection=f'two per category, ascending SHA256({SALT}:case_id), before any R87 generation',
        eligibility='valid prompts and nonempty official edit AND preserved region; no outcome-based exclusions',
        historical_exclusions=sorted(prior_cases),history_plans=plans,
        content_exclusion='EXIF RGB SHA or (dHash distance <=8 AND RGB thumbnail MAE <=12); historical and within-cohort',
        evaluation_mask='outer=edit255/preserve0; row-major official RLE and four edited border edges; never used by generation',
        source='https://github.com/cure-lab/PnPInversion',retrieval='local PIE archive previously retrieved from yanlinli/image-editing-evaluation-handoff',
        annotation_license='CC BY-NC-SA 4.0; source images retain original rights',rejections=dict(rejected),annotations=annotations))
    print(json.dumps(dict(selected=20,historical_ids=len(prior_cases),historical_images=len(fps),rejected=dict(rejected))))
if __name__=='__main__': main()
