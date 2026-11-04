"""Post-hoc localization only. Official evaluation masks never enter the editor."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path);a=p.parse_args();root=a.runtime.resolve();plan=json.loads((root/'plan.json').read_text(encoding='utf-8')); out=root/'results';rows=[]
 cells=[dict(id=x['id'],case=x['case'],seed=x['seed'],arm='CORR_1',anchor=True) for x in plan['r86_anchor_cells']]+[dict(id=x['id'],case=x['case'],seed=x['seed'],arm=x['arm'],anchor=False) for x in plan['jobs']]
 for j in cells:
  row=dict(id=j['id'],case=j['case'],seed=j['seed'],arm=j['arm'],map_source='R85_anchor_not_archived' if j['anchor'] else 'R86_native_probe',map_key='',L='',inside_mean='',outside_mean='',inside_mass='',outside_mass='',mask_sha256='',map_sha256='',status='missing_tensor')
  tensor=out/j['id']/'request_maps.npz'; mask=root/'data'/j['case']/'outer.png'
  if not tensor.is_file(): rows.append(row);continue
  if not mask.is_file(): row['status']='missing_official_mask';rows.append(row);continue
  try:
   with np.load(tensor,allow_pickle=False) as archive:
    key='E_shifted' if j['arm'].startswith('SHIFT') else 'E_original'; E=archive[key]
   if E.shape!=(64,64) or E.dtype!=np.float32 or not np.isfinite(E).all(): raise ValueError('bad tensor shape/dtype/finite')
   with Image.open(mask) as m: weight=np.asarray(m.convert('L').resize((64,64),Image.Resampling.BOX),dtype=np.float64)/255.0
   inside=weight.sum();outside=(1-weight).sum();row.update(map_key=key,inside_mass=float(inside),outside_mass=float(outside),mask_sha256=sha(mask),map_sha256=sha(tensor))
   if inside<=0 or outside<=0:row['status']='empty_edit_region' if inside<=0 else 'empty_preservation_region'
   else:
    im=float((E*weight).sum()/inside);om=float((E*(1-weight)).sum()/outside)
    row.update(inside_mean=im,outside_mean=om,L=im-om,status='ok')
  except (OSError,ValueError,KeyError) as e: row['status']='invalid_tensor_or_mask';row['error']=str(e)
  rows.append(row)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (out/'localization_L.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 meta={'scope':'post-hoc diagnostic; official mask not used by inference','L_definition':'area-weighted mean E within official edit mask minus area-weighted mean E outside it; not a pre-existing benchmark metric','mask':'data/<case>/outer.png; grayscale 0..255 converted to [0,1], PIL BOX area downsample to 64x64; mask 1 = edit region','shift':'E_shifted is toroidal 32,32 roll of native E; for GLOBAL E_original is diagnostic of the probe, not an applied global allocation','R85_anchor':'raw float tensor was not archived; marked missing rather than substituting a new run or JPEG','denominators':'inside and outside weighted cell mass; empty region or missing tensor yields missing L','rows':len(rows),'statuses':{st:sum(r['status']==st for r in rows) for st in sorted({r['status'] for r in rows})}}
 (out/'localization_metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8');print('Localization:',meta['statuses'])
if __name__=='__main__':main()
