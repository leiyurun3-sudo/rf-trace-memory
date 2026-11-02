import argparse,json,os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--revision',default='main');p.add_argument('--endpoint',help='Explicit optional HF-compatible mirror endpoint');a=p.parse_args()
if a.endpoint: os.environ['HF_ENDPOINT']=a.endpoint
model='Qwen/Qwen2.5-VL-7B-Instruct'; target=Path(a.model_dir).resolve();target.mkdir(parents=True,exist_ok=True)
api=HfApi(endpoint=a.endpoint); revision=api.model_info(model,revision=a.revision).sha
print('Downloading',model,'revision',revision,'to',target,flush=True)
snapshot_download(repo_id=model,revision=revision,local_dir=str(target),endpoint=a.endpoint)
(target/'judge_model_revision.json').write_text(json.dumps({'model':model,'revision':revision},indent=2),encoding='utf-8')
print('Downloaded. Run GPU preflight before inference.')
