from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")

def preflight_runtime(plan):
    """Read-only checks before pausing B. No torch import or CUDA allocation."""
    import importlib.metadata
    for name,v in [('diffusers','0.35.1'),('transformers','4.55.4'),('accelerate','1.10.1')]:
        found=importlib.metadata.version(name)
        if found!=v:raise RuntimeError('Existing environment mismatch: '+name+' '+found+'; expected '+v+'. No installation attempted.')
    for name in ['torch','numpy','Pillow','safetensors']:importlib.metadata.version(name)
    repo='models--black-forest-labs--FLUX.1-dev'
    candidates=[Path('/root/autodl-tmp/hf_cache/hub')/repo/'snapshots'/plan['model_revision']]
    if os.environ.get('HF_HOME'):candidates.append(Path(os.environ['HF_HOME'])/'hub'/repo/'snapshots'/plan['model_revision'])
    path=next((p for p in candidates if (p/'model_index.json').exists()),None)
    if path is None:raise RuntimeError('Cached FLUX snapshot missing. No download attempted.')
    for sub in ['transformer','text_encoder','text_encoder_2','vae']:
        folder=path/sub
        weights=set()
        for index in folder.glob('*.index.json'):weights.update(read(index)['weight_map'].values())
        if not weights:weights={p.name for p in folder.glob('*.safetensors')}
        if not weights or any(not (folder/name).is_file() for name in weights):raise RuntimeError('Missing local weight shard: '+sub)
    return str(path)

def load_model(plan,progress):
    import torch,importlib.metadata
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no simulated fallback')
    for name,v in [('diffusers','0.35.1'),('transformers','4.55.4'),('accelerate','1.10.1')]:
        if importlib.metadata.version(name)!=v:raise RuntimeError('Use existing environment: '+name+' '+v)
    if torch.cuda.mem_get_info()[0]<10*1024**3:raise RuntimeError('Need >=10GiB free GPU; wait for other experiments')
    repo='models--black-forest-labs--FLUX.1-dev';rev=plan['model_revision']
    candidates=[Path('/root/autodl-tmp/hf_cache/hub')/repo/'snapshots'/rev]
    if os.environ.get('HF_HOME'):candidates.append(Path(os.environ['HF_HOME'])/'hub'/repo/'snapshots'/rev)
    path=next((p for p in candidates if (p/'model_index.json').exists()),None)
    if path is None:raise RuntimeError('Pinned FLUX.1-dev snapshot missing; no download attempted')
    with progress.phase('checking cached FLUX.1-dev file headers'):
        cfg=read(path/'transformer/config.json')
        if cfg.get('in_channels')!=64:raise RuntimeError('Expected base FLUX.1-dev, not Fill')
        inventory=[]
        for sub in ['transformer','text_encoder','text_encoder_2','vae']:
            folder=path/sub;indices=list(folder.glob('*.index.json'))
            weights=set()
            for index in indices:weights.update(read(index)['weight_map'].values())
            if not weights:weights={p.name for p in folder.glob('*.safetensors')}
            if not weights:raise RuntimeError('No local weights: '+sub)
            for name in sorted(weights):
                f=folder/name
                with f.open('rb') as handle:
                    raw=handle.read(8)
                    if len(raw)!=8:raise RuntimeError('Truncated safetensors: '+str(f))
                    length=int.from_bytes(raw,'little')
                    if not 2<length<100_000_000:raise RuntimeError('Invalid weight header')
                    header=json.loads(handle.read(length))
                end=max(x['data_offsets'][1] for k,x in header.items() if k!='__metadata__')
                if f.stat().st_size!=8+length+end:raise RuntimeError('Wrong weight size: '+str(f))
                inventory.append(dict(file=str(f),bytes=f.stat().st_size,header_sha256=hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest()))
        save(OUT/'model_inventory.json',dict(snapshot=str(path),weights=inventory,integrity_scope='headers and exact data length; not full weight content hashes'))
    from diffusers import FluxPipeline
    with progress.phase('loading BF16 base model; offline'):
        pipe=FluxPipeline.from_pretrained(str(path),torch_dtype=torch.bfloat16,local_files_only=True)
        if pipe.vae_scale_factor!=8:raise RuntimeError('Unexpected VAE/token packing')
        if torch.cuda.mem_get_info()[0]>=27*1024**3:
            pipe.enable_model_cpu_offload();offload_mode='model'
        else:
            pipe.enable_sequential_cpu_offload();offload_mode='sequential'
            print('Low VRAM: sequential CPU offload enabled; inference is slower.',flush=True)
        pipe.vae.enable_tiling();pipe.vae.enable_slicing()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    save(OUT/'environment.json',dict(gpu=torch.cuda.get_device_name(0),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,versions={x:importlib.metadata.version(x) for x in ['torch','diffusers','transformers','accelerate']},cpu_offload=offload_mode,vae_tiling=True,quantized=False,executable=sys.executable))
    return pipe
