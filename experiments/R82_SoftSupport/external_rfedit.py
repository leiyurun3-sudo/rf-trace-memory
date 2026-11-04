"""RF-Edit official-algorithm port to the R80 diffusers FLUX backend.

Modified implementation, 2026-10-30: FP32 integration, bounded CPU V bank,
diffusers to_v hooks, integer interval keys, offline CLI. Algorithm lineage:
wangjiangshan0725/RF-Solver-Edit @ 7ac380143379b656eca5398157f41f97ef1ef2f9
FLUX_Image_Edit/src/flux/{sampling.py,model.py,modules/layers.py}, src/edit.py.
Apache-2.0; complete upstream license and configuration notes in BASELINE.md.
No R80 rf_kv/attention/engine algorithm is used here.
"""
import math

import torch


OFFICIAL_COMMIT = '7ac380143379b656eca5398157f41f97ef1ef2f9'
OFFICIAL_REPO = 'https://github.com/wangjiangshan0725/RF-Solver-Edit'


def _schedule(steps, tokens):
    # Official sampling.get_schedule(), including its FP32 linspace nodes.
    t = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device='cpu')
    mu = 0.5 + (tokens - 256) * (1.15 - 0.5) / (4096 - 256)
    return (math.exp(mu) / (math.exp(mu) + (1 / t - 1))).tolist()


class _ValueBank:
    """Whole joint text+image V, two evaluation slots per interval."""
    def __init__(self, model, tokens, max_bytes):
        self.bank = {}
        self.handles = []
        self.mode = 'off'
        self.key = None
        self.seen = set()
        self.tokens = tokens
        self.max_bytes = max_bytes
        self.bytes = self.peak_bytes = self.captured = self.applied = 0
        self.blocks = list(range(20, 38))  # Official info['id'] > 19.
        selected = [model.single_transformer_blocks[i].attn for i in self.blocks]
        # Validate all interfaces before registering anything.
        for attn in selected:
            if getattr(attn, 'fused_projections', False) or not hasattr(attn, 'to_v'):
                raise ValueError('RF-Edit requires unfused to_v projections')
        try:
            for index, attn in zip(self.blocks, selected):
                self.handles.append(attn.to_v.register_forward_hook(self._hook(index)))
        except BaseException:
            self.close()
            raise

    def _hook(self, index):
        def hook(module, args, output):
            if self.mode == 'off':
                return output
            if index in self.seen:
                raise RuntimeError('Repeated V evaluation: unsupported compiled/cached model')
            self.seen.add(index)
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != self.tokens:
                raise ValueError('Expected joint text+image V with shape [1, tokens, hidden]')
            key = (*self.key, index)
            if self.mode == 'capture':
                size = output.numel() * output.element_size()
                if key in self.bank:
                    raise RuntimeError('Duplicate official inversion cache key')
                if self.bytes + size > self.max_bytes:
                    raise MemoryError('RF-Edit CPU V cache exceeds rfedit_max_cache_mib')
                self.bank[key] = output.detach().to('cpu', copy=True)
                self.bytes += size
                self.peak_bytes = max(self.bytes, self.peak_bytes)
                self.captured += 1
                return output
            source = self.bank.pop(key)  # Release each entry after its sole use.
            self.bytes -= source.numel() * source.element_size()
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError('Inversion/denoising V layout or dtype mismatch')
            self.applied += 1
            return source.to(device=output.device)
        return hook

    def begin(self, mode, interval, second_order):
        self.mode, self.key = mode, (interval, second_order)
        self.seen.clear()

    def finish(self):
        if self.mode != 'off' and self.seen != set(self.blocks):
            raise RuntimeError('to_v hooks bypassed: unsupported fused/cached attention')

    def close(self):
        self.mode = 'off'
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.bank.clear()


@torch.inference_mode()
def execute_external(b, y, ids, src, tgt, job, tick=None):
    """Return (packed CPU FP32 latent, JSON-safe stats); performs own inversion.

    src/tgt must be b.embeddings(source/target prompt), y=b.encode(image)[0].
    Do not provide an internal inversion result or an installed attention manager.
    """
    if job.get('method', 'rfedit').lower() not in ('rfedit', 'rf-edit'):
        raise ValueError('external_rfedit supports method=rfedit only')
    steps = int(job.get('rfedit_steps', job.get('steps', b.steps)))
    inject = int(job.get('rfedit_inject_steps', 5))
    max_mib = float(job.get('rfedit_max_cache_mib', 2048))
    gs = float(job.get('rfedit_src_guidance', 1.0))
    gt = float(job.get('rfedit_tgt_guidance', 5.0))
    if not (1 <= inject <= steps <= 50):
        raise ValueError('Bounded RF-Edit requires 1 <= inject_steps <= steps <= 50')
    if not math.isfinite(max_mib) or not 0 < max_mib <= 16384:
        raise ValueError('rfedit_max_cache_mib must be in (0, 16384]')
    if not all(math.isfinite(v) and 0 <= v <= 20 for v in (gs, gt)):
        raise ValueError('RF-Edit guidance must be finite and in [0,20]')
    if float(getattr(b, 'experiment_strength', 1.0)) != 1.0 or float(job.get('strength', 1.0)) != 1.0:
        raise ValueError('RF-Edit uses the full official 0..1 trajectory; strength must be 1')
    if y.ndim != 3 or y.shape[0] != 1 or y.shape[2] != 64:
        raise ValueError('Expected one packed base FLUX image [1, image_tokens, 64]')
    if src[0].shape != tgt[0].shape or src[0].shape[0] != 1:
        raise ValueError('Source and target must use the same padded text length')
    model = b.pipe.transformer
    if len(model.single_transformer_blocks) != 38 or not model.config.guidance_embeds:
        raise ValueError('This port requires the 38-single-block guidance-distilled FLUX.1-dev')
    # Refuse to compose an internal experiment processor with the external baseline.
    for block in list(model.transformer_blocks) + list(model.single_transformer_blocks):
        processor = block.attn.processor
        if (type(processor).__name__ != 'FluxAttnProcessor' or
                type(processor).__module__ != 'diffusers.models.transformers.transformer_flux'):
            raise ValueError('Use native diffusers 0.35.1 FluxAttnProcessor; close internal attention managers first')
    nt = int(src[0].shape[1])
    tokens = int(y.shape[1]) + nt
    # Same BF16 projection dtype as R80; actual allocations are checked again in hooks.
    itemsize = torch.empty((), dtype=b.dtype).element_size()
    hidden = int(model.single_transformer_blocks[20].attn.to_v.out_features)
    estimated = 2 * inject * 18 * tokens * hidden * itemsize
    max_bytes = int(max_mib * 1024**2)
    if estimated > max_bytes:
        raise MemoryError(f'RF-Edit needs {estimated / 1024**2:.1f} MiB V cache; limit {max_mib:g}. Reduce inject steps or raise explicit cap.')
    sigmas = _schedule(steps, int(y.shape[1]))
    z = y.to(b.device, dtype=torch.float32).clone()
    count = b.nfe
    bank = _ValueBank(model, tokens, max_bytes)
    try:
        for inverse, emb, guidance in ((True, src, gs), (False, tgt, gt)):
            # d is the descending interval index. Official inversion info['t']
            # is its upper endpoint, also used as the denoising cache key.
            order = range(steps - 1, -1, -1) if inverse else range(steps)
            for i, d in enumerate(order):
                t = sigmas[d + 1] if inverse else sigmas[d]
                end = sigmas[d] if inverse else sigmas[d + 1]
                h = end - t
                mode = ('capture' if inverse else 'apply') if d < inject else 'off'
                bank.begin(mode, d, False)
                pred = b.field(z, t, emb, ids, guidance)
                bank.finish()
                mid = z + (h / 2) * pred
                bank.begin(mode, d, True)
                pred_mid = b.field(mid, t + h / 2, emb, ids, guidance)
                bank.finish()
                # Preserve the released code's arithmetic, rather than silently
                # switching to paper-v3's fixed 0.01 derivative probe.
                derivative = (pred_mid - pred) / (h / 2)
                z = z + h * pred + 0.5 * h**2 * derivative
                if tick is not None:
                    tick((0 if inverse else steps) + i + 1, 2 * steps)
        expected = 2 * inject * 18
        if bank.bank or bank.captured != expected or bank.applied != expected:
            raise RuntimeError('Incomplete official RF-Edit feature pairing')
        return z.cpu(), dict(
            method='rfedit', implementation='official_algorithm_diffusers_port',
            official_repo=OFFICIAL_REPO, official_commit=OFFICIAL_COMMIT,
            solver='released RF-Solver: half-interval derivative, two NFE per step',
            schedule='official shifted FP32 linspace, full 1..0', sigma_schedule=sigmas,
            steps=steps, inject_steps=inject, single_blocks=list(range(20, 38)),
            src_guidance=gs, tgt_guidance=gt, text_length=nt,
            nfe=int(b.nfe-count), nfe_inversion=2*steps, nfe_edit=2*steps,
            source_nfe=2*steps, target_nfe=2*steps, expected_nfe=4*steps,
            feature_scope='whole joint text+image Value; target Q/K; both solver evaluations',
            cache_key='descending interval index, second_order slot, single block index',
            cache_peak_bytes=bank.peak_bytes, cache_limit_bytes=max_bytes,
            captured_entries=bank.captured, applied_entries=bank.applied,
            inversion_reused=False, uses_internal_rf_kv=False, uses_internal_attention=False,
            precision='backend network dtype; FP32 latent integration',
            seed_used=False, roi_mask_used=False)
    finally:
        bank.close()


def _main():
    import argparse
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import sys

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Complete local diffusers FLUX.1-dev snapshot')
    p.add_argument('--backend-dir', default=str(Path(__file__).resolve().parent.parent / 'R80_RegionBalance'))
    p.add_argument('--image', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--output', required=True, help='Output preview PNG; metadata at .json')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--steps', type=int, default=28)
    p.add_argument('--inject', type=int, default=5)
    p.add_argument('--max-cache-mib', type=float, default=2048)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--guidance', type=float, default=5)
    args = p.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    for name, version in [('diffusers', '0.35.1'), ('transformers', '4.55.4'), ('accelerate', '1.10.1')]:
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'Use existing {name}=={version}; no installation attempted')
    if not Path(args.model, 'model_index.json').is_file():
        raise FileNotFoundError('Provide a complete local diffusers snapshot, not native flux-dev.safetensors')
    if args.size < 16 or args.size % 16:
        raise ValueError('--size must be a positive multiple of 16')
    if not torch.cuda.is_available():
        raise RuntimeError('Generation requires CUDA')
    sys.path.insert(0, str(Path(args.backend_dir).resolve()))
    from mint_backend import FluxBackend
    from diffusers import FluxPipeline
    from PIL import Image

    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    b = FluxBackend(pipe, args.size, args.size, args.steps, args.max_length)
    with Image.open(args.image) as image:
        y, ids = b.encode(image.convert('RGB'))
    src, tgt = b.embeddings(args.source), b.embeddings(args.target)
    job = dict(method='rfedit', steps=args.steps, rfedit_inject_steps=args.inject,
               rfedit_max_cache_mib=args.max_cache_mib, rfedit_tgt_guidance=args.guidance)
    z, stats = execute_external(b, y, ids, src, tgt, job,
        lambda i, n: print(f'RF-Edit {i}/{n}; NFE={b.nfe}', flush=True))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = b.decode(z)
    preview.thumbnail((512, 512))
    preview.save(output, format='PNG')
    stats.update(model_path=str(Path(args.model).resolve()), source=args.source,
                 target=args.target, input_image=str(Path(args.image).resolve()),
                 resolution=args.size, config=job,
                 versions={n: importlib.metadata.version(n) for n in ['torch', 'diffusers', 'transformers', 'accelerate']})
    output.with_suffix('.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    _main()
