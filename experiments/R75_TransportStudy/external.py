"""FlowEdit FLUX, adapted from the MIT-licensed official implementation.
Source and full license: external_sources.md, commit 2620a9364f0f9c21368c36266e6200ff250273ec.
Inputs src and tgt are FluxBackend.embeddings() tuples; y is packed encoded latent.
"""
import torch


@torch.inference_mode()
def execute_external(b, y, ids, src, tgt, job, tick):
    if job.get('method', 'flowedit').lower() != 'flowedit':
        raise ValueError('external adapter supports method=flowedit only')
    sigmas = b.schedule(y.shape[1])
    steps = len(sigmas) - 1
    n_max = int(job.get('flowedit_n_max', 24))
    n_min = int(job.get('flowedit_n_min', 0))
    n_avg = int(job.get('flowedit_n_avg', 1))
    source_guidance = float(job.get('flowedit_src_guidance', 1.5))
    target_guidance = float(job.get('flowedit_tgt_guidance', 5.5))
    if not (0 <= n_min <= n_max <= steps and n_avg >= 1):
        raise ValueError('FlowEdit requires 0 <= n_min <= n_max <= schedule steps and n_avg >= 1')
    seed = int(job.get('seed', 10))
    generator = torch.Generator(device=b.device).manual_seed(seed)
    x = y.to(b.device, dtype=torch.float32)
    edit = x.clone()
    target = None
    start_nfe = b.nfe
    source_nfe = target_nfe = 0
    for i in range(steps):
        remaining = steps - i
        if remaining <= n_max:
            t, next_t = float(sigmas[i]), float(sigmas[i + 1])
            if remaining > n_min:
                delta = torch.zeros_like(x)
                for _ in range(n_avg):
                    noise = torch.randn(x.shape, generator=generator, device=b.device, dtype=x.dtype)
                    source = (1.0 - t) * x + t * noise
                    target_input = edit + source - x
                    vs = b.field(source, t, src, ids, source_guidance)
                    source_nfe += int(source.shape[0])
                    vt = b.field(target_input, t, tgt, ids, target_guidance)
                    target_nfe += int(target_input.shape[0])
                    delta += (vt - vs) / n_avg
                edit = edit + (next_t - t) * delta
            else:
                if target is None:
                    noise = torch.randn(x.shape, generator=generator, device=b.device, dtype=x.dtype)
                    source = (1.0 - t) * x + t * noise
                    target = edit + source - x
                vt = b.field(target, t, tgt, ids, target_guidance)
                target_nfe += int(target.shape[0])
                target = target + (next_t - t) * vt
        if tick is not None:
            tick(i + 1, steps)
    output = edit if n_min == 0 else target
    return output.cpu(), dict(method='flowedit', nfe=int(b.nfe - start_nfe),
        source_nfe=source_nfe, target_nfe=target_nfe, nfe_inversion=0,
        steps=steps, n_max=n_max, n_min=n_min, n_avg=n_avg, seed=seed,
        src_guidance=source_guidance, tgt_guidance=target_guidance,
        initial_sigma=float(sigmas[0]), sigma_schedule=[float(s) for s in sigmas])
