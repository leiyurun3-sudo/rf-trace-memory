"""R74: condition-separated residual control with stepwise matched RMS budgets."""
import torch
from attention import AttentionMemory


def rms(x):
    return float(x.float().square().mean().sqrt())


@torch.inference_mode()
def execute(b, y, ids, src, tgt, controls, job, trajectory, tick, alignment):
    sigmas = b.schedule(y.shape[1])
    z = trajectory[0].to(b.device).float().clone()
    anchor = y.to(b.device).float()
    mask, allowed = controls['mask'], controls['allowed']
    lam = float(job['corr_lambda'])
    kind = job.get('residual_kind', 'full')
    alpha = float(job.get('condition_weight', 1.))
    memory = None if job['mode'] == 'off' else AttentionMemory(b.pipe.transformer, z.shape[1], controls, job, alignment)
    before = b.nfe
    source_nfe = target_nfe = null_nfe = 0
    logs = []
    target_budget = 0.
    for_memory = float(job['query_write_weight'])
    try:
        for i in range(len(sigmas) - 1):
            sigma, h = float(sigmas[i]), float(sigmas[i] - sigmas[i+1])
            ref = trajectory[i].to(b.device).float()
            next_ref = trajectory[i+1].to(b.device).float()
            path = (next_ref-ref)/h
            eta = float(job['eta'])*min(1., max(0., (sigma-.2)/.25))
            active = memory is not None and job['src_start'] <= i < job['src_steps']
            if memory is not None:
                memory.off()
            correction = torch.zeros_like(z)
            source_velocity = None
            a_rms = b_rms = cosine_ab = None
            gain, clipped = 1., False
            # Source capture retains its source prompt. The null field is evaluated
            # separately, memory-off, at the same node, time and guidance.
            if lam > 0 or active:
                if active:
                    memory.capture()
                fs = b.field(ref, sigma, src, ids, job['guidance'])
                source_nfe += 1
                source_velocity = -(1-eta)*fs + eta*(anchor-ref)/sigma
                correction = path-source_velocity
                if memory is not None:
                    memory.off()
            applied = correction
            if lam > 0 and kind == 'split':
                f0 = b.field(ref, sigma, b.empty_embeddings, ids, job['guidance'])
                null_nfe += 1
                u0 = -(1-eta)*f0 + eta*(anchor-ref)/sigma
                a, component_b = path-u0, u0-source_velocity
                a_rms, b_rms = rms(a), rms(component_b)
                product = a_rms*b_rms
                cosine_ab = float((a*component_b).mean())/product if product > 1e-12 else None
                applied = correction if alpha == 1. else a+alpha*component_b
                if job.get('match_residual_rms', True):
                    numerator, denominator = rms(correction), rms(applied)
                    requested = numerator/max(denominator, 1e-12)
                    gain = min(float(job.get('max_residual_gain', 4.)), requested)
                    clipped = requested > gain
                    applied = gain*applied
            elif kind not in ('full', 'split'):
                raise ValueError('Unknown residual_kind: '+kind)
            if active:
                memory.apply(i, deviation=(z-ref).square().mean(-1).sqrt()[:, None, :, None], gate=for_memory)
            ft = b.field(z, sigma, tgt, ids, job['guidance'])
            target_nfe += 1
            changes = list(memory.actual) if active else []
            if memory is not None:
                memory.off()
            ut = -(1-eta)*ft + eta*(anchor-z)/sigma
            weighted = controls['calibration']*lam*applied
            full_weighted = controls['calibration']*lam*correction
            target_budget += h*rms(mask*full_weighted)
            residual = ut+weighted-path
            proposal = z+h*path+h*mask*residual
            z = torch.where(allowed > 0, proposal, next_ref)
            if not bool(torch.isfinite(z).all()):
                raise FloatingPointError('Nonfinite RF state')
            logs.append(dict(step=i+1, sigma=sigma, h=h, eta=eta,
                corr_lambda=lam, condition_weight=alpha, residual_kind=kind,
                a_rms=a_rms, condition_component_rms=b_rms, component_cosine=cosine_ab,
                full_correction_rms=rms(correction), residual_gain=gain, gain_clipped=clipped,
                reference_correction_rms=rms(mask*full_weighted),
                applied_correction_rms=rms(mask*weighted),
                source_velocity_rms=None if source_velocity is None else rms(source_velocity),
                target_velocity_rms=rms(ut), source_path_rms=rms(path),
                source_path_distance=rms(z-next_ref), injection_active=active,
                scheduled_gate=for_memory if active else 0., injection=changes))
            tick(i+1, len(sigmas)-1)
    finally:
        if memory is not None:
            memory.close()
    impulse = sum(row['h']*row['applied_correction_rms'] for row in logs)
    relative = abs(impulse-target_budget)/target_budget if target_budget > 1e-12 else 0.
    return z.cpu(), dict(nfe=b.nfe-before, source_nfe=source_nfe, target_nfe=target_nfe,
        null_nfe=null_nfe, correction_impulse=impulse, budget_target=target_budget,
        correction_budget_relative_error=relative,
        correction_budget_status='matched' if relative <= .01 else 'gain_limited',
        correction_budget_h=sum(row['h']*lam for row in logs),
        gate_budget_h=sum(row['h']*row['scheduled_gate'] for row in logs), steps=logs)
