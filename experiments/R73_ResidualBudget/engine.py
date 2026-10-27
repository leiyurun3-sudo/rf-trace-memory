"""RF editing with reference residual transfer and token-wise conflict projection."""
import torch
from attention import AttentionMemory


def rms(x):
    return float(x.float().square().mean().sqrt())


def project_correction(correction, direction):
    """Project c onto {v: dot(v,d)>=0}, independently per packed image token."""
    dot = (correction * direction).sum(-1, keepdim=True)
    norm2 = direction.square().sum(-1, keepdim=True)
    coefficient = torch.where(norm2 > 1e-12, dot.clamp_max(0) / norm2.clamp_min(1e-12), 0.)
    return correction - coefficient * direction, (dot < 0) & (norm2 > 1e-12)


@torch.inference_mode()
def execute(b, y, ids, src, tgt, controls, job, trajectory, tick, alignment):
    sigmas = b.schedule(y.shape[1])
    hs = [float(sigmas[k] - sigmas[k + 1]) for k in range(len(sigmas) - 1)]
    span = float(sigmas[0] - sigmas[-1])
    early = [(float(sigmas[k] + sigmas[k + 1]) / 2 - float(sigmas[-1])) / span for k in range(len(hs))]
    schedule = job.get('corr_schedule', 'constant')
    if schedule in ('budget_early', 'budget_late'):
        weights = job['corr_weights']
        if len(weights) != len(hs) or any(abs(a-b) > 1e-7 for a,b in zip(hs,job['budget_h'])):
            raise ValueError('R73 residual profile uses a different sigma grid')
    else:
        weights = {'constant': [1.] * len(hs), 'early': early, 'late': [1. - x for x in early]}[schedule]
    first, end = int(job['src_start']), min(int(job['src_steps']), len(hs))
    fade = [1. - (1. - job.get('gate_end', .3)) * (k - first) / max(1, end - first - 1) for k in range(first, end)]
    matched = sum(hs[k] * fade[k - first] for k in range(first, end)) / sum(hs[first:end])
    gate_kind = job.get('gate_schedule', 'constant')
    gate_values = [0.] * len(hs)
    for k in range(first, end):
        factor = {'constant': 1., 'fade': fade[k - first], 'fade_matched': matched}[gate_kind]
        gate_values[k] = float(job['query_write_weight']) * factor
    z = trajectory[0].to(b.device).float().clone()
    anchor = y.to(b.device).float()
    mask, allowed = (controls[k] for k in ('mask', 'allowed'))
    lam = controls['calibration'] * float(job['corr_lambda'])
    policy, mode = job['corr_policy'], job['mode']
    memory = None if mode == 'off' else AttentionMemory(b.pipe.transformer, z.shape[1], controls, job, alignment)
    logs, before, source_nfe, target_nfe = [], b.nfe, 0, 0
    try:
        for i in range(len(sigmas) - 1):
            sigma = float(sigmas[i])
            h = float(sigmas[i] - sigmas[i + 1])
            ref = trajectory[i].to(b.device).float()
            next_ref = trajectory[i + 1].to(b.device).float()
            path = (next_ref - ref) / h
            eta = job['eta'] * min(1., max(0., (sigma - .2) / .25))
            active = memory is not None and job['src_start'] <= i < job['src_steps']
            if memory is not None:
                memory.off()
            correction = torch.zeros_like(z)
            source_velocity = None
            if job['corr_lambda'] > 0 or active:
                if active:
                    memory.capture()
                source_field = b.field(ref, sigma, src, ids, job['guidance'])
                source_nfe += 1
                source_velocity = -(1 - eta) * source_field + eta * (anchor - ref) / sigma
                correction = path - source_velocity
            if active:
                memory.apply(i, deviation=(z - ref).square().mean(-1).sqrt()[:, None, :, None], gate=gate_values[i])
            target_field = b.field(z, sigma, tgt, ids, job['guidance'])
            target_nfe += 1
            changes = list(memory.actual) if active else []
            if memory is not None:
                memory.off()
            target_velocity = -(1 - eta) * target_field + eta * (anchor - z) / sigma
            applied = correction
            conflict_fraction, retained_norm, removed_rms = None, None, None
            if job['corr_lambda'] > 0:
                # Cross-state velocity difference; includes target memory when active.
                # It is an editing-direction proxy, not a semantic gradient.
                direction = target_velocity - source_velocity
                projected, conflict = project_correction(correction, direction)
                norm = correction.square().sum(dim=(1, 2), keepdim=True).sqrt()
                projected_norm = projected.square().sum(dim=(1, 2), keepdim=True).sqrt()
                ratio = torch.where(norm > 1e-12, projected_norm / norm.clamp_min(1e-12), torch.ones_like(norm))
                if policy == 'conflict':
                    applied = projected
                elif policy == 'norm_matched':
                    # Same instantaneous residual norm as the hypothetical projection
                    # at THIS arm's state, rather than a replay of another trajectory.
                    applied = ratio * correction
                elif policy != 'constant':
                    raise ValueError('Unknown correction policy: ' + policy)
                conflict_fraction = float(conflict.float().mean())
                retained_norm = float(ratio.mean())
                removed_rms = rms(correction - applied)
            step_lam = lam * weights[i]
            residual = target_velocity + step_lam * applied - path
            proposal = z + h * path + h * mask * residual
            z = torch.where(allowed > 0, proposal, next_ref)
            if not bool(torch.isfinite(z).all()):
                raise FloatingPointError('Nonfinite RF state')
            logs.append(dict(step=i + 1, sigma=sigma, h=h, eta=eta,
                             corr_lambda=float(job['corr_lambda']) * weights[i], corr_schedule=schedule, gate_schedule=gate_kind, scheduled_gate=gate_values[i] if active else 0., corr_policy=policy,
                             mode=mode, injection_active=active,
                             source_velocity_rms=None if source_velocity is None else rms(source_velocity),
                             source_path_rms=rms(path), target_velocity_rms=rms(target_velocity),
                             applied_correction_rms=rms(mask * step_lam * applied),
                             correction_removed_rms=removed_rms, conflict_fraction=conflict_fraction,
                             projected_norm_ratio=retained_norm,
                             residual_update_rms=rms(h * mask * residual),
                             source_path_distance=rms(z - next_ref), injection=changes))
            tick(i + 1, len(sigmas) - 1)
    finally:
        if memory is not None:
            memory.close()
    impulse = sum(row['h']*row['applied_correction_rms'] for row in logs)
    budget_error = abs(impulse-job['correction_budget_target'])/job['correction_budget_target']
    return z.cpu(), dict(correction_impulse=impulse, correction_budget_relative_error=budget_error, correction_budget_status='matched' if budget_error <= .01 else 'mismatch', nfe=b.nfe - before, source_nfe=source_nfe, target_nfe=target_nfe,
                         reference=None, correction_budget_h=sum(h*w*float(job['corr_lambda']) for h,w in zip(hs,weights)), gate_budget_h=sum(h*g for h,g in zip(hs,gate_values)), steps=logs)
