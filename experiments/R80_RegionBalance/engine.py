"""R80: region balance, soft memory release and source-only support controls."""
import math
import torch.nn.functional as F
import torch
from attention import AttentionMemory


def rms(x):
    return float(x.float().square().mean().sqrt())


def roi_conditional_rms(x, roi):
    mass = float(roi.float().sum())
    return None if mass <= 1e-12 else float(((x.float().square()*roi.float()).sum()/mass).sqrt())


def spectral_candidate(correction, job, route_mask=None):
    if job.get('correction_shape') == 'roi_uniform':
        if route_mask is None or route_mask.shape != correction.shape:
            raise ValueError('ROI uniform candidate requires a packed route mask')
        return route_mask * correction, None, None
    # FLUX packs each 2x2 latent patch into channels. Filter the actual spatial
    # latent grid, then restore the original token/channel layout.
    b, n, d = correction.shape
    side = math.isqrt(n)
    if side * side != n or d % 4:
        raise ValueError('Spectral arm requires a square FLUX packed latent')
    channels = d // 4
    grid = correction.reshape(b, side, side, channels, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, channels, side*2, side*2)
    kernel = int(job['spectral_kernel'])
    low = F.avg_pool2d(F.pad(grid, (kernel//2,)*4, mode='replicate'), kernel, stride=1)
    high = grid - low
    if route_mask is not None:
        if route_mask.shape != correction.shape:
            raise ValueError('Packed ROI route mask must match correction')
        route_grid = route_mask.reshape(b, side, side, channels, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, channels, side*2, side*2)
    mixed = float(job['spectral_low_weight']) * low + (high if route_mask is None else route_grid * high)
    packed = mixed.reshape(b, channels, side, 2, side, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, n, d)
    return packed, rms(low), rms(grid-low)


@torch.inference_mode()
def execute(b, y, ids, src, tgt, controls, job, trajectory, tick, alignment, budget_profile=None, raw_profile=None):
    sigmas = b.schedule(y.shape[1])
    z = trajectory[0].to(b.device).float().clone()
    anchor = y.to(b.device).float()
    mask, allowed = controls['mask'], controls['allowed']
    lam = float(job['corr_lambda'])
    beta = float(job.get('edit_gain', 1.))
    shaped = job.get('correction_shape', 'uniform') in ('spectral', 'roi_uniform')
    budget_mode = job.get('correction_budget_mode', 'external_u25' if shaped else 'none')
    if shaped and budget_mode not in ('external_u25', 'online_same_state', 'local_coefficient'):
        raise ValueError('Shaped correction requires explicit budget mode')
    if shaped and budget_mode == 'external_u25' and (budget_profile is None or len(budget_profile) != len(sigmas)-1):
        raise ValueError('Missing matching same-run U25_KV per-step budget profile')
    if budget_mode in ('online_same_state', 'local_coefficient') and (budget_profile is not None or job.get('budget_reference_id')):
        raise ValueError('R79 online/local modes do not use external profiles')
    route_mask = controls.get('route_mask')
    route_mask64 = controls.get('route_mask64')
    if bool(job.get('route_mask')) != (route_mask is not None) or (route_mask is not None and not shaped):
        raise ValueError('Route mask requires an explicit shaped correction arm')
    route_info = dict(route_roi=job.get('route_roi'), route_shift_xy=job.get('route_shift_xy'), route_mask=job.get('route_mask'), route_mask_role=job.get('route_mask_role'),
        route_mask_sampling=job.get('route_mask_sampling'),
        route_mask_source=job.get('route_mask_source'), route_feather_px=job.get('route_feather_px'),
        route_feather_coordinate_basis=512 if route_mask is not None else None,
        route_mask_coverage=None if route_mask64 is None else float(route_mask64.mean()),
        route_support_cells=None if route_mask is None else int((route_mask64 > 0).sum()),
        route_total_cells=None if route_mask is None else route_mask64.numel(),
        route_support_fraction=None if route_mask is None else float((route_mask64 > 0).float().mean()),
        route_mask_mean=None if route_mask is None else float(route_mask64.mean()),
        route_changes_update_mask=False)
    query_mask = controls['query_mask']
    query_info = dict(query_selection=job['query_selection'],
        target_query_mask_mean=float(query_mask.float().mean()),
        target_query_mask_support_fraction=float((query_mask > 0).float().mean()),
        source_kv_pool_tokens=z.shape[1], source_kv_pool_unchanged=True)
    kind = job.get('residual_kind', 'full')
    alpha = float(job.get('condition_weight', 1.))
    tau = float(job.get('transport_tau', 0.))
    if not 0. <= tau <= 1. or (tau and kind != 'transport'):
        raise ValueError('transport_tau must be in [0,1] with residual_kind=transport')
    memory = None if job['mode'] == 'off' else AttentionMemory(b.pipe.transformer, z.shape[1], controls, job, alignment)
    before = b.nfe
    source_nfe = target_nfe = null_nfe = source_current_nfe = 0
    native_target_nfe = 0
    logs = []
    target_budget = 0.
    for_memory = float(job['query_write_weight'])
    try:
        for i in range(len(sigmas) - 1):
            step_nfe_before = b.nfe
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
            current_source_velocity = None
            state_shift_rms = None
            a_rms = b_rms = cosine_ab = None
            gain, clipped = 1., False
            # Source capture retains its source prompt. The null field is evaluated
            # separately, memory-off, at the same node, time and guidance.
            # At tau=1 the reference field is only needed to capture memory.
            if (lam > 0 and tau < 1.) or active:
                if active:
                    memory.capture()
                fs = b.field(ref, sigma, src, ids, job['guidance'])
                source_nfe += 1
                source_velocity = -(1-eta)*fs + eta*(anchor-ref)/sigma
                correction = path-source_velocity
                if memory is not None:
                    memory.off()
            if lam > 0 and kind == 'transport' and tau > 0.:
                # Keep source-memory capture on r_i; evaluate the correction field
                # at z_i with memory disabled so no target prompt enters this call.
                f_current = b.field(z, sigma, src, ids, job['guidance'])
                source_current_nfe += 1
                current_source_velocity = -(1-eta)*f_current + eta*(anchor-z)/sigma
                if source_velocity is not None:
                    state_shift_rms = rms(current_source_velocity-source_velocity)
                mixed_source = current_source_velocity if tau == 1. else (
                    (1.-tau)*source_velocity+tau*current_source_velocity)
                correction = path-mixed_source
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
            elif kind not in ('full', 'split', 'transport'):
                raise ValueError('Unknown residual_kind: '+kind)
            native_ut = None
            if beta != 1. and active:
                # Separate prompt contrast from memory: amplify native text
                # difference, leaving the memory contribution at its original gain.
                native_ft = b.field(z, sigma, tgt, ids, job['guidance'])
                native_target_nfe += 1
                native_ut = -(1-eta)*native_ft + eta*(anchor-z)/sigma
            if active:
                memory.apply(i, deviation=(z-ref).square().mean(-1).sqrt()[:, None, :, None], gate=for_memory)
            ft = b.field(z, sigma, tgt, ids, job['guidance'])
            target_nfe += 1
            changes = list(memory.actual) if active else []
            if memory is not None:
                memory.off()
            ut = -(1-eta)*ft + eta*(anchor-z)/sigma
            if native_ut is None and not active:
                native_ut = ut
            direction = None if native_ut is None or current_source_velocity is None else native_ut-current_source_velocity
            memory_delta = None if native_ut is None else ut-native_ut
            low_rms = high_rms = desired_rms = None
            weighted = controls['calibration']*lam*applied
            if shaped:
                if budget_mode == 'external_u25':
                    baseline_step = budget_profile[i]
                    if abs(float(baseline_step['h'])-h) > 1e-8 or abs(float(baseline_step['sigma'])-sigma) > 1e-8:
                        raise ValueError('Budget reference schedule differs at step '+str(i))
                    desired_rms = float(baseline_step['applied_correction_rms'])
                elif budget_mode == 'online_same_state':
                    desired_rms = .25 * rms(controls['calibration']*mask*correction)
                candidate, low_rms, high_rms = spectral_candidate(correction, job, route_mask=route_mask)
                candidate = controls['calibration']*candidate
                candidate_rms = rms(mask*candidate)
                if budget_mode == 'local_coefficient':
                    gain = lam  # Exactly .25*M*c: no global RMS renormalization.
                else:
                    if candidate_rms <= 1e-12 and desired_rms > 1e-12:
                        raise FloatingPointError('Nonzero matched budget with zero ROI residual')
                    gain = desired_rms/max(candidate_rms, 1e-12)
                requested_gain = gain
                if job.get('region_gain_cap') is not None:
                    gain = min(gain, float(job['region_gain_cap']))
                    clipped = gain < requested_gain
                weighted = candidate*gain
            full_weighted = controls['calibration']*lam*correction
            if budget_mode != 'local_coefficient':
                target_budget += h*(desired_rms if shaped else rms(mask*full_weighted))
            edited_velocity = ut if beta == 1. else ut+(beta-1.)*direction
            residual = edited_velocity+weighted-path
            proposal = z+h*path+h*mask*residual
            z = torch.where(allowed > 0, proposal, next_ref)
            if not bool(torch.isfinite(z).all()):
                raise FloatingPointError('Nonfinite RF state')
            distance = rms(z-next_ref)
            raw_pair = {}
            if raw_profile is not None:
                denom = raw_profile['steps'][i]
                if (int(denom.get('step_index', int(denom['step'])-1)) == i
                        and abs(float(denom['sigma'])-sigma) <= 1e-8
                        and abs(float(denom['h'])-h) <= 1e-8):
                    value = float(denom['source_path_distance'])
                    if math.isfinite(value) and value > 1e-12:
                        raw_pair = dict(source_path_distance_raw_denominator=value,
                            source_path_distance_raw_ratio=distance/value,
                            source_path_distance_raw_reference_id=raw_profile['id'])
            logs.append(dict(step=i+1, step_index=i, sigma=sigma, h=h, eta=eta,
                nfe_step=b.nfe-step_nfe_before, nfe_edit_cumulative=b.nfe-before,
                corr_lambda=lam, condition_weight=alpha, residual_kind=kind,
                region_gain=gain,
                current_correction_global_rms=rms(controls['calibration']*mask*correction),
                current_correction_roi_conditional_rms=roi_conditional_rms(controls['calibration']*mask*correction, route_mask),
                candidate_global_rms=candidate_rms,
                applied_correction_global_rms=rms(mask*weighted),
                applied_correction_roi_conditional_rms=roi_conditional_rms(mask*weighted, route_mask),
                query_selection=job['query_selection'],
                target_query_mask_mean=query_info['target_query_mask_mean'],
                target_query_mask_support_fraction=query_info['target_query_mask_support_fraction'],
                query_gate_mean=sum(v['query_gate_mean'] for v in changes)/len(changes) if changes else 0.,
                query_gate_support_fraction=sum(v['query_gate_support_fraction'] for v in changes)/len(changes) if changes else 0.,
                spectral_kernel=job.get('spectral_kernel'), spectral_low_weight=job.get('spectral_low_weight'),
                route_support_fraction=route_info['route_support_fraction'],
                route_mask_mean=route_info['route_mask_mean'],
                route_mask_coverage=route_info['route_mask_coverage'],
                route_mask_source=route_info['route_mask_source'], route_feather_px=route_info['route_feather_px'],
                edit_gain=beta, correction_shape=job.get('correction_shape', 'uniform'),
                native_prompt_difference_rms=None if direction is None else rms(direction),
                memory_delta_rms=None if memory_delta is None else rms(memory_delta),
                correction_low_rms=low_rms, correction_high_rms=high_rms,
                matched_step_budget_rms=desired_rms if budget_mode == 'external_u25' else None,
                desired_applied_correction_rms=desired_rms,
                correction_budget_mode=budget_mode,
                external_trajectory_budget_matched=shaped and budget_mode=='external_u25',
                same_state_uniform25_rms=.25*rms(controls['calibration']*mask*correction) if budget_mode=='online_same_state' else None,
                transport_tau=tau, source_state_shift_rms=state_shift_rms,
                source_current_velocity_rms=None if current_source_velocity is None else rms(current_source_velocity),
                a_rms=a_rms, condition_component_rms=b_rms, component_cosine=cosine_ab,
                full_correction_rms=rms(correction), residual_gain=gain, gain_clipped=clipped,
                reference_correction_rms=rms(mask*full_weighted),
                applied_correction_rms=rms(mask*weighted),
                source_velocity_rms=None if source_velocity is None else rms(source_velocity),
                target_velocity_rms=rms(ut), source_path_rms=rms(path),
                source_path_distance=distance, source_path_distance_absolute_rms=distance,
                **raw_pair, injection_active=active,
                scheduled_gate=for_memory if active else 0., injection=changes))
            tick(i+1, len(sigmas)-1)
    finally:
        if memory is not None:
            memory.close()
    impulse = sum(row['h']*row['applied_correction_rms'] for row in logs)
    relative = abs(impulse-target_budget)/target_budget if target_budget > 1e-12 else 0.
    shifts = [row['source_state_shift_rms'] for row in logs if row['source_state_shift_rms'] is not None]
    transported = kind == 'transport' and tau > 0.
    return z.cpu(), dict(nfe=b.nfe-before, source_nfe=source_nfe+source_current_nfe, target_nfe=target_nfe+native_target_nfe,
        **route_info, **query_info, correction_budget_mode=budget_mode,
        region_gain_mean=sum(r['region_gain'] for r in logs)/len(logs),
        applied_correction_global_rms_mean=sum(r['applied_correction_global_rms'] for r in logs)/len(logs),
        applied_correction_roi_conditional_rms_mean=sum(r['applied_correction_roi_conditional_rms'] for r in logs)/len(logs),
        query_gate_mean_over_steps=sum(r['query_gate_mean'] for r in logs)/len(logs),
        external_trajectory_budget_matched=shaped and budget_mode=='external_u25', memory_window=[job['src_start'], job['src_steps']],
        memory_window_indexing='zero_based_start_inclusive_end_exclusive',
        memory_active_steps=sum(row['injection_active'] for row in logs),
        spectral_kernel=job.get('spectral_kernel'), spectral_low_weight=job.get('spectral_low_weight'),
        native_target_nfe=native_target_nfe, edit_gain=beta, correction_shape=job.get("correction_shape", "uniform"),
        source_reference_nfe=source_nfe, source_current_nfe=source_current_nfe,
        transport_tau=tau, source_state_shift_mean_rms=sum(shifts)/len(shifts) if shifts else None,
        source_state_shift_steps=len(shifts),
        null_nfe=null_nfe, correction_impulse=impulse, budget_target=target_budget if budget_mode != 'local_coefficient' and (shaped or not transported) else None,
        correction_budget_relative_error=relative if budget_mode != 'local_coefficient' and (shaped or not transported) else None,
        correction_budget_status=('capped_online_budget' if job.get('region_gain_cap') is not None
            else 'local_coefficient_unnormalized' if budget_mode=='local_coefficient'
            else ('self_normalized_same_state_rms' if relative <= 1e-5 else 'budget_mismatch')),
        correction_budget_h=sum(row['h']*lam for row in logs),
        scheduled_gate_budget_h=sum(row['h']*row['scheduled_gate'] for row in logs),
        gate_budget_h=sum(row['h']*row['query_gate_mean'] for row in logs), steps=logs)
