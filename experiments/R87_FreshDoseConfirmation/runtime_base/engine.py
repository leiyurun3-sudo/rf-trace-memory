"""R80: region balance, soft memory release and source-only support controls."""
import math
import torch.nn.functional as F
import torch
from attention import AttentionMemory
from controls import pack
from r86_compat import validate_profile


def rms(x):
    return float(x.float().square().mean().sqrt())


def roi_conditional_rms(x, roi):
    if roi is None:
        return None
    mass = float(roi.float().sum())
    return None if mass <= 1e-12 else float(((x.float().square()*roi.float()).sum()/mass).sqrt())


def spectral_candidate(correction, job, route_mask=None):
    if job.get('correction_shape') in ('roi_uniform', 'request_energy_global'):
        if route_mask is None or route_mask.shape != correction.shape:
            raise ValueError('ROI uniform candidate requires a packed route mask')
        floor = float(job.get('route_floor', 0.0))
        if not 0.0 <= floor <= 1.0:
            raise ValueError('route_floor must be in [0,1]')
        effective_mask = floor + (1.0 - floor) * route_mask
        return effective_mask * correction, None, None
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
def prompt_response_route(b, ids, src, tgt, controls, job, trajectory, sigmas):
    """Native prompt contrast on identical fixed source-trajectory states."""
    requested = job.get('response_sigmas', [0.25, 0.50, 0.75])
    if len(requested) != 3:
        raise ValueError('Prompt-response routing requires three sigma targets')
    indices = [min(range(len(sigmas)-1), key=lambda i: abs(float(sigmas[i])-v))
               for v in requested]
    if len(set(indices)) != 3:
        raise ValueError('Prompt-response routing requires three distinct scheduler nodes')
    side = int(job.get('resolution', 512)) // 8
    existing_route = controls.get('route_mask64')
    if side % 2 or (existing_route is not None and existing_route.shape != (1, 1, side, side)):
        raise ValueError('Prompt-response routing requires a square even latent grid')
    maps, nodes = [], []
    start_nfe = b.nfe
    probe_source_nfe = probe_target_nfe = 0
    for requested_sigma, i in zip(requested, indices):
        sigma = float(sigmas[i])
        ref = trajectory[i].to(b.device).float()
        count = b.nfe
        source = b.field(ref, sigma, src, ids, job['guidance'])
        probe_source_nfe += int(b.nfe-count)
        count = b.nfe
        target = b.field(ref, sigma, tgt, ids, job['guidance'])
        probe_target_nfe += int(b.nfe-count)
        batch, tokens, packed_channels = source.shape
        half = side // 2
        if batch != 1 or tokens != half * half or packed_channels % 4 or target.shape != source.shape:
            raise ValueError('Prompt-response field does not match packed FLUX grid')
        channels = packed_channels // 4
        delta = (target-source).reshape(batch, half, half, channels, 2, 2)
        delta = delta.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, side, side)
        magnitude = delta.float().square().mean(dim=1, keepdim=True).sqrt()
        lo = torch.quantile(magnitude.flatten(), 0.10)
        hi = torch.quantile(magnitude.flatten(), 0.90)
        span = hi-lo
        degenerate = bool(span <= 1e-12)
        response = torch.zeros_like(magnitude) if degenerate else ((magnitude-lo)/span).clamp(0., 1.)
        maps.append(response)
        nodes.append(dict(requested_sigma=requested_sigma, step_index=i, sigma=sigma,
                          q10=float(lo), q90=float(hi), degenerate=degenerate))
    edit_response = torch.stack(maps).mean(dim=0)
    route = 1.0-edit_response
    controls['edit_response64'] = edit_response
    floor = float(job.get('memory_response_floor', 0.25)) if job.get('memory_response_mode') is not None else float(job.get('route_floor', 0.0))
    if not 0.0 <= floor <= 1.0:
        raise ValueError('Memory response floor must be in [0,1]')
    controls['effective_route_mask64'] = floor + (1.0-floor)*route
    if existing_route is not None:
        if controls.get('route_mask') is None:
            raise ValueError('Packed route mask is missing')
        existing_route.copy_(route)
        controls['route_mask'].copy_(pack(route.expand(-1, channels, -1, -1)))
    return dict(route_probe_nfe=int(b.nfe-start_nfe), route_probe_source_nfe=probe_source_nfe,
                route_probe_target_nfe=probe_target_nfe, route_probe_nodes=nodes,
                route_probe_guidance=float(job['guidance']),
                route_probe_memory='off_before_attention_hooks',
                route_probe_estimator='channel_rms_q10_q90_clip_mean3',
                route_probe_degenerate_count=sum(n['degenerate'] for n in nodes),
                route_edit_response_mean=float(edit_response.mean()),
                route_uses_eval_roi=False)


@torch.inference_mode()
def execute(b, y, ids, src, tgt, controls, job, trajectory, tick, alignment, budget_profile=None, raw_profile=None, probe_cache=None):
    sigmas = b.schedule(y.shape[1])
    z = trajectory[0].to(b.device).float().clone()
    anchor = y.to(b.device).float()
    mask, allowed = controls['mask'], controls['allowed']
    lam = float(job['corr_lambda'])
    beta = float(job.get('edit_gain', 1.))
    shape = job.get('correction_shape', 'uniform')
    request_global = shape == 'request_energy_global'
    shaped = shape in ('spectral', 'roi_uniform', 'request_energy_global')
    budget_mode = job.get('correction_budget_mode', 'external_u25' if shaped and not request_global else 'none')
    r86_fixed = budget_mode == 'r86_fixed_profile'
    if bool(job.get('r86_arm')) != r86_fixed:
        raise ValueError('R86 arm and fixed-profile mode must be specified together')
    if r86_fixed:
        validate_profile(job, budget_profile, sigmas)
        if not bool((mask == 1).all()) or not bool((allowed > 0).all()):
            raise ValueError('R86 requires full update support')
    if request_global:
        if budget_mode != 'none':
            raise ValueError('request_energy_global requires correction_budget_mode=none')
        if not job.get('route_mask') or job.get('route_mask_source') != 'prompt_response_reference':
            raise ValueError('request_energy_global requires prompt-response route_mask')
        if budget_profile is not None or job.get('budget_reference_id'):
            raise ValueError('request_energy_global uses its own current state, not an external budget profile')
        if job.get('region_gain_cap') is not None:
            raise ValueError('request_energy_global RMS matching does not use region_gain_cap')
    if shaped and not request_global and budget_mode not in ('external_u25', 'online_same_state', 'local_coefficient', 'r86_fixed_profile'):
        raise ValueError('Shaped correction requires explicit budget mode')
    if shaped and budget_mode == 'external_u25' and (budget_profile is None or len(budget_profile) != len(sigmas)-1):
        raise ValueError('Missing matching same-run U25_KV per-step budget profile')
    if budget_mode in ('online_same_state', 'local_coefficient') and (budget_profile is not None or job.get('budget_reference_id')):
        raise ValueError('R79 online/local modes do not use external profiles')
    before = b.nfe
    probe_info = dict(route_probe_nfe=0, route_probe_source_nfe=0, route_probe_target_nfe=0,
                      route_probe_nodes=[], route_uses_eval_roi=False)
    memory_response_mode = job.get('memory_response_mode')
    if memory_response_mode not in (None, 'spatial', 'mean'):
        raise ValueError('Unknown memory_response_mode')
    if memory_response_mode is not None and job.get('route_mask_source') != 'prompt_response_reference':
        raise ValueError('Memory response requires prompt_response_reference probes')
    if job.get('route_mask_source') == 'prompt_response_reference':
        routed = bool(job.get('route_mask'))
        if routed and shape not in ('roi_uniform', 'request_energy_global'):
            raise ValueError('Prompt-response correction routing requires roi_uniform or request_energy_global')
        if request_global and (controls.get('route_mask64') is None or controls.get('route_mask') is None):
            raise ValueError('request_energy_global requires latent and packed probe route masks')
        if not routed and (memory_response_mode is None or job.get('correction_shape', 'uniform') != 'uniform'):
            raise ValueError('Unrouted prompt-response requires uniform correction and memory response')
        if routed and float(job.get('route_floor', 0.0)) != 0.25:
            raise ValueError('Prompt-response routing requires route_floor=0.25')
        if probe_cache is not None and 'response' in probe_cache:
            # Only the fixed-reference native response is reused, never a field at z_i.
            edit_response = probe_cache['response'].to(b.device).float().clone()
            if tuple(edit_response.shape) != (1,1,int(job.get('resolution',512))//8,int(job.get('resolution',512))//8) or not bool(torch.isfinite(edit_response).all()):
                raise ValueError('Invalid cached request response')
            route = 1.0 - edit_response
            controls['edit_response64'] = edit_response
            controls['effective_route_mask64'] = float(job.get('route_floor',0.0)) + (1.0-float(job.get('route_floor',0.0)))*route
            controls['route_mask64'].copy_(route)
            channels = controls['route_mask'].shape[-1] // 4
            controls['route_mask'].copy_(pack(route.expand(-1,channels,-1,-1)))
            probe_info = dict(probe_cache['info'])
            probe_info.update(route_probe_nfe=0, route_probe_source_nfe=0, route_probe_target_nfe=0)
            probe_info['route_probe_reused'] = True
            probe_info['route_probe_standalone_nfe'] = int(probe_cache['standalone_nfe'])
        else:
            probe_info = prompt_response_route(b, ids, src, tgt, controls, job, trajectory, sigmas)
            probe_info['route_probe_reused'] = False
            probe_info['route_probe_standalone_nfe'] = int(probe_info['route_probe_nfe'])
            if probe_cache is not None:
                probe_cache['response'] = controls['edit_response64'].detach().cpu().clone()
                probe_cache['info'] = dict(probe_info)
                probe_cache['standalone_nfe'] = int(probe_info['route_probe_nfe'])
    if memory_response_mode is not None:
        if job.get('query_selection') != 'all_image_queries':
            raise ValueError('Memory response requires all image queries')
        protection = controls['effective_route_mask64']
        token_protection = F.avg_pool2d(protection, 2, 2).flatten(2).transpose(1, 2)
        query_gate = token_protection if memory_response_mode == 'spatial' else torch.ones_like(token_protection)*token_protection.mean()
        if controls['query_mask'].shape != query_gate.shape:
            raise ValueError('Memory response query mask does not match image tokens')
        controls['query_mask'].copy_(query_gate)

    if job.get('r86_arm') or job.get('arm') == 'REQ':
        controls['r86_E_original'] = controls['edit_response64'].detach().clone()
        controls['r86_P_original'] = controls['effective_route_mask64'].detach().clone()
    if job.get('r86_arm') == 'SHIFT':
        route64 = controls['route_mask64']
        if route64.shape != (1, 1, 64, 64): raise ValueError('R86 expects 64x64 latent response grid')
        shifted = torch.roll(route64, shifts=(32, 32), dims=(-2, -1))
        if not torch.equal(torch.sort(route64.flatten()).values, torch.sort(shifted.flatten()).values): raise RuntimeError('Shift histogram mismatch')
        controls['route_mask64'].copy_(shifted)
        channels = controls['route_mask'].shape[-1] // 4
        controls['route_mask'].copy_(pack(shifted.expand(-1, channels, -1, -1)))
        controls['effective_route_mask64'] = 0.25 + 0.75*shifted
    if job.get('r86_arm') == 'PERM':
        from spatial_controls import permutation, permute_grid
        indices = permutation(job['case'])
        permuted = permute_grid(controls['route_mask64'], indices)
        controls['r87_block_permutation'] = indices
        controls['r87_E_permuted'] = permute_grid(controls['r86_E_original'], indices)
        controls['route_mask64'].copy_(permuted)
        channels = controls['route_mask'].shape[-1] // 4
        controls['route_mask'].copy_(pack(permuted.expand(-1, channels, -1, -1)))
        controls['effective_route_mask64'] = 0.25 + 0.75*permuted
    route_mask = controls.get('route_mask')
    route_mask64 = controls.get('route_mask64')
    if bool(job.get('route_mask')) != (route_mask is not None) or (route_mask is not None and not shaped):
        raise ValueError('Route mask requires an explicit shaped correction arm')
    floor = float(job.get('route_floor', 0.0))
    control_kind = ('request_energy_global_same_state' if request_global else
                    'request_local_release' if shape == 'roi_uniform' and budget_mode == 'local_coefficient'
                    and job.get('route_mask_source') == 'prompt_response_reference' else None)
    if r86_fixed:
        control_kind = 'actual_profile_' + job['r86_arm'].lower()
    effective_route = None if route_mask64 is None else floor + (1.0-floor)*route_mask64
    route_info = dict(route_floor=floor, effective_route_mask_mean=None if effective_route is None else float(effective_route.mean()), effective_route_support_fraction=None if effective_route is None else float((effective_route > 0).float().mean()), route_roi=job.get('route_roi'), route_shift_xy=job.get('route_shift_xy'), route_mask=job.get('route_mask'), route_mask_role=job.get('route_mask_role'),
        route_mask_sampling=job.get('route_mask_sampling'),
        route_mask_source=job.get('route_mask_source'), route_feather_px=job.get('route_feather_px'),
        route_feather_coordinate_basis=512 if job.get('route_mask_source') in ('fixed_pattern_roi', 'fixed_center_box', 'shifted_pattern_roi') else None,
        route_mask_coverage=None if route_mask64 is None else float(route_mask64.mean()),
        route_support_cells=None if route_mask is None else int((route_mask64 > 0).sum()),
        route_total_cells=None if route_mask is None else route_mask64.numel(),
        route_support_fraction=None if route_mask is None else float((route_mask64 > 0).float().mean()),
        route_mask_mean=None if route_mask is None else float(route_mask64.mean()),
        route_changes_update_mask=False)
    query_mask = controls['query_mask']
    query_info = dict(query_selection=job['query_selection'],
        memory_response_mode=memory_response_mode,
        memory_response_floor=(float(job.get('memory_response_floor', 0.25)) if memory_response_mode else None),
        memory_query_gate_mean=(float(query_mask.float().mean()) if memory_response_mode else None),
        memory_query_gate_min=(float(query_mask.float().min()) if memory_response_mode else None),
        memory_query_gate_max=(float(query_mask.float().max()) if memory_response_mode else None),
        memory_mean_matching=('same_reference_protection_spatial_mean' if memory_response_mode == 'mean' else None),
        memory_response_query_selection=('request_'+memory_response_mode if memory_response_mode else None),
        memory_response_mean_gate_target=(float(query_mask.float().mean())*float(job['query_write_weight']) if memory_response_mode else None),
        target_query_mask_mean=float(query_mask.float().mean()),
        target_query_mask_support_fraction=float((query_mask > 0).float().mean()),
        source_kv_pool_tokens=z.shape[1], source_kv_pool_unchanged=True)
    kind = job.get('residual_kind', 'full')
    alpha = float(job.get('condition_weight', 1.))
    tau = float(job.get('transport_tau', 0.))
    if not 0. <= tau <= 1. or (tau and kind != 'transport'):
        raise ValueError('transport_tau must be in [0,1] with residual_kind=transport')
    memory = None if job['mode'] == 'off' else AttentionMemory(b.pipe.transformer, z.shape[1], controls, job, alignment)
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
            low_rms = high_rms = desired_rms = candidate_rms = None
            candidate_local_rms = None
            weighted = controls['calibration']*lam*applied
            if request_global:
                # Same-state local-release candidate P*c, with P=.25+.75*(1-E).
                # Only its scalar energy is used; retain the full-image c direction.
                candidate, _, _ = spectral_candidate(correction, job, route_mask=route_mask)
                candidate_rms = rms(mask*controls['calibration']*candidate)
                candidate_local_rms = rms(mask*controls['calibration']*lam*candidate)
                global_rms = rms(mask*controls['calibration']*correction)
                energy_ratio = candidate_rms/max(global_rms, 1e-12)
                gain = lam*energy_ratio
                weighted = controls['calibration']*lam*correction*energy_ratio
                desired_rms = candidate_local_rms
            elif shaped and not r86_fixed:
                if budget_mode == 'external_u25':
                    baseline_step = budget_profile[i]
                    if abs(float(baseline_step['h'])-h) > 1e-8 or abs(float(baseline_step['sigma'])-sigma) > 1e-8:
                        raise ValueError('Budget reference schedule differs at step '+str(i))
                    desired_rms = float(baseline_step['applied_correction_rms'])
                elif budget_mode == 'online_same_state':
                    desired_rms = float(job.get('online_budget_fraction', lam)) * rms(controls['calibration']*mask*correction)
                candidate, low_rms, high_rms = spectral_candidate(correction, job, route_mask=route_mask)
                candidate = controls['calibration']*candidate
                candidate_rms = rms(mask*candidate)
                if budget_mode == 'local_coefficient':
                    gain = lam  # Exactly .25*M*c: no global RMS renormalization.
                    if control_kind == 'request_local_release':
                        candidate_local_rms = rms(mask*candidate*lam)
                else:
                    if candidate_rms <= 1e-12 and desired_rms > 1e-12:
                        raise FloatingPointError('Nonzero matched budget with zero ROI residual')
                    gain = desired_rms/max(candidate_rms, 1e-12)
                requested_gain = gain
                if job.get('region_gain_cap') is not None:
                    gain = min(gain, float(job['region_gain_cap']))
                    clipped = gain < requested_gain
                weighted = candidate*gain
            r86_target = None
            if job.get('r86_arm'):
                if budget_profile is None or len(budget_profile) != len(sigmas)-1: raise RuntimeError('Missing fixed R86 profile')
                anchor_step = budget_profile[i]
                if anchor_step['step_index'] != i or abs(float(anchor_step['sigma'])-sigma)>1e-8 or abs(float(anchor_step['h'])-h)>1e-8: raise RuntimeError('R86 schedule mismatch')
                r86_target = float(job['r86_beta']) * float(anchor_step['applied_correction_rms'])
                candidate = controls['calibration'] * (correction if job['r86_arm']=='GLOBAL' else spectral_candidate(correction, job, route_mask=route_mask)[0])
                if not bool(torch.isfinite(candidate).all()): raise FloatingPointError('Nonfinite R86 candidate')
                candidate_rms = rms(mask*candidate)
                if not math.isfinite(r86_target) or r86_target <= 0 or not math.isfinite(candidate_rms) or candidate_rms <= 0: raise FloatingPointError('Invalid or zero R86 target/candidate RMS')
                gain = r86_target / candidate_rms
                if not math.isfinite(gain): raise FloatingPointError('Nonfinite R86 gain')
                weighted = candidate * gain
                desired_rms = r86_target
                if not bool(torch.isfinite(weighted).all()): raise FloatingPointError('Nonfinite weighted correction')
            full_weighted = controls['calibration']*lam*correction
            actual_rms = rms(mask*weighted)
            if r86_target is not None and (not math.isfinite(actual_rms) or abs(actual_rms-r86_target)>max(1e-8,1e-5*r86_target)): raise FloatingPointError('R86 realized RMS mismatch')
            if request_global:
                # Account for the actual same-state control, including the epsilon
                # denominator; this is not a .25-global or cross-trajectory budget.
                target_budget += h*actual_rms
            elif budget_mode != 'local_coefficient':
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
                corr_lambda=lam, online_budget_fraction=None if request_global else float(job.get('online_budget_fraction', lam)), condition_weight=alpha, residual_kind=kind,
                control_kind=control_kind, candidate_local_rms=candidate_local_rms, actual_rms=actual_rms, r86_target_rms=r86_target, r86_beta=job.get('r86_beta'), r86_applied_update_rms=None if r86_target is None else h*actual_rms, r86_target_update_rms=None if r86_target is None else h*r86_target, r86_abs_rms_error=None if r86_target is None else abs(actual_rms-r86_target), r86_relative_rms_error=None if r86_target is None else abs(actual_rms-r86_target)/r86_target,
                region_gain=gain,
                current_correction_global_rms=rms(controls['calibration']*mask*correction),
                current_correction_roi_conditional_rms=roi_conditional_rms(controls['calibration']*mask*correction, route_mask),
                candidate_global_rms=candidate_rms,
                applied_correction_global_rms=actual_rms,
                applied_correction_roi_conditional_rms=roi_conditional_rms(mask*weighted, route_mask),
                query_selection=job['query_selection'],
                target_query_mask_mean=query_info['target_query_mask_mean'],
                target_query_mask_support_fraction=query_info['target_query_mask_support_fraction'],
                query_gate_mean=sum(v['query_gate_mean'] for v in changes)/len(changes) if changes else 0.,
                query_gate_support_fraction=sum(v['query_gate_support_fraction'] for v in changes)/len(changes) if changes else 0.,
                spectral_kernel=job.get('spectral_kernel'), spectral_low_weight=job.get('spectral_low_weight'),
                route_floor=floor, effective_route_mask_mean=route_info['effective_route_mask_mean'], effective_route_support_fraction=route_info['effective_route_support_fraction'],
                route_support_fraction=route_info['route_support_fraction'],
                route_mask_mean=route_info['route_mask_mean'],
                route_mask_coverage=route_info['route_mask_coverage'],
                route_mask_source=route_info['route_mask_source'], route_feather_px=route_info['route_feather_px'],
                edit_gain=beta, correction_shape=job.get('correction_shape', 'uniform'),
                native_prompt_difference_rms=None if direction is None else rms(direction),
                memory_delta_rms=None if memory_delta is None else rms(memory_delta),
                correction_low_rms=low_rms, correction_high_rms=high_rms,
                matched_step_budget_rms=desired_rms if budget_mode in ('external_u25', 'r86_fixed_profile') else None,
                desired_applied_correction_rms=desired_rms,
                correction_budget_mode=budget_mode,
                external_trajectory_budget_matched=shaped and budget_mode in ('external_u25', 'r86_fixed_profile'),
                same_state_uniform25_rms=.25*rms(controls['calibration']*mask*correction) if budget_mode=='online_same_state' else None,
                transport_tau=tau, source_state_shift_rms=state_shift_rms,
                source_current_velocity_rms=None if current_source_velocity is None else rms(current_source_velocity),
                a_rms=a_rms, condition_component_rms=b_rms, component_cosine=cosine_ab,
                full_correction_rms=rms(correction), residual_gain=gain, gain_clipped=clipped,
                reference_correction_rms=rms(mask*full_weighted),
                applied_correction_rms=actual_rms,
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
    local_rms_values = [row['candidate_local_rms'] for row in logs if row['candidate_local_rms'] is not None]
    return z.cpu(), dict(nfe=b.nfe-before, source_nfe=source_nfe+source_current_nfe+probe_info['route_probe_source_nfe'], target_nfe=target_nfe+native_target_nfe+probe_info['route_probe_target_nfe'],
        **probe_info, **route_info, **query_info, correction_budget_mode=budget_mode,
        control_kind=control_kind,
        candidate_local_rms_mean=sum(local_rms_values)/len(local_rms_values) if local_rms_values else None,
        actual_rms_mean=sum(row['actual_rms'] for row in logs)/len(logs),
        region_gain_mean=sum(r['region_gain'] for r in logs)/len(logs),
        applied_correction_global_rms_mean=sum(r['applied_correction_global_rms'] for r in logs)/len(logs),
        applied_correction_roi_conditional_rms_mean=(sum(r['applied_correction_roi_conditional_rms'] for r in logs)/len(logs) if route_mask is not None else None),
        query_gate_mean_over_steps=sum(r['query_gate_mean'] for r in logs)/len(logs),
        external_trajectory_budget_matched=shaped and budget_mode in ('external_u25', 'r86_fixed_profile'), memory_window=[job['src_start'], job['src_steps']],
        memory_window_indexing='zero_based_start_inclusive_end_exclusive',
        memory_active_steps=sum(row['injection_active'] for row in logs),
        spectral_kernel=job.get('spectral_kernel'), spectral_low_weight=job.get('spectral_low_weight'),
        native_target_nfe=native_target_nfe, edit_gain=beta, correction_shape=job.get("correction_shape", "uniform"),
        source_reference_nfe=source_nfe, source_current_nfe=source_current_nfe,
        transport_tau=tau, source_state_shift_mean_rms=sum(shifts)/len(shifts) if shifts else None,
        source_state_shift_steps=len(shifts),
        null_nfe=null_nfe, correction_impulse=impulse, budget_target=target_budget if budget_mode != 'local_coefficient' and (shaped or not transported) else None,
        correction_budget_relative_error=relative if budget_mode != 'local_coefficient' and (shaped or not transported) else None,
        correction_budget_status=(('fixed_profile_matched' if relative <= 1e-5 else 'budget_mismatch') if r86_fixed
            else 'same_state_local_release_energy_control' if request_global
            else 'capped_online_budget' if job.get('region_gain_cap') is not None
            else 'local_coefficient_unnormalized' if budget_mode=='local_coefficient'
            else ('self_normalized_same_state_rms' if relative <= 1e-5 else 'budget_mismatch') if budget_mode == 'online_same_state' and shaped
            else ('external_budget_matched' if relative <= 1e-5 else 'budget_mismatch') if budget_mode == 'external_u25' and shaped
            else 'unnormalized'),
        correction_budget_h=sum(row['h']*(row['region_gain'] if request_global or r86_fixed else lam) for row in logs),
        scheduled_gate_budget_h=sum(row['h']*row['scheduled_gate'] for row in logs),
        gate_budget_h=sum(row['h']*row['query_gate_mean'] for row in logs), steps=logs)
