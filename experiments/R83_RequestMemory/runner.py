"""Offline R83 runner. Local engine/backend imports occur only for generation."""
import argparse
from collections import OrderedDict
import contextlib
import math
import os
from pathlib import Path
import signal
import time
import traceback

import reporting as report

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT, PUBLIC, RESULT = report.locations(ROOT)
STOP = False


def request_pause(signum, frame):
    global STOP
    STOP = True
    print(f'PAUSE requested by signal {signum}; finishing current image and package.', flush=True)


def should_pause(root=ROOT):
    return STOP or (Path(root) / 'PAUSE').exists()


class Progress:
    def __init__(self, out, planned):
        self.out = Path(out)
        self.s = dict(pid=os.getpid(), status='running', stage='preparing', planned=planned, completed=0, failed=0)

    def update(self, **values):
        self.s.update(values, updated=time.time())
        report.save(self.out / 'progress.json', self.s)
        print(report.canonical(self.s), flush=True)

    @contextlib.contextmanager
    def phase(self, name):
        self.update(stage=name)
        yield


def compact(value):
    """Preserve scalar diagnostics; replace large tensor/array payloads with shape."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): compact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) > 512:
            return dict(omitted='large_array', length=len(value))
        return [compact(v) for v in value]
    if hasattr(value, 'shape'):
        if tuple(value.shape) == ():
            return compact(value.item())
        return dict(omitted='tensor_or_array', shape=list(value.shape))
    raise TypeError('Unsupported diagnostic type: ' + type(value).__name__)


def effective_job(job, task):
    result = dict(job)
    result.setdefault('conflict_tau', .25)
    result.setdefault('corr_policy', 'constant')
    result.setdefault('corr_lambda', 0.)
    if not math.isfinite(float(result['corr_lambda'])) or float(result['corr_lambda']) < 0:
        raise ValueError('corr_lambda must be finite and nonnegative')
    result['calibration'] = 'full' if result['corr_lambda'] > 0 else 'none'
    result['edit_rubric'] = task['edit_rubric']
    result['diagnostic_only'] = bool(result.get('diagnostic_only') or result.get('cohort') == 'reconstruction' or result['source_prompt'] == result['target_prompt'])
    return result


def job_provenance(root, job, provenance):
    folder = Path(root) / 'data' / job['case']
    source_sha = report.sha(folder / 'source.png')
    outer_sha = report.sha(folder / 'outer.png')
    for name, value in (('source.png', source_sha), ('outer.png', outer_sha)):
        if value != provenance['input_sha256']['data/'+job['case']+'/'+name]:
            raise RuntimeError('Input changed before generation: ' + str(folder / name))
    return dict(source_sha256=source_sha, outer_sha256=outer_sha,
                source_prompt_sha256=report.digest(job['source_prompt']),
                target_prompt_sha256=report.digest(job['target_prompt']))


def assert_snapshot(root, provenance):
    if report.fingerprint(root) != provenance:
        raise RuntimeError('Code, plan or inputs changed during run; stop and use a new directory.')


@contextlib.contextmanager
def queue_locks(project=PROJECT, wait=True, root=ROOT):
    with contextlib.ExitStack() as stack:
        for name in ('.ab_queue.lock', '.mint_runner.lock'):
            while True:
                try:
                    stack.enter_context(report.file_lock(Path(project) / name))
                    break
                except (BlockingIOError, PermissionError):
                    if not wait:
                        raise RuntimeError('Runner active; retry CPU packaging/review after current run pauses.') from None
                    if should_pause(root):
                        raise Paused()
                    print('WAIT '+str(Path(project) / name), flush=True)
                    time.sleep(3)
        stack.enter_context(report.file_lock(Path(root) / '.results.lock'))
        yield


class Paused(Exception):
    pass


def work(root=ROOT):
    root = Path(root)
    out, _, _ = report.locations(root)
    plan, tasks = report.load_plan(root)
    provenance = report.fingerprint(root)
    report.guard_existing(root, provenance)
    mark = provenance['fingerprint']
    progress = Progress(out, len(plan['jobs']))
    # Queue locks cover all result writes; CPU pack/review also take these locks.
    try:
        with queue_locks(root.parent, root=root):
            assert_snapshot(root, provenance)
            report.guard_existing(root, provenance)
            out.mkdir(exist_ok=True)
            report.save(out / 'manifest.json', provenance)
            done = {j['id'] for j in plan['jobs'] if report.completed(root, j, mark)}
            progress.update(completed=len(done))
            report.package(root, plan, provenance)
            if should_pause(root):
                progress.update(status='paused', stage='paused before generation')
                report.package(root, plan, provenance)
                return
            if len(done) == len(plan['jobs']):
                progress.update(status='completed', stage='all completed; resume skipped all images')
                report.package(root, plan, provenance)
                return
            try:
                generate(root, plan, tasks, provenance, progress, done)
            except BaseException as exc:
                progress.update(status='failed', stage='run failed', error=str(exc))
                report.package(root, plan, provenance)
                raise
    except Paused:
        print('Paused while waiting for queue; existing results retained.', flush=True)


def generate(root, plan, tasks, provenance, progress, done):
    # No API calls, package installation or external experiment imports.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import gc
    import torch
    from PIL import Image
    from mint_backend import FluxBackend
    import model_loader
    from rf_kv import invert
    from controls import load_controls
    from engine import execute
    from attention import token_alignment

    out, _, _ = report.locations(root)
    # Some reused loaders expose OUT for their small environment inventory.
    if hasattr(model_loader, 'OUT'):
        model_loader.OUT = out
    if hasattr(model_loader, 'preflight_runtime'):
        with progress.phase('offline runtime preflight'):
            model_loader.preflight_runtime(plan)
    pipe = model_loader.load_model(plan, progress)
    pipe.transformer.eval()
    b = FluxBackend(pipe, 512, 512, int(plan.get('steps', 28)), int(plan.get('max_length', 512)))
    prompts = {}
    images = OrderedDict()
    deadline = time.monotonic() + float(plan.get('max_runtime_seconds', float('inf')))
    for planned_job in plan['jobs']:
        if planned_job['id'] in done:
            continue
        if should_pause(root) or time.monotonic() >= deadline:
            break
        # Fatal provenance changes stop the run rather than mixing experiment versions.
        assert_snapshot(root, provenance)
        task = report.validate_job(planned_job, tasks)
        job = effective_job(planned_job, task)
        verified = job_provenance(root, job, provenance)
        start = time.monotonic()
        dest = out / job['id']
        dest.mkdir(exist_ok=True)
        folder = root / 'data' / job['case']
        timings = {}
        stage = 'preparing'
        before = b.nfe
        inv_nfe = edit_nfe = 0
        inv_before = edit_before = None
        z = None
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            res = int(job.get('resolution', 512))
            if res <= 0 or res % 16:
                raise ValueError('resolution must be positive and divisible by 16')
            b.width = b.height = res
            b.steps = int(job.get('steps', plan.get('steps', 28)))
            b.schedule_reference_tokens = job.get('schedule_reference_tokens')
            b.experiment_strength = float(job.get('strength', job.get('experiment_strength', 1.)))
            progress.update(stage=stage, job=job['id'], step=0, steps=b.steps)
            prep_start = time.monotonic()
            image_key = (job['case'], res, verified['source_sha256'])
            if image_key not in images:
                with progress.phase('encoding source'):
                    with Image.open(folder / 'source.png') as image:
                        images[image_key] = b.encode(image.convert('RGB'))
                while len(images) > 2:
                    images.popitem(last=False)
            images.move_to_end(image_key)
            y, ids = images[image_key]
            b.token_shape = (res // 16, res // 16)
            for prompt in ('', job['source_prompt'], job['target_prompt']):
                if prompt not in prompts:
                    with progress.phase('encoding prompt'):
                        prompts[prompt] = b.embeddings(prompt)
            src, tgt = prompts[job['source_prompt']], prompts[job['target_prompt']]
            b.empty_embeddings = prompts['']
            sigmas = b.schedule(y.shape[1])
            sigma_values = [float(v) for v in sigmas]
            sigma_sha = report.digest(report.canonical(sigma_values))
            timings['prepare'] = time.monotonic() - prep_start

            def tick(step, total):
                # Signals and PAUSE are deliberately checked only between images.
                progress.update(stage=stage, job=job['id'], step=int(step), steps=int(total), new_nfe=int(b.nfe))

            budget_profile = None
            budget_reference_nfe = 0
            external_matched = job.get('correction_budget_mode') == 'external_u25'
            if external_matched != bool(job.get('budget_reference_id')):
                raise ValueError('Only external-matched arms require a U25 dependency')
            if external_matched:
                baseline_id = job['budget_reference_id']
                baseline_job = next(j for j in plan['jobs'] if j['id'] == baseline_id)
                baseline_record = report.completed(root, baseline_job, provenance['fingerprint'])
                if baseline_record is None:
                    raise RuntimeError('Budget dependency must finish first: '+baseline_id)
                if baseline_job['arm'] != 'U25_KV' or baseline_record['sigma_sha256'] != sigma_sha:
                    raise ValueError('U25 reference arm/schedule mismatch')
                for key in ('case','seed','source_prompt','target_prompt','resolution','steps','guidance','eta','gamma','inv_substeps'):
                    if baseline_job.get(key) != job.get(key):
                        raise ValueError('U25 reference protocol mismatch: '+key)
                budget_profile = report.read(dest.parent / baseline_id / 'steps.json')['edit']['steps']
                budget_reference_nfe = baseline_record['nfe_standalone']
            raw_profile = report.exact_raw_profile(root, job, sigma_sha)
            is_flowedit = job.get('method') == 'flowedit'
            is_rfedit = job.get('method') == 'rfedit'
            is_external = is_flowedit or is_rfedit
            inversion_stats = None
            trajectory = None
            if not is_external:
                stage = 'inversion'
                inv_start = time.monotonic()
                inv_before = b.nfe
                with progress.phase('RF inversion'):
                    trajectory, inversion_stats = invert(b, y, ids, prompts[''], job['seed'], job, tick)
                inv_nfe = int(b.nfe - inv_before)
                timings['inversion'] = time.monotonic() - inv_start
                controls = load_controls(folder, job, b.device)
                if controls['mask'].shape != y.shape:
                    raise ValueError(f"Mask/latent mismatch: {controls['mask'].shape} vs {y.shape}")
                alignment = token_alignment(pipe.tokenizer_2, job['source_prompt'], job['target_prompt'], b.max_length)
            else:
                timings['inversion'] = 0.0
            stage = 'editing'
            edit_start = time.monotonic()
            edit_before = b.nfe
            with progress.phase('edit '+job['id']):
                if is_rfedit:
                    import external_rfedit
                    z, stats = external_rfedit.execute_external(b, y, ids, src, tgt, job, tick)
                elif is_flowedit:
                    import external
                    z, stats = external.execute_external(b, y, ids, src, tgt, job, tick)
                else:
                    z, stats = execute(b, y, ids, src, tgt, controls, job, trajectory, tick, alignment, budget_profile, raw_profile=raw_profile)
            # The request probes mutate controls inside execute. Save actual P afterwards.
            if not is_external and controls.get('route_mask64') is not None:
                actual = controls.get('effective_route_mask64')
                if actual is None:
                    floor = float(job.get('route_floor', 0.0))
                    actual = floor + (1.0-floor)*controls['route_mask64']
                mask_pixels = (actual[0,0].detach().float().cpu().clamp(0,1).numpy()*255).astype('uint8')
                Image.fromarray(mask_pixels).resize((320,320), Image.Resampling.NEAREST).save(dest/'route_mask.jpg', quality=90)
                stats['route_preview_from'] = 'post_execute_controls_effective_P'
            if not is_external and controls.get('edit_response64') is not None:
                response = controls['edit_response64'].detach().float().cpu().squeeze().clamp(0,1)
                if response.ndim != 2:
                    raise ValueError('edit_response64 preview requires a 2D spatial map')
                response_pixels = (response.numpy()*255).astype('uint8')
                Image.fromarray(response_pixels).save(dest/'edit_response64.jpg', quality=90)
            if not is_external and job.get('memory_response_mode'):
                side = int(job['resolution']) // 16
                gate_pixels = (controls['query_mask'].detach().float().cpu().reshape(side, side).clamp(0, 1).numpy()*255).astype('uint8')
                Image.fromarray(gate_pixels).resize((320,320), Image.Resampling.NEAREST).save(dest/'memory_query_gate.jpg', quality=90)
            edit_nfe = int(b.nfe - edit_before)
            if is_rfedit:
                inv_nfe = int(stats['nfe_inversion'])
                edit_nfe = int(stats['nfe_edit'])
            stats.update(budget_reference_nfe_standalone=budget_reference_nfe,
                         budget_reference_reused_in_batch=external_matched,
                         nfe_with_budget_reference=inv_nfe+edit_nfe+budget_reference_nfe)
            timings['edit'] = time.monotonic() - edit_start
            stage = 'saving preview'
            decode_start = time.monotonic()
            with progress.phase(stage):
                raw = b.decode(z).convert('RGB')
                raw.save(dest / 'raw.png')
                raw.resize((320, 320), Image.Resampling.LANCZOS).save(dest / 'preview.jpg', quality=80)
                x0, y0, x1, y1 = job['pattern_roi']
                box = (math.floor(x0*raw.width/512), math.floor(y0*raw.height/512),
                       math.ceil(x1*raw.width/512), math.ceil(y1*raw.height/512))
                pattern = raw.crop(box)
                pattern.save(dest / 'pattern.png')
                pattern.save(dest / 'pattern.jpg', quality=92)
            timings['decode_save'] = time.monotonic() - decode_start
            source = report.source_preview(root, job['case'])
            source_eval = report.source_eval(root, job['case'])
            metric_start = time.monotonic()
            try:
                measures = report.metrics(dest / 'raw.png', source_eval, folder / 'outer.png', job['pattern_roi'])
            except Exception as exc:
                measures = dict(metrics_status='failed', metrics_error=str(exc), metrics_from='raw_png_resized_512_and_source_eval_png')
            timings['metrics'] = time.monotonic() - metric_start
            # Keep every step's injection/correction scalar diagnostics; no latent arrays.
            report.save(dest / 'steps.json', dict(edit=compact(stats), inversion=compact(inversion_stats)))
            step_records = stats.get('steps', [])
            scalar_steps = [{k:v for k,v in row.items() if not isinstance(v, (dict,list,tuple))} for row in step_records] if isinstance(step_records, list) else []
            report.save(dest / 'step_summary.json', dict(budget_reference_id=job.get('budget_reference_id'),
                diagnostics={k:compact(v) for k,v in stats.items() if k != 'steps'},
                memory_window=[job['src_start'],job['src_steps']],
                internal_memory_enabled=(not is_external and job['mode']!='off'),
                external_feature_memory_enabled=is_rfedit,
                external_captured_entries=stats.get('captured_entries'), external_applied_entries=stats.get('applied_entries'),
                memory_window_indexing='zero_based_start_inclusive_end_exclusive', steps=scalar_steps))
            hashes = {name:report.sha(dest / name) for name in ('raw.png', 'preview.jpg', 'pattern.png', 'pattern.jpg', 'steps.json')}
            source_eval_key = 'sources/' + job['case'] + '_source_eval.png'
            hashes[source_eval_key] = report.sha(source_eval)
            torch.cuda.synchronize()
            runtime_seconds = time.monotonic()-start
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            record = dict(runtime_seconds_synchronized=runtime_seconds,
                peak_cuda_allocated_bytes=peak_allocated, peak_cuda_reserved_bytes=peak_reserved,
                cuda_memory_scope='this process PyTorch allocator peaks during job; not total machine VRAM',
                raw_pair_reference_id=None if raw_profile is None else raw_profile['id'], job=job, planned_job=planned_job, status='completed', fingerprint=provenance['fingerprint'],
                plan_sha256=provenance['plan_sha256'], code_sha256=provenance['code_sha256'], **verified,
                artifact_sha256=hashes, preview_sha256=hashes['preview.jpg'], raw_sha256=hashes['raw.png'],
                source_preview_sha256=report.sha(source), source_eval_sha256=hashes[source_eval_key],
                generated_width=raw.width, generated_height=raw.height, generated_resolution=res, saved_resolution=320,
                pattern_crop_box_native=list(box), metrics=measures, sigma_sha256=sigma_sha,
                nfe_inversion=inv_nfe, nfe_edit=edit_nfe, nfe_total=int(b.nfe-before),
                nfe_standalone=inv_nfe+edit_nfe, target_nfe=compact(stats.get('target_nfe')),
                source_nfe=compact(stats.get('source_nfe')), null_nfe=compact(stats.get('null_nfe', 0 if is_external else None)),
                **{key:compact(stats.get(key)) for key in ('route_support_cells','route_total_cells','route_support_fraction',
                    'route_floor','effective_route_mask_mean','effective_route_support_fraction','route_mask_mean','route_mask_coverage','route_mask_source','route_feather_px','route_feather_coordinate_basis','route_changes_update_mask','memory_active_steps','budget_reference_nfe_standalone',
                    'budget_reference_reused_in_batch','nfe_with_budget_reference','correction_budget_mode','external_trajectory_budget_matched','query_selection','target_query_mask_mean','target_query_mask_support_fraction','source_kv_pool_tokens','source_kv_pool_unchanged','region_gain_mean','applied_correction_global_rms_mean','applied_correction_roi_conditional_rms_mean','query_gate_mean_over_steps','scheduled_gate_budget_h','memory_response_mode','memory_response_floor','memory_query_gate_mean','memory_query_gate_min','memory_query_gate_max','memory_mean_matching',)},
                source_reference_nfe=compact(stats.get('source_reference_nfe')),
                source_current_nfe=compact(stats.get('source_current_nfe')),
                native_target_nfe=compact(stats.get('native_target_nfe')), 
                transport_tau=compact(stats.get('transport_tau')),
                source_state_shift_mean_rms=compact(stats.get('source_state_shift_mean_rms')),
                source_state_shift_steps=compact(stats.get('source_state_shift_steps')),
                null_eval_performed=(stats.get('null_nfe', 0) > 0 if is_external or isinstance(stats.get('null_nfe'), (int, float)) else None),
                engine_reported_nfe=compact(stats.get('nfe')), budget_target=compact(stats.get('budget_target')),
                correction_impulse=compact(stats.get('correction_impulse')),
                correction_budget_relative_error=compact(stats.get('correction_budget_relative_error', stats.get('budget_relative_error'))),
                correction_budget_status=compact(stats.get('correction_budget_status', stats.get('budget_status'))),
                correction_budget_h=compact(stats.get('correction_budget_h', stats.get('budget_h'))), gate_budget_h=compact(stats.get('gate_budget_h')),
                seconds=time.monotonic()-start, timings=timings, diagnostic_only=job['diagnostic_only'],
                edit_ok=None, subject_ok=None, pattern_ok=None, artifact=None, judge_status='pending')
            # Record is the final commit marker; resume verifies all artifact hashes.
            for key,value in stats.items():
                if key.startswith(('response_', 'request_', 'probe_', 'nfe_', 'route_probe_', 'route_edit_', 'route_uses_')):
                    record.setdefault(key, compact(value))
            record.update(internal_memory_enabled=(not is_external and job['mode']!='off'),
                          external_feature_memory_enabled=is_rfedit,
                          external_captured_entries=stats.get('captured_entries'), external_applied_entries=stats.get('applied_entries'))
            report.save(dest / 'record.json', record)
            (dest / 'error.json').unlink(missing_ok=True)
            done.add(job['id'])
        except Exception as exc:
            if stage == 'inversion' and inv_before is not None:
                inv_nfe = int(b.nfe - inv_before)
            if stage == 'editing' and edit_before is not None:
                edit_nfe = int(b.nfe - edit_before)
            error = dict(peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                         peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                         runtime_seconds_synchronized=None, runtime_seconds_until_exception=time.monotonic()-start,
                         job=job, planned_job=planned_job, fingerprint=provenance['fingerprint'], **verified,
                         stage=stage, message=str(exc), traceback=traceback.format_exc(), seconds=time.monotonic()-start,
                         nfe_total=int(b.nfe-before), nfe_inversion=inv_nfe, nfe_edit=edit_nfe)
            report.save(dest / 'error.json', error)
            print(error['traceback'], flush=True)
            traceback.clear_frames(exc.__traceback__)
        finally:
            z = None
            gc.collect()
            if hasattr(pipe, 'maybe_free_model_hooks'):
                pipe.maybe_free_model_hooks()
            torch.cuda.empty_cache()
        rows = report.rows_for(root, plan, provenance['fingerprint'])
        progress.update(stage='image complete; packaging', completed=len(done), failed=sum(r['status']=='failed' for r in rows))
        report.package(root, plan, provenance)
    rows = report.rows_for(root, plan, provenance['fingerprint'])
    paused = should_pause(root) or time.monotonic() >= deadline
    failures = sum(r['status'] == 'failed' for r in rows)
    status = 'paused' if paused else ('completed_with_errors' if failures else 'completed')
    progress.update(status=status, stage=status, completed=len(done), failed=failures)
    report.package(root, plan, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', action='store_true', help='CPU-only gallery/CSV/review/ZIP refresh')
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_pause)
    if args.pack:
        with report.file_lock(ROOT / '.results.lock'):
            report.package(ROOT)
    else:
        work()


if __name__ == '__main__':
    main()
