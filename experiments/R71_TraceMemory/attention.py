"""Early source feature transfer in RF-Inversion.

Mid-step image-image / image-text maps and residual features are adapted from
ReFlex, arXiv:2507.01496, sections 4.1-4.3 and B.2. This implementation uses
the existing controlled RF inverse path and a fixed segmentation mask, not
ReFlex's noised inversion or its mask estimator. It keeps the target V.
R57 adds spatial-map interpolation and late attribute reweighting with a
fixed valid-text attention budget. These are experiment variants.
"""
from difflib import SequenceMatcher
import math
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_flux import _get_qkv_projections
from diffusers.models.embeddings import apply_rotary_emb


def token_alignment(tokenizer, source, target, length):
    def encode(text):
        encoded = tokenizer(text, padding='max_length', max_length=length,
                            truncation=True, return_attention_mask=True)
        ids = encoded['input_ids']
        valid = int(sum(encoded['attention_mask']))
        return ids, valid
    sids, ns = encode(source)
    tids, nt = encode(target)
    mapping = [-1]*length
    for block in SequenceMatcher(None, sids[:ns], tids[:nt], autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            mapping[block.b+offset] = block.a+offset
    changed = [i for i in range(nt) if mapping[i] < 0 and tids[i] not in tokenizer.all_special_ids]
    return dict(target_to_source=mapping, changed=changed,
                source_tokens=tokenizer.convert_ids_to_tokens(sids[:ns]),
                target_tokens=tokenizer.convert_ids_to_tokens(tids[:nt]),
                source_valid=ns, target_valid=nt,
                padding_policy='Keep native target padding columns; boost only changed non-special tokens.')


def sdpa(q, k, v, mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0., is_causal=False)


def balanced_gate(weight, budget):
    """Fixed per-image token mean. Compare allocation without changing mean strength."""
    if budget <= 0:return torch.zeros_like(weight)
    if budget >= 1:return torch.ones_like(weight)
    logits=weight.float().clamp_min(1e-8).log()
    logits=logits-logits.mean(dim=2,keepdim=True)
    lo=torch.full_like(logits[:,:,:1],-40.);hi=-lo
    for _ in range(22):
        mid=(lo+hi)/2
        mass=(logits+mid).sigmoid().mean(dim=2,keepdim=True)
        lo=torch.where(mass<budget,mid,lo);hi=torch.where(mass<budget,hi,mid)
    return (logits+(lo+hi)/2).sigmoid()


def ranked_gate(weight, budget, contrast=0.25, shuffle_seed=None):
    if abs(budget-0.5)>1e-6 or weight.shape[2] % 2:
        raise ValueError('Rank control requires an even token count and mean gate 0.5')
    if not 0 < contrast <= 0.5:
        raise ValueError('gate_contrast must be in (0, 0.5]')
    gate=torch.full_like(weight, budget-contrast)
    selected=weight.topk(weight.shape[2]//2,dim=2,sorted=False).indices
    gate.scatter_(2,selected,budget+contrast)
    if shuffle_seed is not None:
        generator=torch.Generator(device='cpu').manual_seed(int(shuffle_seed))
        permutation=torch.randperm(weight.shape[2],generator=generator,device='cpu').to(weight.device)
        gate=gate.index_select(2,permutation)
    return gate


class Processor:
    def __init__(self, owner, index, original):
        self.owner, self.index, self.original = owner, index, original

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        owner = self.owner
        if owner.mode == 'off':
            return self.original(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        q, k, v, tq, tk, tv = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
        v = v.unflatten(-1, (attn.heads, -1))
        if encoder_hidden_states is not None:
            tq = attn.norm_added_q(tq.unflatten(-1, (attn.heads, -1)))
            tk = attn.norm_added_k(tk.unflatten(-1, (attn.heads, -1)))
            tv = tv.unflatten(-1, (attn.heads, -1))
            q, k, v = torch.cat((tq, q), 1), torch.cat((tk, k), 1), torch.cat((tv, v), 1)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        nt = q.shape[2]-owner.tokens
        full = sdpa(q, k, v, attention_mask)
        if owner.mode == 'capture':
            if owner.variant == 'band_k':
                owner.bank[self.index] = k[:, :, nt:].detach().cpu().clone()
            elif owner.variant in ('attn_src', 'attn_budget', 'aligned_kv_budget'):
                owner.bank[self.index] = (k[:, :, nt:].detach().cpu().clone(),
                                          v[:, :, nt:].detach().cpu().clone())
            else:
                owner.bank[self.index] = (q[:, :, nt:].detach().cpu().clone(), k.detach().cpu().clone())
        else:
            self.map_stats = {}
            self.dynamic_gate = None
            if attention_mask is not None:
                raise ValueError('This experiment uses the default unmasked FLUX joint attention')
            if owner.variant == 'band_k':
                source_k = owner.bank[self.index].to(k.device, dtype=k.dtype)
                hybrid = k.clone()
                kw = owner.kweight.to(k.device)
                hybrid[:, :, nt:] = (k[:, :, nt:].float()+kw*(source_k.float()-k[:, :, nt:].float())).to(k.dtype)
                modified = sdpa(q[:, :, nt:], hybrid, v)
            elif owner.variant == 'attn_src':
                ks, vs = (x.to(q.device, dtype=q.dtype) for x in owner.bank[self.index])
                modified = sdpa(q[:, :, nt:], ks, vs)
            elif owner.variant in ('attn_budget', 'aligned_kv_budget'):
                modified = self.content_memory(q, k, v, nt)
            else:
                modified = self.adapt_maps(q, k, v, nt)
            gate = (self.dynamic_gate if self.dynamic_gate is not None else owner.gate).to(q.device, dtype=q.dtype)
            delta = gate*(modified-full[:, :, nt:])
            full[:, :, nt:] = full[:, :, nt:]+delta
            owner.actual.append(dict(block=self.index, component=owner.variant,
                                     sa=owner.step < owner.job['sa_steps'] if owner.variant != 'band_k' else False,
                                     output_delta_rms=float(delta.float().square().mean().sqrt()),
                                     **self.map_stats))
        output = full.transpose(1, 2).flatten(2).to(q.dtype)
        if encoder_hidden_states is not None:
            text, img = output[:, :nt], output[:, nt:]
            return attn.to_out[1](attn.to_out[0](img)), attn.to_add_out(text)
        return output

    def content_memory(self, q, k, v, nt):
        """Keep native text contribution AND native image mass for each query."""
        o = self.owner
        ks, vs = (x.to(q.device, dtype=q.dtype) for x in o.bank[self.index])
        image_k, image_v = ks, vs
        if o.variant == 'attn_budget':
            image_k = ks if o.job.get('memory_source_k', True) else k[:, :, nt:]
            image_v = vs if o.job.get('memory_source_v', True) else v[:, :, nt:]
        if o.variant == 'aligned_kv_budget':
            strength = float(o.job.get('kv_mix', 0.5))
            image_k = torch.lerp(k[:, :, nt:].float(), ks.float(), strength).to(k.dtype)
            image_v = torch.lerp(v[:, :, nt:].float(), vs.float(), strength).to(v.dtype)
        pairing = o.job.get('memory_pairing', 'paired')
        if pairing == 'v_deranged':
            if not (o.job.get('memory_source_k', True) and o.job.get('memory_source_v', True)):
                raise ValueError('PairBreak requires paired source K and V')
            cache = getattr(self, '_pair_permutations', None)
            if cache is None:
                cache = self._pair_permutations = {}
            n = image_v.shape[2]
            seed = int(o.job['case']) % 2147483647 + int(o.job['seed'])*1009 + self.index*9176
            key = (n, seed, str(image_v.device))
            if key not in cache:
                # Random single cycle: every V slot changes, no K slot changes.
                order = torch.randperm(n, generator=torch.Generator(device='cpu').manual_seed(seed))
                permutation = torch.empty_like(order)
                permutation[order] = order.roll(1)
                cache[key] = permutation.to(image_v.device)
            image_v = image_v.index_select(2, cache[key])
        elif pairing != 'paired':
            raise ValueError('Unknown memory pairing: ' + pairing)
        scale = 1 / math.sqrt(q.shape[-1])
        kt = k.float().transpose(-1, -2)
        chunks, text_mass, image_mass = [], 0., 0.
        concentration=[]
        source_kt=image_k.float().transpose(-1,-2)
        for start in range(0, o.tokens, 128):
            end = min(start + 128, o.tokens)
            qi = q[:, :, nt+start:nt+end]
            native = (qi.float() @ kt * scale).softmax(-1)
            tm = native[..., :nt].sum(-1, keepdim=True)
            im = native[..., nt:].sum(-1, keepdim=True)
            text_output = native[..., :nt].to(v.dtype) @ v[:, :, :nt]
            source_probability=(qi.float() @ source_kt * scale).softmax(-1)
            entropy=-(source_probability*source_probability.clamp_min(1e-12).log()).sum(-1,keepdim=True)
            concentration.append((1-entropy/math.log(o.tokens)).clamp(0,1).mean(dim=1,keepdim=True))
            retrieved = source_probability.to(image_v.dtype) @ image_v
            chunks.append((text_output.float() + im * retrieved.float()).to(v.dtype))
            text_mass += float(tm.sum())
            image_mass += float(im.sum())
        denom = q.shape[0] * q.shape[1] * o.tokens
        self.map_stats = dict(native_text_mass=text_mass / denom,
                              native_image_mass=image_mass / denom,
                              memory_source_k=o.job.get('memory_source_k', True),
                              memory_source_v=o.job.get('memory_source_v', True),
                              preserves_target_text_contribution=True,
                              memory_window=[o.job.get('src_start', 0), o.job['src_steps']])
        c=torch.cat(concentration,dim=2)
        drift=o.pixel_deviation.to(q.device).float()
        median=drift.median(dim=2,keepdim=True).values
        kind=o.job.get('gate_mode','uniform');budget=float(o.job['query_write_weight'])
        weight=(c+1e-6).square()
        if kind in ('dual','dual_rank','dual_rank_shuffle'):
            weight=weight*torch.exp(-(drift/(median+1e-6)).clamp(max=30))
        if kind=='uniform':self.dynamic_gate=torch.full_like(c,budget)
        elif kind in ('dual_rank','dual_rank_shuffle'):
            # Fixed permutation per layer/seed across timesteps preserves temporal consistency.
            seed=(int(o.job['case']) % 2147483647)+o.job['seed']*1009+self.index*9176
            self.dynamic_gate=ranked_gate(weight,budget,o.job.get('gate_contrast',0.25),
                shuffle_seed=seed if kind=='dual_rank_shuffle' else None)
        else:self.dynamic_gate=balanced_gate(weight,budget)
        centered_c=c-c.mean();centered_d=drift-drift.mean()
        denom_corr=centered_c.square().sum().sqrt()*centered_d.square().sum().sqrt()
        corr=float((centered_c*centered_d).sum()/denom_corr) if float(denom_corr)>1e-12 else None
        self.map_stats.update(gate_mode=kind,gate_mean=float(self.dynamic_gate.mean()),
            gate_std=float(self.dynamic_gate.std(unbiased=False)),
            feature_concentration_mean=float(c.mean()),trajectory_deviation_mean=float(drift.mean()),
            feature_drift_corr=corr,
            gate_budget_error=float((self.dynamic_gate.mean(dim=2)-budget).abs().max()))
        if self.index==o.job.get('gate_probe_block',13) and o.step in o.job.get('gate_probe_steps',[0,3,6]):
            self.map_stats['gate_probe']=dict(step=o.step,block=self.index,
                concentration=c[0,0,:,0].detach().cpu().tolist(),
                drift=drift[0,0,:,0].detach().cpu().tolist(),
                gate=self.dynamic_gate[0,0,:,0].detach().cpu().tolist())
        return torch.cat(chunks, dim=2)

    def adapt_maps(self, q, k, v, nt):
        o = self.owner
        early_ca = o.step < o.job['ca_steps']
        early_sa = o.step < o.job['sa_steps']
        early = early_ca or early_sa
        if early:
            qs, ks = (x.to(q.device, dtype=q.dtype) for x in o.bank[self.index])
            matched, mapped = o.matched.to(q.device), o.mapped.to(q.device)
            skt = ks.float().transpose(-1, -2)
        changed = o.changed.to(q.device)
        kt = k.float().transpose(-1, -2)
        scale = 1/math.sqrt(q.shape[-1])
        chunks = []
        late = o.attribute_beta > 1.0
        attr_before = torch.zeros((), device=q.device)
        attr_after = torch.zeros((), device=q.device)
        query_weight_sum = torch.zeros((), device=q.device)
        image_error = torch.zeros((), device=q.device)
        text_mass_error = torch.zeros((), device=q.device)
        if early_sa and o.job['spatial_blend']:
            spatial_alpha = o.spatial_alpha.to(q.device)
        if late:
            valid = min(o.target_valid, nt)
            text_weights = torch.ones(valid, device=q.device)
            text_weights[changed] = o.attribute_beta
            query_gate = o.gate.to(q.device).float()
        # Chunk rows to avoid storing 26 full FP32 joint-attention matrices.
        for start in range(0, o.tokens, 128):
            end = min(o.tokens, start+128)
            target = (q[:, :, nt+start:nt+end].float() @ kt * scale).softmax(-1)
            if early:
                source = (qs[:, :, start:end].float() @ skt * scale).softmax(-1)
            adapted = target.clone()
            if early_ca:
                if matched.numel():
                    adapted[..., matched] = source.index_select(-1, mapped)
                if changed.numel():
                    adapted[..., changed] = target.index_select(-1, changed)*o.job['attribute_boost']
            if early_sa:
                image_map = source[..., nt:].clone()
                # Step 1 uses source relations; from step 2, adapt top-k entries.
                if o.step >= 1:
                    indices = image_map.topk(min(o.job['sa_topk'], o.tokens), dim=-1).indices
                    old_mass = image_map.gather(-1, indices).sum(-1, keepdim=True)
                    selected = target[..., nt:].gather(-1, indices)
                    selected = selected*(old_mass/selected.sum(-1, keepdim=True).clamp_min(1e-12))
                    image_map.scatter_(-1, indices, selected)
                if o.job['spatial_blend']:
                    alpha = spatial_alpha[:, :, start:end]
                    image_map = target[..., nt:] + alpha*(image_map-target[..., nt:])
                adapted[..., nt:] = image_map
            if early:
                # Preserve R56 normalization in the unchanged baseline.
                adapted /= adapted.sum(-1, keepdim=True).clamp_min(1e-12)
            if late:
                # Late phase uses only current target attention and target V.
                # Redistribute the existing valid-text budget; do not take
                # mass from image tokens or contextualized padding tokens.
                adapted = target.clone()
                old_text = target[..., :valid]
                text_mass = old_text.sum(-1, keepdim=True)
                weighted = old_text*text_weights
                adapted[..., :valid] = weighted*(text_mass/weighted.sum(-1, keepdim=True).clamp_min(1e-12))
                gate = query_gate[:, :, start:end]
                attr_before += (target.index_select(-1, changed).sum(-1, keepdim=True)*gate).sum()
                attr_after += (adapted.index_select(-1, changed).sum(-1, keepdim=True)*gate).sum()
                query_weight_sum += gate.sum()*target.shape[0]*target.shape[1]
                image_error = torch.maximum(image_error, (adapted[..., nt:]-target[..., nt:]).abs().amax())
                text_mass_error = torch.maximum(text_mass_error, (adapted[..., :valid].sum(-1, keepdim=True)-text_mass).abs().amax())
            chunks.append((adapted.to(v.dtype) @ v))
        self.map_stats = dict(ca=early_ca, spatial_blend=bool(early_sa and o.job['spatial_blend']),
                              late_attribute=late, attribute_beta=o.attribute_beta)
        if late:
            self.map_stats.update(attribute_mass_before=float(attr_before/query_weight_sum.clamp_min(1e-12)),
                                  attribute_mass_after=float(attr_after/query_weight_sum.clamp_min(1e-12)),
                                  image_attention_max_change=float(image_error),
                                  valid_text_mass_max_error=float(text_mass_error))
        return torch.cat(chunks, dim=2)


class AttentionMemory:
    def __init__(self, model, tokens, controls, job, alignment):
        self.mode, self.step = 'off', 0
        self.variant, self.tokens, self.job = job['mode'], tokens, job
        self.bank, self.residual_bank, self.actual = {}, {}, []
        self.saved, self.handles, self.inputs = [], [], {}
        self.gate = controls['token_mask'].reshape(1, 1, tokens, 1)*job['query_write_weight']
        self.kweight = controls['k_weight'].reshape(1, 1, tokens, 1)
        core_fraction = (controls['token_core']/controls['token_mask'].clamp_min(1e-8)).clamp(0, 1)
        self.spatial_alpha = (job['spatial_core_weight']*core_fraction+
                              job['spatial_band_weight']*(1-core_fraction)).reshape(1, 1, tokens, 1)
        self.target_valid = alignment['target_valid']
        self.attribute_beta = 1.0
        mapping = alignment['target_to_source']
        matches = [(i, j) for i, j in enumerate(mapping) if j >= 0]
        self.matched = torch.tensor([i for i, j in matches], dtype=torch.long)
        self.mapped = torch.tensor([j for i, j in matches], dtype=torch.long)
        self.changed = torch.tensor(alignment['changed'], dtype=torch.long)
        for index in job['attention_blocks']:
            attn = model.single_transformer_blocks[index].attn
            self.saved.append((attn, attn.processor))
            attn.set_processor(Processor(self, index, attn.processor))
        if self.variant == 'mid_maps_res':
            blocks = [('D'+str(i), model.transformer_blocks[i]) for i in job['residual_double_blocks']]
            blocks += [('S'+str(i), model.single_transformer_blocks[i]) for i in job['residual_single_blocks']]
            for name, block in blocks:
                self.handles.append(block.register_forward_pre_hook(self.before_block(name), with_kwargs=True))
                self.handles.append(block.register_forward_hook(self.after_block(name), with_kwargs=True))

    def residual_active(self):
        return self.mode == 'capture' or (self.mode == 'apply' and self.step < self.job['residual_steps'])

    def before_block(self, name):
        def hook(module, args, kwargs):
            if self.residual_active():
                hidden = kwargs.get('hidden_states')
                if hidden is None:
                    hidden = args[0]
                # Both Flux block types receive separated image and text inputs.
                self.inputs[name] = hidden.detach().clone()
        return hook

    def after_block(self, name):
        def hook(module, args, kwargs, output):
            if not self.residual_active():
                return output
            original = self.inputs.pop(name)
            text, image = output
            residual = image.float()-original.float()
            if self.mode == 'capture':
                self.residual_bank[name] = residual.to(image.dtype).cpu().clone()
                return output
            source = self.residual_bank[name].to(image.device).float()
            gate = self.gate[:, 0].to(image.device)
            delta = gate*(source-residual)
            result = (image.float()+delta).to(image.dtype)
            self.actual.append(dict(block=name, component='mid_residual',
                                    output_delta_rms=float(delta.square().mean().sqrt())))
            return text, result
        return hook

    def capture(self):
        self.mode = 'capture'
        self.bank.clear()
        self.residual_bank.clear()
        self.actual = []

    def apply(self, step, deviation=None):
        self.mode, self.step = 'apply', step
        self.pixel_deviation=deviation if deviation is not None else torch.zeros_like(self.gate)
        self.actual = []
        self.attribute_beta = 1.0
        if self.job['late_attribute'] and self.job['attribute_start'] <= step < self.job['attribute_end']:
            ramp = self.job['attribute_ramp_steps']
            phase = min(1., (step-self.job['attribute_start']+1)/ramp,
                        (self.job['attribute_end']-step)/ramp)
            self.attribute_beta += (self.job['late_attribute_boost']-1)*phase

    def off(self):
        self.mode = 'off'

    def close(self):
        self.off()
        for attn, processor in self.saved:
            attn.set_processor(processor)
        for handle in self.handles:
            handle.remove()
        self.bank.clear()
        self.residual_bank.clear()
        self.inputs.clear()
