"""Core-only centered V difference; no assumption that it is geometry-only."""
import torch


class CenteredValue:
    def __init__(self, model, blocks, image_tokens, controls):
        self.mode = 'off'
        self.bank, self.actual, self.handles = {}, [], []
        self.tokens = image_tokens
        self.core = controls['token_core']
        self.weight = controls['v_weight']
        for index in blocks:
            self.handles.append(model.single_transformer_blocks[index].attn.to_v.register_forward_hook(self.hook(index)))

    def hook(self, index):
        def apply(module, args, output):
            if self.mode == 'off':
                return output
            current = output[:, -self.tokens:]
            if self.mode == 'capture':
                self.bank[index] = current.detach().cpu().clone()
                return output
            difference = self.bank[index].to(current.device).float() - current.float()
            core = self.core.to(current.device)
            mean = (core*difference).sum(dim=1, keepdim=True)/core.sum(dim=1, keepdim=True).clamp_min(1e-8)
            delta = self.weight.to(current.device)*(difference-mean)
            result = output.clone()
            result[:, -self.tokens:] = (current.float()+delta).to(current.dtype)
            self.actual.append(dict(block=index, component='centered_V',
                                    delta_rms=float(delta.square().mean().sqrt()),
                                    removed_mean_rms=float(mean.square().mean().sqrt())))
            return result
        return apply

    def capture(self):
        self.mode = 'capture'
        self.bank.clear()
        self.actual = []

    def apply(self):
        self.mode = 'apply'
        self.actual = []

    def off(self):
        self.mode = 'off'

    def close(self):
        self.off()
        for handle in self.handles:
            handle.remove()
        self.bank.clear()
