import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pack(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//2, 2, w//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, h*w//4, c*4)


def load_controls(folder, job, device):
    width = int(job.get('resolution', 512))
    if width % 16:
        raise ValueError('Resolution must be divisible by 16')
    side = width // 8
    def load(name):
        image = Image.open(folder/name).convert('L').resize((side, side), Image.Resampling.BOX)
        return torch.from_numpy(np.asarray(image).copy()).float()[None, None] / 255.
    unrestricted = job.get('global_update', False)
    mask = torch.ones(1, 1, side, side) if unrestricted else load(job['edit_mask'])
    # Preserve the exact nonzero support and attention gate. Only the RF
    # update budget of fractional cells changes in the E arms.
    update_mask = mask.pow(job['update_mask_exponent'])
    if job.get('update_sampling')=='max':
        source=Image.open(folder/job['edit_mask']).convert('L').resize((width,width),Image.Resampling.NEAREST)
        full=torch.from_numpy(np.asarray(source).copy()).float()[None,None]/255.
        update_mask=F.max_pool2d(full,8,8)*(mask>0).float()
    core = torch.zeros_like(mask) if unrestricted else torch.minimum(load('core.png'), mask)
    band = (mask-core).clamp_min(0)
    relative_band = torch.where(mask > 0, band/mask.clamp_min(1e-8), torch.zeros_like(mask))
    lam = job['calibration_core'] + (job['calibration_band']-job['calibration_core'])*relative_band
    if job['calibration'] == 'none': lam = torch.zeros_like(mask)
    elif job['calibration'] == 'full': lam = torch.ones_like(mask)
    token_core = F.avg_pool2d(core, 2, 2).flatten(2).transpose(1, 2)
    token_band = F.avg_pool2d(band, 2, 2).flatten(2).transpose(1, 2)
    return dict(mask=pack(update_mask.expand(-1, 16, -1, -1)).to(device),
                original_mask64=mask, update_mask64=update_mask,
                token_mask=F.avg_pool2d(mask, 2, 2).flatten(2).transpose(1, 2).to(device),
                token_core=token_core.to(device),
                allowed=pack((mask > 0).float().expand(-1, 16, -1, -1)).to(device),
                calibration=pack(lam.expand(-1, 16, -1, -1)).to(device),
                k_weight=(job['k_core']*token_core+job['k_band']*token_band).to(device),
                v_weight=(job['v_core']*token_core+job['v_band']*token_band).to(device),
                core=pack(core.expand(-1, 16, -1, -1)).to(device),
                band=pack(band.expand(-1, 16, -1, -1)).to(device))
