import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pack(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//2, 2, w//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, h*w//4, c*4)


def pattern_roi_route(job, width, side):
    """Supplied 512-coordinate ROI; inward linear feather, then area downsample."""
    roi = job['route_roi']
    if (len(roi) != 4 or not all(0 <= float(v) <= 512 for v in roi)
            or not (roi[0] < roi[2] and roi[1] < roi[3])):
        raise ValueError('Invalid fixed pattern ROI')
    feather = float(job['route_feather_px'])
    if feather <= 0:
        raise ValueError('ROI feather must be positive in 512-coordinate pixels')
    # Evaluate at actual-resolution pixel centers in the original ROI coordinates.
    centers = (torch.arange(width, dtype=torch.float32) + .5) * (512. / width)
    x, y = centers[None, :], centers[:, None]
    dx = torch.minimum(x - float(roi[0]), float(roi[2]) - x)
    dy = torch.minimum(y - float(roi[1]), float(roi[3]) - y)
    full = (torch.minimum(dx, dy) / feather).clamp(0., 1.)[None, None]
    return F.interpolate(full, size=(side, side), mode='area')


def source_detail_route(folder, side):
    """Source-image-only contrast support; no prompt, annotation or evaluation ROI."""
    image = Image.open(folder/'source.png').convert('L').resize((side, side), Image.Resampling.BOX)
    grey = torch.from_numpy(np.asarray(image).copy()).float()[None, None]/255.
    gx = F.pad((grey[:, :, :, 1:]-grey[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((grey[:, :, 1:, :]-grey[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    score = F.avg_pool2d(F.pad(gx+gy, (2, 2, 2, 2), mode='replicate'), 5, stride=1)
    threshold = torch.quantile(score, .75)
    # A soft, smoothed top-quartile contrast support, not an object segmentation.
    support = (score >= threshold).float()
    return F.avg_pool2d(F.pad(support, (2, 2, 2, 2), mode='replicate'), 5, stride=1)


def load_controls(folder, job, device):
    width = int(job.get('resolution', 512))
    if width % 16:
        raise ValueError('Resolution must be divisible by 16')
    side = width // 8
    def load(name):
        image = Image.open(folder/name).convert('L').resize((side, side), Image.Resampling.BOX)
        return torch.from_numpy(np.asarray(image).copy()).float()[None, None] / 255.
    # Routing is independent of update/allowed/attention masks.
    route_mask64 = route_mask = None
    if job.get('route_mask'):
        if job.get('route_mask_source') == 'source_detail_contrast':
            route_mask64 = source_detail_route(folder, side).to(device)
        elif job.get('route_mask_source') in ('fixed_pattern_roi', 'fixed_center_box', 'shifted_pattern_roi'):
            route_mask64 = pattern_roi_route(job, width, side).to(device)
        else:
            raise ValueError('Unknown route mask source')
        route_mask = pack(route_mask64.expand(-1, 16, -1, -1))
    query_selection = job.get('query_selection', 'all_image_queries')
    if query_selection in ('pattern_roi', 'soft_roi'):
        if route_mask64 is None:
            raise ValueError('Manual query ROI requires the same route ROI mask')
        # Average the identical 2x2 latent patches used by FLUX packing, row-major.
        query_mask = F.avg_pool2d(route_mask64, 2, 2).flatten(2).transpose(1, 2)
        if query_selection == 'soft_roi':
            background_gate = float(job['background_query_gate'])
            query_mask = background_gate + (1.-background_gate)*query_mask
    elif query_selection == 'all_image_queries':
        query_mask = torch.ones(1, side*side//4, 1, device=device)
    else:
        raise ValueError('Unknown target query selection')
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
    return dict(query_mask=query_mask, route_mask=route_mask, route_mask64=route_mask64, mask=pack(update_mask.expand(-1, 16, -1, -1)).to(device),
                original_mask64=mask, update_mask64=update_mask,
                token_mask=F.avg_pool2d(mask, 2, 2).flatten(2).transpose(1, 2).to(device),
                token_core=token_core.to(device),
                allowed=pack((mask > 0).float().expand(-1, 16, -1, -1)).to(device),
                calibration=pack(lam.expand(-1, 16, -1, -1)).to(device),
                k_weight=(job['k_core']*token_core+job['k_band']*token_band).to(device),
                v_weight=(job['v_core']*token_core+job['v_band']*token_band).to(device),
                core=pack(core.expand(-1, 16, -1, -1)).to(device),
                band=pack(band.expand(-1, 16, -1, -1)).to(device))
