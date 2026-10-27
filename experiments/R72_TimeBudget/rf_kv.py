"""Unchanged R53 source inversion and sigma checkpoint convention."""
import torch

@torch.inference_mode()
def invert(b,y,ids,empty,seed,j,tick):
    s=b.schedule(y.shape[1]);asc=s.flip(0);z=y.to(b.device).float().clone()
    noise=torch.randn(z.shape,generator=torch.Generator(device=b.device).manual_seed(seed),device=b.device)
    states=[z.cpu().clone()];logs=[];sub=int(j['inv_substeps']);count=b.nfe
    for i in range(len(asc)-1):
        lo,hi=float(asc[i]),float(asc[i+1]);h=(hi-lo)/sub
        for k in range(sub):
            t=lo+k*h
            if not 0<=t<1:raise ValueError('Invalid inversion model time')
            v=b.field(z,t,empty,ids,0.)
            z=z+h*((1-j['gamma'])*v+j['gamma']*(noise-z)/(1-t))
            if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite inversion')
            logs.append(dict(model_time=t,h=h,node=i,substep=k,rms=float(z.square().mean().sqrt())))
            tick(i*sub+k+1,(len(asc)-1)*sub)
        states.append(z.cpu().clone())
    # Coarse checkpoints exactly match descending generation nodes, even for sub=2.
    return list(reversed(states)),dict(seed=seed,gamma=j['gamma'],sigmas=s.tolist(),state_sigmas=s.tolist(),logs=logs,nfe=b.nfe-count,conditioning='empty text, guidance embedding 0',substeps=sub)
