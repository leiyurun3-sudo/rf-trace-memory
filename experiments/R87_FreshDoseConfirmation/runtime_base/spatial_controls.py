"""Deterministic request-independent permutation of 8x8 blocks on a 64x64 map."""
import hashlib
import torch

def permutation(case):
    seed=hashlib.sha256(('R87-permute-v1:'+case).encode('utf-8')).digest()
    return sorted(range(64), key=lambda index:(hashlib.sha256(seed+index.to_bytes(2,'big')).digest(),index))

def permute_grid(grid,indices):
    if tuple(grid.shape)!=(1,1,64,64):raise ValueError('R87 permutation requires a [1,1,64,64] grid')
    if sorted(indices)!=list(range(64)):raise ValueError('Invalid 64-block permutation')
    blocks=grid.reshape(1,1,8,8,8,8).permute(0,1,2,4,3,5).reshape(64,8,8)
    output=blocks[torch.tensor(indices,device=grid.device)].reshape(8,8,8,8).permute(0,2,1,3).reshape(1,1,64,64)
    if not torch.equal(torch.sort(grid.flatten()).values,torch.sort(output.flatten()).values):raise RuntimeError('R87 block permutation changed histogram')
    return output
