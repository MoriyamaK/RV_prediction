import pandas as pd
import torch
from ...model import GCFP1D

@torch.no_grad()
def evaluate_AE_GCFP1D(gcfp, phi, x, last_obs, 
                        **unused_kwargs):
    
    '''
    x: (sz_batch, steps, dim)
    '''
    ae, _, _,  _ = gcfp.ae(
        x, phi, last_obs=last_obs,
        )
    # se: (sz_batch, steps)
    obs_ae = torch.transpose(
        ae[:, last_obs:],    # (sz_batch, steps-last_obs)
        0, 1).contiguous()
    #obs_se: (steps-last_obs, sz_batch)
    return obs_ae

@torch.no_grad()
def collate_AE(aes: list, symbs,
                **unused_kwargs):
    '''
    ses: List of (steps, sz_batch)
    [tensor(steps, sz_batch)]
    '''
    n_stocks = len(symbs)
    aes = torch.cat(aes, dim=1).data.cpu().numpy()
    if aes.shape[1] > n_stocks:
        aes = aes[:, :n_stocks].contiguous().data.cpu().numpy()
    return pd.DataFrame(aes, columns=symbs)