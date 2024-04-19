import pandas as pd
import torch
from ...model import GCFP1D

@torch.no_grad()
def evaluate_SE_GCFP1D(gcfp, phi, x, last_obs,
                        **unused_kwargs):
    
    '''
    x: (sz_batch, steps, dim)
    '''
    se, _, _,  _ = gcfp.se(
        x, phi, last_obs=last_obs,
        )
    # se: (sz_batch, steps)
    obs_se = torch.transpose(
        se[:, last_obs:],    # (sz_batch, steps-last_obs)
        0, 1).contiguous()
    #obs_se: (steps-last_obs, sz_batch)
    return obs_se

@torch.no_grad()
def collate_SE(ses: list, symbs,
                **unused_kwargs):
    '''
    ses: List of (steps, sz_batch)
    [tensor(steps, sz_batch)]
    '''
    n_stocks = len(symbs)
    ses = torch.cat(ses, dim=1).data.cpu().numpy()

    if ses.shape[1] > n_stocks:
        ses = ses[:, :n_stocks].contiguous().data.cpu().numpy()
    return pd.DataFrame(ses, columns=symbs)