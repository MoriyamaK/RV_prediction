import pandas as pd
import torch
from ...model import GCFP1D

@torch.no_grad()
def evaluate_NLL_GCFP1D(gcfp, phi, x, last_obs,
                        **unused_kwargs):
    '''
    x: (sz_batch, steps, dim)
    '''
    latent_nll, logdet = gcfp.nll(
        x, phi, last_obs=last_obs,
        )
    observ_nll = latent_nll + logdet
    # latent_nll, observ_nll, logdet: (sz_batch, steps)

    observ_nll = torch.transpose(
        observ_nll[:, last_obs:],    # (sz_batch, steps-last_obs)
        0, 1).contiguous()
    # observ_nll: (steps-last_obs, sz_batch)
    return observ_nll

@torch.no_grad()
def collate_NLL(nlls: list, symbs,
                **unused_kwargs):
    '''
    nlls: List of (steps, sz_batch)
    '''
    n_stocks = len(symbs)
    nlls = torch.cat(nlls, dim=1).data.cpu().numpy()
    # nlls: (steps, n_stocks+?)
    if nlls.shape[1] > n_stocks:
        nlls = nlls[:, :n_stocks].contiguous().data.cpu().numpy()
    return pd.DataFrame(nlls, columns=symbs)


