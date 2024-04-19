import numpy as np
import torch
import pandas as pd
import scipy
from scipy import stats
from collections import defaultdict
from typing_extensions import Literal
from ...model import GCFP1D

@torch.no_grad()
def sampling_GCFP1D(gcfp: GCFP1D, phi, x, last_obs,
                    observ_stats, latent_stats,
                    calibrate,
                    args, **unused_kwargs):
    '''
    x: (sz_batch, steps, dim)
    '''

    N = args.momentcal_sampling_paths
    n = args.momentcal_batch_size

    samples = []
    if calibrate:
        for j in range((N+n-1) // n):
            batch = gcfp.sampling(
                x, phi, last_obs=last_obs,
                n_sampling_paths=min(n, N-j*n),
                observ_stats=observ_stats, latent_stats=latent_stats)
            samples.append(batch.data.cpu().numpy())
    else:
        for j in range((N+n-1) // n):
            batch = gcfp.predmodel.sampling(
                x, last_obs=last_obs,
                n_sampling_paths=min(n, N-j*n))
            samples.append(batch.data.cpu().numpy())
        # (N, sz_batch, steps+1, 1)

    samples = np.concatenate(samples, axis=0)  # (N, sz_batch, ?steps+1, 1)
    samples = samples[:, :, :-1, :]            # (N, sz_batch, ?steps, 1)

    sz_batch = x.shape[1]
    samples = np.transpose(samples, (1, 0, 2, 3))
    # (sz_batch, N, steps+1, 1)
    return samples

def cal_moments(samples: np.ndarray):
    '''
    samples: List of (sz_batch, ?)
    '''
    if samples.ndim == 2:
        return pd.DataFrame({
            'mean': np.mean(samples, axis=1),
            'variance': np.var(samples, axis=1),
            'skewness': stats.skew(samples, axis=1),
            'kurtosis': stats.kurtosis(samples, axis=1),
        })
    elif samples.ndim == 1:
        return cal_moments(samples[None]).loc[0]
    else:
        return ValueError

def evaluate_moments_GCFP1D(samples, symbs,
                            tstart: int, tend: int,
                            **unused_kwargs):
    '''
    samples: List of np.ndarray (sz_batch, N, steps+1, *)
    '''
    samples = np.concatenate(samples, axis=0)
    #print(f'samples.shape={samples.shape}')

    n_assets = len(symbs)
    if samples.shape[0] > n_assets:
        samples = samples[:n_assets]
    
    sz_batch = samples.shape[0]
    samples = samples[:, :, tstart:tend].reshape(sz_batch, -1)

    moments = cal_moments(samples)
    moment_all = cal_moments(samples.reshape(-1))
    # DataFrame (n_assets+?, n_moments)
    return moments, moment_all