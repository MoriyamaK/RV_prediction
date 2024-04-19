
import pandas as pd
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import scipy
import numpy as np

class ReturnDataset(Dataset):
    def __init__(self, rets, dim, device, dtype=torch.float32):
        assert isinstance(rets, pd.DataFrame)
        _, n_assets = rets.shape
        symbs = rets.columns
        rets = torch.tensor(rets.values, dtype=dtype, device=device)

        ''' build up a list of indices sequentially '''
        n_samples = (n_assets + dim - 1) // dim
        indices = [
            list(range(i*dim, min(n_assets, (i+1)*dim))) 
            for i in range(n_samples)
        ]
        ''' complete `indices` if the last sample has insufficient elements '''
        remaining = dim - len(indices[-1])
        if remaining > 0:
            indices[-1] += list(range(remaining))

        # --------------------------------
        self.n_assets = n_assets
        self.symbs = symbs
        self.rets = rets
        self.n_samples = n_samples
        self.indices = indices

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        ids = self.indices[i]
        rets = self.rets[:, ids]
        symbs = [self.symbs[j] for j in ids]
        return rets, symbs

def make_collate_fn(n_sampling_paths):

    def collate_fn(samples):
        """ 
        Args:
            samples (List[Tensor, List[Str]]): a list of Tensors from ReturnDataset.
        """
        xs = torch.stack([x for x, symbs in samples], dim=1)
        # (steps, sz_batch, dim)
        if n_sampling_paths > 1:
            xs = xs.repeat([1, n_sampling_paths, 1])
        # (steps, n_sampling_paths * sz_batch, dim)

        symbs_list = [symbs for x, symbs in samples]
        return xs, symbs_list

    return collate_fn


def make_sequential_dataloader(rets, dim, sz_batch, device):
    dataset = ReturnDataset(rets, dim, device=device)
    collate_fn = make_collate_fn(1)
    dataloader = DataLoader(dataset, batch_size=sz_batch, shuffle=False,
                            collate_fn=collate_fn)

    return dataloader



