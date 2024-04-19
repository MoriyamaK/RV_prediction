from typing import (
    Union,
    List
)
import torch
from torch import nn

class NamedEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, 
                 embedding_names: List[str],
                 embedding_dim: int,
                 **kwargs):
        assert num_embeddings == len(embedding_names)
        super(NamedEmbedding, self).__init__(num_embeddings, embedding_dim,
                                             **kwargs)
        torch.nn.init.normal_(self.weight, 0, 0.1)
        self.embedding_names = embedding_names
        self.name2idx = {name:idx for name, idx in zip(embedding_names, range(num_embeddings))}
        self.idx2name = {idx:name for name, idx in self.name2idx.items()}
    
    def forward(self, namelists: Union[List[str], List[List[str]]]):
        name2idx = self.name2idx
        device = self._parameters['weight'].device

        indices = []
        for names in namelists:
            if isinstance(names, list):
                indices.append(
                    [name2idx[name] for name in names]
                )
            elif isinstance(names, str):
                indices.append(name2idx[names])
            else:
                raise NotImplementedError
        indices = torch.tensor(indices, device=device, dtype=torch.long)
        return super(NamedEmbedding, self).forward(indices)

