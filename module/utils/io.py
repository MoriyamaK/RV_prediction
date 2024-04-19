
import os
import json
from argparse import Namespace
import torch

from .. import (
    MixedUnivariateNeuralODEFlow, 
    ExtendedNeuralODEFlow,
    IdentityFlow,
    ExtendedStackedFlow,
    StackedFlow, 
    WallaceFlow,
    BoxCoxSymmetric, 
    TanhMixture,
    YeoJohnson, 
    LogFlow,
)

from .. import (
    HAR,
)

from .. import (
    GCFP1D,
    NamedEmbedding,
)

def build_gcfp(args) -> GCFP1D:

    flow_names = args.flow.lower().split('+')
    flows = []
    for name in flow_names:
        if name == 'node':
            if (args.phi_dim > 0):
                supported_flows = ['node', 'identity']
                assert name in supported_flows, \
                    'When `phi_dim>0`, only the following flows ' \
                    f'and their concatenation are supported: {supported_flows}'
                    
            flow = ExtendedNeuralODEFlow(
                dim=args.dim,
                phi_dim=args.phi_dim,
                hidden_dims=args.d_hidden,
                nonlinearity=args.nonlinearity,
                divergence_fn=args.divergence_fn,
                time_length=args.time_length,
                train_T=args.train_T,
                layer_type=args.layer_type,
                num_blocks=args.n_blocks,
                batch_norm=args.has_batchnorm,
                bn_lag=args.batchnorm_lag,
                solver=args.ode_solver,
                atol=args.atol,
                rtol=args.rtol,
                step_size=args.step_size,
                test_solver=args.test_ode_solver,
                test_atol=args.test_atol,
                test_rtol=args.test_rtol,
                rademacher=not args.no_rademacher,
                residual=args.residual,
            )
        elif name == 'identity':
            flow = IdentityFlow()
        elif name == 'wallace':
            flow = WallaceFlow()
        elif name == 'log':
            flow = LogFlow()
        elif name == 'boxcox':
            flow = BoxCoxSymmetric()
        elif name == 'tanh':
            flow = TanhMixture()
        elif name == 'yeojohnson':
            flow = YeoJohnson()
        flows.append(flow)
    flow = ExtendedStackedFlow(flows) if name != 'mult-node' else StackedFlow(flows)

    predmodel = args.predmodel.lower()
    if predmodel == 'har':
        predmodel = HAR(distribution=args.predmodel_dist)
    else:
        raise NotImplementedError

    gcfp = GCFP1D(predmodel=predmodel, flow=flow)

    return gcfp

def build_NamedEmbedding_from_checkpoint(checkpoint: dict):
    assert ('state_dict' in checkpoint) and (checkpoint['state_dict']['weight'].ndim == 2)
    assert 'embedding_names' in checkpoint
    # assert 'name2idx' in checkpoint
    # assert 'idx2name' in checkpoint

    num_embeddings = len(checkpoint['embedding_names'])
    embedding_dim = checkpoint['state_dict']['weight'].shape[1]
    emb = NamedEmbedding(num_embeddings=num_embeddings,
                         embedding_names=checkpoint['embedding_names'],
                         embedding_dim=embedding_dim)
    emb.load_state_dict(checkpoint['state_dict'])
    return emb


def load_from_checkpoint(dirpath, map_location='cpu'):
    with open(f'{dirpath}/args.json') as f:
        args = Namespace(**json.load(f))
    
    gcfp = build_gcfp(args)

    gcfp_checkpoint = torch.load(f'{dirpath}/gcfp.checkpoint.pt', map_location)
    gcfp.load_state_dict(gcfp_checkpoint['state_dict'])
    gcfp = gcfp.to(map_location)

    phis_checkpoint = torch.load(f'{dirpath}/phis.checkpoint.pt', map_location)
    phis = build_NamedEmbedding_from_checkpoint(phis_checkpoint)
    phis = phis.to(map_location)
    return {
        'gcfp': gcfp,
        'phis': phis,
        'args': args,
    }


def save_checkpoint(dirpath, gcfp, phis, args):
    os.makedirs(dirpath, exist_ok=True)

    torch.save({
        'state_dict': gcfp.state_dict()
    }, f'{dirpath}/gcfp.checkpoint.pt')

    torch.save({
        'state_dict': phis.state_dict(),
        'embedding_names': phis.embedding_names,
        'name2idx': phis.name2idx,
        'idx2name': phis.idx2name,
    }, f'{dirpath}/phis.checkpoint.pt')

    with open(f'{dirpath}/args.json', 'wt') as f:
        json.dump(args, f, indent=4)