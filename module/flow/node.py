from typing import (
    Union, 
    Tuple,
)
import torch
import torch.nn as nn

from .base import Flow, ExtendedFlow
from .nodelib import layers

def build_cnf(dim, hidden_dims, layer_type, nonlinearity, 
              divergence_fn, residual, rademacher, 
              time_length, train_T, solver, rtol, atol,
              regularization_fns):
    
    diffeq = layers.ODEnet(
        hidden_dims=hidden_dims,
        input_shape=(dim,),
        strides=None,
        conv=False,
        layer_type=layer_type,
        nonlinearity=nonlinearity,
        )
    odefunc = layers.ODEfunc(
        diffeq=diffeq,
        divergence_fn=divergence_fn,
        residual=residual,
        rademacher=rademacher,
    )
    cnf = layers.CNF(
        odefunc=odefunc,
        T=time_length,
        train_T=train_T,
        regularization_fns=regularization_fns,
        solver=solver,
        rtol=rtol,
        atol=atol,
    )
    return cnf


class NeuralODEFlow(Flow, nn.Module):

    def __init__(self,
                 dim: int,
                 hidden_dims: Union[str, Tuple],
                 nonlinearity: str,
                 divergence_fn: str,
                 time_length: float,
                 train_T: bool,
                 layer_type: str,
                 num_blocks,
                 batch_norm,
                 bn_lag,
                 solver,
                 atol,
                 rtol,
                 step_size,
                 test_solver,
                 test_atol,
                 test_rtol,
                 rademacher,
                 residual,
                 regularization_fns=None,
                 ):
        super(NeuralODEFlow, self).__init__()
        if isinstance(hidden_dims, str):
            hidden_dims = tuple([int(x) for x in hidden_dims.split(',')])
        elif isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        
        chain = [build_cnf(
            dim=dim,
            hidden_dims=hidden_dims,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
            divergence_fn=divergence_fn,
            time_length=time_length,
            train_T=train_T,
            residual=residual,
            rademacher=rademacher,
            solver=solver,
            rtol=rtol,
            atol=atol,
            regularization_fns=regularization_fns,
        ) for _ in range(num_blocks)]
        if batch_norm:
            bn_layers = [
                layers.MovingBatchNorm1d(
                    dim, bn_lag=bn_lag, effective_shape=dim,
                )
                for _ in range(num_blocks)
            ]

            bn_chain = [
                layers.MovingBatchNorm1d(
                    dim, bn_lag=bn_lag, effective_shape=dim,
                )
            ]
            for a, b in zip(chain, bn_layers):
                bn_chain.append(a)
                bn_chain.append(b)
            chain = bn_chain
        
        self.model = layers.SequentialFlow(chain)
        self._set_cnf_options(solver, atol, rtol, step_size, 
                              test_solver, test_atol, test_rtol, 
                              rademacher, residual)
        
        self.dim = dim
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm
        self.bn_lag = bn_lag
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.step_size = step_size
        self.test_solver = test_solver
        self.test_atol = test_atol
        self.test_rtol = test_rtol
        self.rademacher = rademacher
        self.residual = residual
        self.regularization_fns = regularization_fns

    def _set_cnf_options(self, solver, atol, rtol, step_size, test_solver, test_atol, test_rtol, 
                        rademacher, residual):
        def _set(module):
            if isinstance(module, layers.CNF):
                # Set training settings
                module.solver = solver
                module.atol = atol
                module.rtol = rtol
                if step_size is not None:
                    module.solver_options['step_size'] = step_size
                
                if solver in ['fixed_adams', 'explicit_adams']:
                    module.solver_options['max_order'] = 4
                
                module.test_solver = test_solver if test_solver else solver
                module.test_atol = test_atol if test_atol else atol
                module.test_rtol = test_rtol if test_rtol else rtol
            
            if isinstance(module, layers.ODEfunc):
                module.rademacher = rademacher
                module.residual = residual
        
        self.model.apply(_set)

    def _forward(self, x, reverse=False, return_auxiliary_losses=False):
        """
        Functions for running the ctfp model

        Parameters
        ----------
        x: observations, a 3-D tensor of shape batchsize x max_length x input_size
            (sz_batch, seqlen, dim)
        vars: Difference between consequtive observation time stampes.
            2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
            position is observation or padded dummy variables
        args: arguments returned from parse_arguments

        Returns
        ----------
        z: (sz_batch, seqlen, dim)
        flow_logdet: (sz_batch, seqlen)
        """
        sz_batch, seqlen, dim = x.size()
        device = x.device
        dtype = x.dtype

        x = x.view(-1, x.shape[2])

        """ forward """
        z, flow_logdet = self.model(
            x, torch.zeros(sz_batch*seqlen, 1, dtype=dtype, device=device),
            reverse=reverse,
        )

        z = z.view(sz_batch, seqlen, dim)
        flow_logdet = flow_logdet.view(sz_batch, seqlen)

        if return_auxiliary_losses:
            # aux_losses = {'scale_reg': (z.std() - x.std()).abs()}
            aux_losses = {}
            return z, flow_logdet, aux_losses
        
        else:
            return z, flow_logdet


