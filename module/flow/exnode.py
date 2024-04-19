from typing import (
    Union,
    Tuple,
    List,
)
import copy
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import (
    odeint as odeint,
)

from .base import ExtendedFlow
from .nodelib import layers
from .nodelib.layers import diffeq_layers
from .nodelib.layers.odefunc import (
    NONLINEARITIES,
    sample_gaussian_like,
    sample_rademacher_like,
    divergence_bf,
)
from .nodelib.layers.odefunc_aug import (
    divergence_bf_aug,
    divergence_approx_aug,
)


class ExtendedODEnet(nn.Module):

    def __init__(self,
                 hidden_dims,
                 input_shape,
                 phi_dim,
                 layer_type='concat',
                 nonlinearity='swish',
                 ):
        super(ExtendedODEnet, self).__init__()

        layers = []
        activation_fns = []

        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "hyper": diffeq_layers.HyperLinear,
            "squash": diffeq_layers.SquashLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "blend": diffeq_layers.BlendLinear,
            "concatcoord": diffeq_layers.ConcatLinear,
        }[layer_type]
        
        dims = (input_shape[0]+phi_dim, ) + hidden_dims + (input_shape[0], )
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layer = base_layer(dim_in, dim_out)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

        self.input_shape = input_shape
        self.phi_dim = phi_dim

    def forward(self, t, y):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # dx: (batch_size, input_shape[0])

        zeropad = torch.zeros(dx.shape[0], self.phi_dim, dtype=y.dtype, device=y.device)
        dx = torch.cat( (dx, zeropad), dim=1)
        # dx: (batch_size, input_shape[0] + phi_dim)

        return dx


class ExtendedCNF(layers.CNF):

    def forward(self, z, phi, logpz=None, integration_times=None, reverse=False):


        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz   
        
        if integration_times is None:
            integration_times = torch.tensor(
                [0.0, self.sqrt_end_time * self.sqrt_end_time]
            ).to(z)
        if reverse:
            integration_times = layers.cnf._flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if phi.shape[-1] > 0:
            z = torch.cat( (z, phi), dim=1)
            
        if self.training:
            try:
                # atol = [self.atol, self.atol] + [1e20] * len(reg_states) \
                #         if self.solver == "dopri5" \
                #         else self.atol
                # rtol = [self.rtol, self.rtol] + [1e20] * len(reg_states) \
                #         if self.solver == "dopri5" \
                #         else self.rtol
                atol = self.atol
                rtol = self.rtol
                # atol = 1e-5
                # rtol = 1e-5
                state_t = odeint(
                    self.odefunc,
                    (z, _logpz) + reg_states,
                    integration_times.to(z),
                    atol=atol,
                    rtol=rtol,
                    method=self.solver,
                    options=self.solver_options,
                )
            except Exception as e:
                print(e)
                raise
        else:
            atol = self.test_atol
            rtol = self.test_rtol
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=rtol,
                rtol=rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t


class ExtendedODEfunc(layers.ODEfunc):

    def __init__(self, diffeq, dim, phi_dim, divergence_fn="approximate", residual=False, rademacher=False):
        super(layers.ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher

        self.divergence_fn_name = divergence_fn
        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf_aug
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx_aug

        self.register_buffer("_num_evals", torch.tensor(0.0))

        self.dim = dim
        self.phi_dim = phi_dim

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = t.type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                # For 1+1=2 dimensional input,
                # in evaluation, `divergence` is exact even if divergence_fn is `approximate`
                divergence = divergence_bf_aug(dy, y, effective_dim=self.dim).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e, effective_dim=self.dim).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(
                np.prod(y.shape[1:]), dtype=torch.float32
            ).to(divergence)
        return tuple(
            [dy, -divergence]
            + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]]
        )


def build_cnf(dim, phi_dim, hidden_dims, layer_type, nonlinearity, 
                       divergence_fn, residual, rademacher, 
                       time_length, train_T, solver, rtol, atol,
                       regularization_fns):
    
    diffeq = ExtendedODEnet(
        hidden_dims=hidden_dims,
        input_shape=(dim,),
        phi_dim=phi_dim,
        layer_type=layer_type,
        nonlinearity=nonlinearity,
        )
    odefunc = ExtendedODEfunc(
        diffeq=diffeq,
        dim=dim,
        phi_dim=phi_dim,
        divergence_fn=divergence_fn,
        residual=residual,
        rademacher=rademacher,
    )
    cnf = ExtendedCNF(
        odefunc=odefunc,
        T=time_length,
        train_T=train_T,
        regularization_fns=regularization_fns,
        solver=solver,
        rtol=rtol,
        atol=atol,
    )
    return cnf

class ExtendedSequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layersList):
        super(ExtendedSequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, phi, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, phi, reverse=reverse)

            return x
        else:
            try:
                for i in inds:
                    x, logpx = self.chain[i](x, phi, logpx, reverse=reverse)
            except Exception as e:
                print(e)
                raise
            return x, logpx


class ExtendedNeuralODEFlow(ExtendedFlow, nn.Module):

    def __init__(self,
                 dim: int,
                 phi_dim: int,
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
        super(ExtendedNeuralODEFlow, self).__init__()
        if isinstance(hidden_dims, str):
            hidden_dims = tuple([int(x) for x in hidden_dims.split(',')])
        elif isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        
        chain = [build_cnf(
            dim=dim,
            phi_dim=phi_dim,
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
        
        self.model = ExtendedSequentialFlow(chain)
        self._set_cnf_options(solver, atol, rtol, step_size, 
                              test_solver, test_atol, test_rtol, 
                              rademacher, residual)
        
        self.dim = dim
        self.phi_dim = phi_dim
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

    def _forward(self, x, phi, reverse=False, return_auxiliary_losses=False, batch_size=1024*128):
        """
        Functions for running the ctfp model

        Parameters
        ----------
        x: observations, a 3-D tensor of shape batchsize x max_length x input_size
            (sz_batch, seqlen, dim)
        phi: (sz_batch, seqlen, dim)
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

        size = sz_batch * seqlen
    
        #print(x.shape, phi.shape)
        x = x.view(size, x.shape[2])
        phi = phi.reshape(size, phi.shape[2])
        
        z, flow_logdet = [], []
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            z_, flow_logdet_ = self.model(
                x[start:end].contiguous(),
                phi[start:end].contiguous(), 
                torch.zeros(end - start, 1, dtype=dtype, device=device),
                reverse=reverse,
            )
            z.append(z_)
            #print(flow_logdet_.shape)
            flow_logdet.append(flow_logdet_)
        z = torch.cat(z, dim=0)
        flow_logdet = torch.cat(flow_logdet, dim=0)
        # z: (size=sz_batch*seqlen, dim+phi_dim)

        z = z[:, :dim]
        # z: (sz_batch*seqlen, dim)
        z = z.view(sz_batch, seqlen, dim)
        #print(z.shape)
        flow_logdet = flow_logdet.view(sz_batch, seqlen)
        #print(flow_logdet.shape)
        if return_auxiliary_losses:
            # aux_losses = {'scale_reg': (z.std() - x.std()).abs()}
            aux_losses = {}
            return z, flow_logdet, aux_losses
        
        else:
            return z, flow_logdet
