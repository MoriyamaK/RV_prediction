from __future__ import annotations
from typing import (
    Union,
    Tuple,
)
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from .base import Flow
from .nodelib import layers


class MixedUnivariateODEnet(layers.ODEnet):
    def __init__(
        self,
        hidden_dims,
        input_shape,
        strides,
        conv,
        mixture: Tensor | Parameter,
        layer_type="concat",
        nonlinearity="softplus",
        num_squeeze=0,
    ):
        dim = input_shape[0]
        assert (
            mixture.ndim == 2 and mixture.shape[0] == mixture.shape[1] == dim
        ), f"Input mixture matrix must have the shape ({dim}, {dim}, not {self.mixture.shape})"

        super().__init__(
            hidden_dims=hidden_dims,
            input_shape=(1, *input_shape[1:]),
            strides=strides,
            conv=conv,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
            num_squeeze=num_squeeze,
        )
        if type(mixture) is Parameter:
            self.mixture = mixture
        elif type(mixture) is Tensor:
            self.register_buffer("mixture", mixture)
        else:
            raise TypeError(mixture)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """
        batchified execution of the following procedure
        dx = mixture @ [g(t1, x1), g(t2, x2), ..., g(tD, xD)]^T

        Args:
            t (Tensor): (batch_size, dim)
            x (Tensor): (batch_size, dim)

        Returns:
            Tensor: (batch_size, dim)
        """
        dim = x.shape[-1]
        dx = []
        for i in range(dim):
            ti = t[:, i : i + 1] if t.ndim == 2 else t
            xi = x[:, i : i + 1]
            dxi = super().forward(ti, xi)  # (batch_size, 1)
            dx.append(dxi)
        dx = torch.cat(dx, dim=-1)  # (batch_size, dim)
        mixed_dx = dx @ self.mixture.T  # (batch_size, dim)
        return mixed_dx


def build_cnf(
    dim,
    hidden_dims,
    layer_type,
    nonlinearity,
    mixture: Tensor | Parameter,
    divergence_fn,
    residual,
    rademacher,
    time_length,
    train_T,
    solver,
    rtol,
    atol,
    regularization_fns,
):

    diffeq = MixedUnivariateODEnet(
        hidden_dims=hidden_dims,
        input_shape=(dim,),
        strides=None,
        conv=False,
        mixture=mixture,
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


class MixedUnivariateNeuralODEFlow(Flow, Module):
    def __init__(
        self,
        dim: int,
        hidden_dims: Union[str, Tuple],
        nonlinearity: str,
        mixture: Tensor | Parameter,
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
        super(MixedUnivariateNeuralODEFlow, self).__init__()
        if isinstance(hidden_dims, str):
            hidden_dims = tuple([int(x) for x in hidden_dims.split(",")])
        elif isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        chain = [
            build_cnf(
                dim=dim,
                hidden_dims=hidden_dims,
                layer_type=layer_type,
                nonlinearity=nonlinearity,
                mixture=mixture,
                divergence_fn=divergence_fn,
                time_length=time_length,
                train_T=train_T,
                residual=residual,
                rademacher=rademacher,
                solver=solver,
                rtol=rtol,
                atol=atol,
                regularization_fns=regularization_fns,
            )
            for _ in range(num_blocks)
        ]
        if batch_norm:
            bn_layers = [
                layers.MovingBatchNorm1d(
                    dim,
                    bn_lag=bn_lag,
                    effective_shape=dim,
                )
                for _ in range(num_blocks)
            ]

            bn_chain = [
                layers.MovingBatchNorm1d(
                    dim,
                    bn_lag=bn_lag,
                    effective_shape=dim,
                )
            ]
            for a, b in zip(chain, bn_layers):
                bn_chain.append(a)
                bn_chain.append(b)
            chain = bn_chain

        self.model = layers.SequentialFlow(chain)
        self._set_cnf_options(
            solver,
            atol,
            rtol,
            step_size,
            test_solver,
            test_atol,
            test_rtol,
            rademacher,
            residual,
        )

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

    def _set_cnf_options(
        self,
        solver,
        atol,
        rtol,
        step_size,
        test_solver,
        test_atol,
        test_rtol,
        rademacher,
        residual,
    ):
        def _set(module):
            if isinstance(module, layers.CNF):
                # Set training settings
                module.solver = solver
                module.atol = atol
                module.rtol = rtol
                if step_size is not None:
                    module.solver_options["step_size"] = step_size

                if solver in ["fixed_adams", "explicit_adams"]:
                    module.solver_options["max_order"] = 4

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
            x,
            torch.zeros(sz_batch * seqlen, 1, dtype=dtype, device=device),
            reverse=reverse,
        )

        z = z.view(sz_batch, seqlen, dim)
        flow_logdet = flow_logdet.view(sz_batch, seqlen)

        return z, flow_logdet
