# MIT License
#
# Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Link: https://github.com/rtqichen/ffjord

import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint
#from torchdiffeq import odeint_adjoint as odeint

from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]


class CNF(nn.Module):
    def __init__(
            self,
            odefunc,
            T=1.0,
            train_T=False,
            regularization_fns=None,
            solver="dopri5",
            atol=1e-5,
            rtol=1e-5,
    ):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter(
                "sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T)), requires_grad=True)
            )
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, z, logpz=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz
        if integration_times is None:
            integration_times = torch.tensor(
                [0.0, self.sqrt_end_time * self.sqrt_end_time]
            ).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        # print('z:', z.shape)
        if self.training:
            try:
                state_t = odeint(
                    self.odefunc,
                    (z, _logpz) + reg_states,
                    integration_times.to(z),
                    atol=[self.atol, self.atol] + [1e20] * len(reg_states)
                    if self.solver == "dopri5"
                    else self.atol,
                    rtol=[self.rtol, self.rtol] + [1e20] * len(reg_states)
                    if self.solver == "dopri5"
                    else self.rtol,
                    method=self.solver,
                    options=self.solver_options,
                )
            except Exception as e:
                print(e)
                raise
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
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

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]
