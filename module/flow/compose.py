
import torch
import torch.nn as nn
from .base import (
    Flow,
    ExtendedFlow,
)

class StackedFlow(Flow, nn.Module):

    def __init__(self, flows):
        super(StackedFlow, self).__init__()
        self.flows = nn.ModuleList([flow for flow in flows if isinstance(flow, nn.Module)])
    
    def _forward(self, x, reverse=False):

        sz_batch, steps, dim = x.shape

        flows = self.flows
        if reverse:
            flows = self.flows[::-1]

        cumlogdet = torch.zeros(sz_batch, steps, dtype=x.dtype, device=x.device)
        for flow in flows:
            x, logdet = flow._forward(x, reverse=reverse)
            cumlogdet += logdet

        return x, cumlogdet

class ExtendedStackedFlow(ExtendedFlow, nn.Module):

    def __init__(self, flows):
        super(ExtendedStackedFlow, self).__init__()
        self.flows = nn.ModuleList([flow for flow in flows if isinstance(flow, nn.Module)])
    
    def _forward(self, x, phi, reverse=False):

        sz_batch, steps, dim = x.shape

        flows = self.flows
        if reverse:
            flows = self.flows[::-1]

        cumlogdet = torch.zeros(sz_batch, steps, dtype=x.dtype, device=x.device)
        for flow in flows:
            x, logdet = flow._forward(x, phi, reverse=reverse)
            cumlogdet += logdet

        return x, cumlogdet