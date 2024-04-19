import numpy as np
from scipy.stats import yeojohnson
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Flow


class IdentityFlow(Flow, nn.Module):
    def _forward(self, x, *args, reverse=False):
        """
        x: (*, dim)
        """
        dims = list(x.shape)[:-1]
        flow_logdet = torch.zeros(*dims, device=x.device, dtype=x.dtype)
        return x, flow_logdet
        


def deriv_tanh(x):
    """derivative of tanh"""
    y = torch.tanh(x)
    return 1.0 - y * y

class LogFlow(Flow, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.a = nn.Parameter(torch.randn(1))
    
    def _forward(self, x, *args, reverse=False, eps=1e-10):
        
        sign = torch.sign(x)
        a = torch.abs(self.a)
        if not reverse:
            y = torch.sqrt(a * torch.log(1 + (eps+x**2) / a))
            det =  a * x / ((a + x**2 + eps) * torch.sqrt(a * torch.log(1 + (eps + x**2) / a)))
            logdet = -torch.log(det.abs()).sum(dim=2)
        else:
            y = torch.sqrt(a * (torch.exp(x**2 / a) - 1) - eps) 
            det = x * torch.exp(x**2 / a) / torch.sqrt(a * (torch.exp(x**2/a) - 1) -eps)
            logdet = -torch.log(det.abs()).sum(dim=2)
            
        return y * sign, logdet    
            
        

class WallaceFlow(Flow, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.a = nn.Parameter(torch.randn(1))

    def _forward(self, x, *args, reverse=False, eps=1e-5):
        """
        x: (sz_batch, seqlen, dim=1)
        """
        sz_batch, seqlen, dim = x.shape
        assert (
            dim == 1
        ), f"Wallace flow only supports 1-dimensional input, not {x.shape}"

        sign = torch.sign(x)

        a = self.a.abs() + 2
        b = (8 * a + 1) / (8 * a + 3)

        if not reverse:
            plus_square = 1 + (x**2 + eps) / a
            log_plus_square = (a * torch.log(plus_square)) ** 0.5

            y = b * log_plus_square
            # logdet: (sz_batch, seqlen, dim=1)

            det = b * (2 * x / plus_square) / (2 * log_plus_square + eps)
            logdet = -torch.log(det.abs()).sum(dim=2)
            # logdet = - (torch.log(b) + torch.log(2*x) - torch.log(plus_square) - torch.log(2 * log_plus_square + eps)).sum(dim=2).abs()
            # logdet: (sz_batch, seqlen)

        else:
            exp = torch.exp((x**2) / a / (b**2)) + eps
            y = (a * (exp - 1)) ** 0.5

            det = x * exp / (y + eps) / b**2
            logdet = -torch.log(det.abs() + eps).sum(dim=2)

        return y * sign, logdet
        
class YeoJohnson(Flow, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._lambd = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def lambd(self):
        return 1 + self._lambd if self._lambd >=0 else self._lambd.exp()

    def _forward(self, x, *args, reverse=False):
        """
        x: (sz_batch, seqlen, dim=1)
        """
        assert x.size(-1) == 1

        lambd = self.lambd

        y = torch.zeros_like(x)

        def f_dy_dx(x):
            pos_mask = x >= 0
            dy_dx = torch.zeros_like(x)
            dy_dx[pos_mask] = torch.pow(x[pos_mask] + 1, lambd - 1)
            if lambd == 2:
                dy_dx[~pos_mask] = 1 / (x[~pos_mask] + 1)
            else:
                dy_dx[~pos_mask] = - torch.pow(-x[~pos_mask] + 1, 1 - lambd)
            return dy_dx

        pos_mask = x >= 0
        if not reverse:
            y[pos_mask] = (torch.pow(x[pos_mask] + 1, lambd) - 1) / lambd
            if lambd == 2:
                y[~pos_mask] = - torch.log(-x[~pos_mask] + 1)
            else:
                y[~pos_mask] = - (torch.pow(-x[~pos_mask] + 1, 2 - lambd) - 1) / (2 - lambd)

            dy_dx = f_dy_dx(x).sum(dim=-1) # (sz_batch, seqlen)
            log_dy_dx = dy_dx.abs().log()
        else:
            y[pos_mask] = torch.pow(lambd * x[pos_mask] + 1, 1 / lambd) - 1
            if lambd == 2:
                y[~pos_mask] = torch.exp(x[~pos_mask] - 1)
            else:
                y[~pos_mask] = - torch.pow(1 - (2 - lambd) * x[~pos_mask], 1 / (2 - lambd)) + 1
            
            dx_dy = f_dy_dx(y).sum(dim=-1)
            log_dy_dx = - dx_dy.abs().log()
        
        return y, -log_dy_dx

'''
class BoxCoxSymmetric(Flow, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._lambd = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def lambd(self):
        return 1 + self._lambd if self._lambd >=0 else self._lambd.exp()

    def forward(self, x, *args, reverse=False):
        """
        x: (sz_batch, seqlen, dim=1)
        """
        assert x.size(-1) == 1

        lambd = self.lambd

        y = torch.zeros_like(x)

        def f_dy_dx(x):
            pos_mask = x >= 0
            dy_dx = torch.zeros_like(x)
            dy_dx[pos_mask] = torch.pow(x[pos_mask] + 1, lambd - 1)
            dy_dx[~pos_mask] = - torch.pow(-x[~pos_mask] + 1, lambd - 1)
            return dy_dx

        pos_mask = x >= 0
        if not reverse:
            y[pos_mask] = (torch.pow(x[pos_mask] + 1, lambd) - 1) / lambd
            y[~pos_mask] = - (torch.pow(-x[~pos_mask] + 1, lambd) - 1) /lambd

            dy_dx = f_dy_dx(x).sum(dim=-1) # (sz_batch, seqlen)
            log_dy_dx = dy_dx.abs().log()
        else:
            y[pos_mask] = torch.pow(lambd * x[pos_mask] + 1, 1 / lambd) - 1
            y[~pos_mask] = - (torch.pow(- lambd * x[~pos_mask] + 1, 1 / lambd) - 1)
            
            dx_dy = f_dy_dx(y).sum(dim=-1)
            log_dy_dx = - dx_dy.abs().log()
        
        return y, -log_dy_dx
'''

class BoxCoxSymmetric(Flow, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._lambd = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def lambd(self):
        return 1 + self._lambd if self._lambd >=0 else self._lambd.exp()
    
    def _forward(self, x, *args, reverse=False):
        """
        x: (sz_batch, seqlen, dim=1)
        """
        assert x.size(-1) == 1

        lambd = self.lambd

        y = torch.zeros_like(x)

        if not reverse:
            y = (torch.pow(x, lambd) - 1) / lambd
            dy_dx = (torch.pow(x, lambd - 1)).sum(dim=-1) # (sz_batch, seqlen)
            log_dy_dx = dy_dx.abs().log()
        else:
            y = torch.pow(lambd * x + 1, 1 / lambd)
            
            dx_dy = (torch.pow(lambd * x + 1, 1 / lambd - 1)).sum(dim=-1)
            log_dy_dx = dx_dy.abs().log()
        
        return y, -log_dy_dx
'''
class TanhMixture(Flow, nn.Module):
    def __init__(self, k=1, bias=True):
        nn.Module.__init__(self)
        self.u = nn.Parameter(torch.randn(k) * 0.01 + 1.0)
        self.v = nn.Parameter(torch.randn(k) * 0.01 + 1.0)
        if bias:
            self.b = nn.Parameter(torch.randn(k) * 0.01)
        else:
            self.register_buffer("b", torch.zeros(k))

    def _forward(self, x, *args, reverse=False, eps=1e-5):
        """
        x: (sz_batch, seqlen, dim=1)
        """
        assert x.size(-1) == 1

        u = self.u.abs()
        v = self.v.abs()
        b = self.b

        if not reverse:

            affine = x * v + b # (sz_batch, seqlen, k)
            y = (u * torch.tanh(affine)).mean(dim=-1, keepdim=True)  # (sz_batch, seqlen, 1)
            dy_dx = (deriv_tanh(affine) * u * v).mean(dim=-1) # (sz_batch, seqlen)
            log_dy_dx = dy_dx.log()
        else:
            def f(inputs):
                # inputs: (sz_batch, seqlen, 1)
                affine = inputs * v + b # (sz_batch, seqlen, k)
                return (u * torch.tanh(affine)).mean(dim=-1, keepdim=True)

            lo = torch.full_like(x, -1.0e3)
            loval = f(lo)
            hi = torch.full_like(x, 1.0e3)
            hival = f(hi)

            with torch.no_grad():
                for i in range(100):
                    mid = (lo + hi) * 0.5
                    midval = f(mid)

                    lo = torch.where(x >= midval, mid, lo)
                    hi = torch.where(x < midval, mid, hi)

                    if torch.all(torch.abs(hi - lo) < 1e-5):
                        break
                y = mid

            affine = y * v + b # (sz_batch, seqlen, k)
            dx_dy = (deriv_tanh(affine) * u * v).mean(dim=-1) # (sz_batch, seqlen)
            log_dy_dx = - dx_dy.log()
        
        return y, -log_dy_dx
    '''
    
class TanhMixture(Flow, nn.Module):
    def __init__(self, k=5, bias=True):
        nn.Module.__init__(self)
        self.u = nn.Parameter(torch.randn(k) * 0.01 + 1.0)
        self.v = nn.Parameter(torch.randn(k) * 0.01 + 1.0)
        if bias:
            self.b = nn.Parameter(torch.randn(k) * 0.01)
        else:
            self.register_buffer("b", torch.zeros(k))

    def _forward(self, x, *args, reverse=False, eps=1e-5):
        """
        x: (sz_batch, seqlen, dim=1)
        """
        assert x.size(-1) == 1

        u = self.u.abs()
        v = self.v.abs()
        b = self.b

        if not reverse:

            affine = x * v + b # (sz_batch, seqlen, k)
            y = (u * torch.tanh(affine)).mean(dim=-1, keepdim=True)  # (sz_batch, seqlen, 1)
            dy_dx = (deriv_tanh(affine) * u * v).mean(dim=-1) # (sz_batch, seqlen)
            #--------for dy_dx==0
            dy_dx = torch.where(dy_dx==torch.tensor(0.0), eps, dy_dx)
            #---------
            log_dy_dx = dy_dx.log()
        else:
            def f(inputs):
                # inputs: (sz_batch, seqlen, 1)
                affine = inputs * v + b # (sz_batch, seqlen, k)
                return (u * torch.tanh(affine)).mean(dim=-1, keepdim=True)

            lo = torch.full_like(x, -1.0e3)
            loval = f(lo)
            hi = torch.full_like(x, 1.0e3)
            hival = f(hi)

            with torch.no_grad():
                for i in range(100):
                    mid = (lo + hi) * 0.5
                    midval = f(mid)

                    lo = torch.where(x >= midval, mid, lo)
                    hi = torch.where(x < midval, mid, hi)

                    if torch.all(torch.abs(hi - lo) < 1e-5):
                        break
                y = mid

            affine = y * v + b # (sz_batch, seqlen, k)
            dx_dy = (deriv_tanh(affine) * u * v).mean(dim=-1) # (sz_batch, seqlen)
            log_dy_dx = - dx_dy.log()
        return y, -log_dy_dx    