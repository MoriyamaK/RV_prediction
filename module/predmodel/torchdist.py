'''
A wrapper of arch.univariate.distribution
'''


import math
import numpy as np
from scipy import stats
import torch
from arch.univariate import distribution

def tensor2array(x):
    return x.data.cpu().numpy()

class Normal:
    ''' batched 1-dim normal distribution'''

    def __init__(self, mean, std):
        assert mean.shape == std.shape, f'{mean.shape}, {std.shape}'
        self._shape = mean.shape

        self.mean = mean
        self.std = std
    
    def log_prob(self, x):
        '''
        Parameters
        ----------
        x: (*shape)

        Returns:logl: (*shape)
        ----------
        '''
        assert x.shape == self._shape

        std2 = self.std**2
        logl = -0.5 * (math.log(2*math.pi) + \
                       torch.log(std2) + \
                       (x-self.mean)**2/std2)
        return logl

    def sample(self, n):
        '''
        Parameters
        ----------
        n: integer

        Returns
        ----------
        samples: (n, *shape)
        '''
        mean, std = self.mean, self.std
        return torch.distributions.Normal(mean, std).sample((n,))


    def icdf(self, p):
        '''
        Parameters
        ----------
        p: scalar within (0, 1)

        Returns
        ----------
        y: percentile of the shape `self._shape`
        '''
        mean, std = self.mean, self.std
        return torch.tensor(stats.norm.ppf(p, loc=mean.data.cpu().numpy(), 
                                           scale=std.data.cpu().numpy()),
                            device=self.mean.device, 
                            dtype=self.mean.dtype)


class StudentsT:
    ''' batched 1-dim Student's t distribution '''

    def __init__(self, mean, std, nu):
        '''
        Parameters
        ----------
        mean: (*shape)
        std: (*shape)
        nu: (*shape) degree of freedom
        '''
        self.mean = mean
        self.std = std
        self.nu = nu

        assert mean.shape == std.shape == nu.shape, f'{mean.shape}, {std.shape}, {nu.shape}'
        self._shape = mean.shape

    def log_prob(self, x):
        '''
        Parameters
        ----------
        x: (*shape)

        Returns
        ----------
        logl: (*shape)
        '''
        assert x.shape == self._shape

        mean, std, nu = self.mean, self.std, self.nu
        std2 = std**2

        logl = torch.special.gammaln((nu + 1) / 2) - torch.special.gammaln(nu / 2) \
               - 0.5 * torch.log(math.pi * (nu - 2)) \
               - 0.5 * torch.log(std2) \
               - ((nu + 1) / 2) * (torch.log(1 + ((x-mean) ** 2) / (std2 * (nu - 2))))
        return logl

    def sample(self, n):
        mean, std, nu = list(map(tensor2array, [self.mean, self.std, self.nu]))
        shape = self._shape

        mean, std, nu = mean.reshape(-1), std.reshape(-1), nu.reshape(-1)
        samples = []
        for m, s, n_ in zip(mean, std, nu):
            sample = distribution.StudentsT().simulate([n_])(n)
            samples.append(sample*s + m)
        samples = np.stack(samples, axis=1).reshape(n, *shape)
        
        return torch.tensor(samples, dtype=self.mean.dtype, device=self.mean.device)


    def icdf(self, p):
        '''
        Parameters
        ----------
        p: scalar within (0, 1)

        Returns
        ----------
        y: percentile of the shape `self._shape`
        '''

        mean, std, nu = self.mean, self.std, self.nu

        standard_std = (nu / (nu - 2)) ** 0.5
        scale = std / standard_std
        return torch.tensor(stats.t.ppf(p, df=tensor2array(nu),
                                        loc=tensor2array(mean),
                                        scale=tensor2array(scale)),
                            device=mean.device,
                            dtype=mean.dtype)
        
class SkewStudentsT:
    ''' batched 1-dim skewed Student's t distribution '''

    def __init__(self, mean, std, nu, lambd):
        '''
        Parameters
        ----------
        mean: (*shape)
        std: (*shape)
        nu: (*shape) degree of freedom
        '''
        self.mean = mean
        self.std = std
        self.nu = nu
        self.lambd = lambd

        assert mean.shape == std.shape == nu.shape == lambd.shape, f'{mean.shape}, {std.shape}, {nu.shape}, {lambd.shape}'
        self._shape = mean.shape

    def log_prob(self, x):
        '''
        Parameters
        ----------
        x: (*shape)

        Returns
        ----------
        logl: (*shape)
        '''
        assert x.shape == self._shape

        mean, std, nu = self.mean, self.std, self.nu
        std2 = std**2

        logl = torch.special.gammaln((nu + 1) / 2) - torch.special.gammaln(nu / 2) \
               - 0.5 * torch.log(math.pi * (nu - 2)) \
               - 0.5 * torch.log(std2) \
               - ((nu + 1) / 2) * (torch.log(1 + ((x-mean) ** 2) / (std2 * (nu - 2))))
        return logl

    def sample(self, n):
        mean, std, nu, lambd = list(map(tensor2array, [self.mean, self.std, self.nu, self.lambd]))
        shape = self._shape

        mean, std, nu, lambd = mean.reshape(-1), std.reshape(-1), nu.reshape(-1), lambd.reshape(-1)
        samples = []
        for m, s, n_, l in zip(mean, std, nu, lambd):
            sample = distribution.SkewStudent().simulate([n_, l])(n)
            samples.append(sample*s + m)
        samples = np.stack(samples, axis=1).reshape(n, *shape)
        
        return torch.tensor(samples, dtype=self.mean.dtype, device=self.mean.device)


    def icdf(self, p):
        '''
        Parameters
        ----------
        p: scalar within (0, 1)

        Returns
        ----------
        y: percentile of the shape `self._shape`
        '''

        mean, std, nu, lambd = list(map(tensor2array, [self.mean, self.std, self.nu, self.lambd]))
        
        a = tensor2array(self._a(self.nu, self.lambd))
        b = tensor2array(self._b(self.nu, self.lambd))

        icdf = -999.99 * np.ones_like(lambd, dtype=np.float32)
        cond = p < (1 - lambd) / 2
        standard_std = 1
        scale = std / standard_std
        icdf[cond] = stats.t.ppf(p / (1 - lambd[cond]), nu[cond], scale=scale[cond])
        icdf[~cond] = stats.t.ppf(0.5 + (p - (1 - lambd[~cond]) / 2) / (1 + lambd[~cond]), nu[~cond], scale=scale[~cond])

        icdf = icdf * (1 + np.sign(p - (1 - lambd) / 2) * lambd) * (1 - 2 / nu) ** 0.5 - a
        icdf = icdf / b

        return torch.tensor(icdf + mean,
                            device=self.mean.device,
                            dtype=self.mean.dtype)

    def _a(self, nu, lambd):
        """
        Compute a constant.

        Parameters
        ----------
        parameters : ndarray
            Shape parameters of the skew-t distribution

        Returns
        -------
        a : float
            Constant used in the distribution

        """
        c = self._c(nu, lambd)
        return 4 * lambd * torch.exp(c) * (nu - 2) / (nu - 1)

    def _b(self, nu, lambd):
        """
        Compute b constant.

        Parameters
        ----------
        parameters : ndarray
            Shape parameters of the skew-t distribution

        Returns
        -------
        b : float
            Constant used in the distribution
        """
        a = self._a(nu, lambd)
        return (1 + 3 * lambd ** 2 - a ** 2) ** 0.5

    def _c(self, nu, lambd):
        """
        Compute c constant.

        Parameters
        ----------
        parameters : ndarray
            Shape parameters of the skew-t distribution

        Returns
        -------
        c : float
            Log of the constant used in loglikelihood
        """
        # return gamma((eta+1)/2) / ((pi*(eta-2))**.5 * gamma(eta/2))
        return torch.special.gammaln((nu + 1) / 2) - torch.special.gammaln(nu / 2) - torch.log(math.pi * (nu - 2)) / 2
    


class GeneralizedError:
    ''' batched 1-dim generalized error distribution '''

    def __init__(self, mean, std, nu):
        '''
        Parameters
        ----------
        mean: (*shape)
        std: (*shape)
        nu: (*shape) degree of freedom
        '''
        self.mean = mean
        self.std = std
        self.nu = nu

        assert mean.shape == std.shape == nu.shape, f'{mean.shape}, {std.shape}, {nu.shape}'
        self._shape = mean.shape

    def log_prob(self, x):
        '''
        Parameters
        ----------
        x: (*shape)

        Returns
        ----------
        logl: (*shape)
        '''
        assert x.shape == self._shape

        mean, std, nu = self.mean, self.std, self.nu
        std2 = std**2

        log_c = 0.5 * (-2 / nu * math.log(2) + torch.special.gammaln(1 / nu) - torch.special.gammaln(3 / nu))
        c = log_c.exp()
        logl = nu.log() - log_c - torch.special.gammaln(1 / nu) - (1 + 1 / nu) * math.log(2) \
               - 0.5 * std2.log() \
               - 0.5 * (x / (std2**0.5 * c)).abs() ** nu

        return logl

    def sample(self, n):
        mean, std, nu = list(map(tensor2array, [self.mean, self.std, self.nu]))
        shape = self._shape

        mean, std, nu = mean.reshape(-1), std.reshape(-1), nu.reshape(-1)
        samples = []
        for m, s, n_ in zip(mean, std, nu):
            sample = distribution.GeneralizedError().simulate([n_])(n)
            samples.append(sample*s + m)
        samples = np.stack(samples, axis=1).reshape(n, *shape)
        
        return torch.tensor(samples, dtype=self.mean.dtype, device=self.mean.device)


    def icdf(self, p):
        '''
        Parameters
        ----------
        p: scalar within (0, 1)

        Returns
        ----------
        y: percentile of the shape `self._shape`
        '''

        mean, std, nu = list(map(tensor2array, [self.mean, self.std, self.nu]))

        return torch.tensor(stats.gennorm.ppf(p, nu, scale=std, loc=mean),
                            device=self.mean.device,
                            dtype=self.mean.dtype)