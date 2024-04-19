from functools import partial
import os
pool_size = int(os.environ.get('ARCH_POOLSIZE', 4))
from multiprocessing import Pool
pool = Pool(pool_size)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .base import Dynamics
from . import torchdist
import statsmodels.api as sm
import os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import torch
from arch.univariate import HARX, GARCH, HARCH
from arch.univariate import Normal, StudentsT, SkewStudent, GeneralizedError
import scipy

class HAR(Dynamics):
    def __init__(self, distribution = 'normal'):
        self.distribution = distribution
        
    
    def _fit(self, x, last_obs):
        # x: (sz_batch, steps, dim)
        sz_batch, steps, dim = x.shape
        if dim > 1:
            means, vars, params = self._mult_fit(x, last_obs)
            #means vars: (sz_batch, steps+1, dim)
        else:
            #x = x.squeeze(dim=2)
            # x: (sz_batch, steps)
            device, dtype = x.device, x.dtype
            params = []
            vars = []
            means = []
            for i in range(len(x)):
                sample = x[i]
                #print(sample.shape)
                if False:
                    model = self._train_model(sample[:last_obs].contiguous(), last_obs)
                    mean, var = self._simulate_model(model, sample.contiguous(), last_obs)
                    params.append(model['params'])
                elif False:
                    model = self._train_model_tensor(sample[:last_obs].contiguous(), last_obs)
                    mean, var = self._simulate_model_tensor(model, sample.contiguous(), last_obs)
                    params.append(model['params'].data.cpu().numpy().astype(np.float64))
                elif False:
                    model, mean, var = self._apply_model_harch(sample, last_obs)
                    params.append(model['params'])
                else:
                    model, mean, var = self._apply_model(sample, last_obs)
                    params.append(model['params'])
                vars.append(var)
                means.append(mean)
            #params: (sz_batch, 5(4)), vars: (sz_batch, steps+1)
            '''
            vars = torch.stack(vars).unsqueeze(dim=2)
            means = torch.stack(means).unsqueeze(dim=2)
            '''
            
            vars = torch.tensor(np.stack(vars), dtype=dtype, device=device) \
            if isinstance(vars[0], np.ndarray) else torch.stack(vars)
            means = torch.tensor(np.stack(means), dtype=dtype, device=device) \
            if isinstance(means[0], np.ndarray) else torch.stack(means)
            
            vars = vars.unsqueeze(dim=2)
            means = means.unsqueeze(dim=2)
            
            params = pd.DataFrame(params) 
        return means, vars, params
    
    def _apply_model(self, x, last_obs):
        x = x.squeeze(dim=1) #x:(steps, )
        x = x.data.cpu().numpy().astype(np.float64)
        harx = HARX(x, lags=[1, 5, 22], rescale=False)
        if self.distribution == 'normal':
            harx.distribution = Normal()
        elif self.distribution == 't':
            harx.distribution = StudentsT()
        elif self.distribution == 'skewt':
            harx.distribution = SkewStudent()
        elif self.distribution == 'ged':
            harx.distribution = GeneralizedError()
        harx.constraints()
        result = harx.fit(last_obs=last_obs, disp='off')
        f = harx.forecast(result.params, align='target',reindex=True, horizon=1, start=21)
        mean = f.mean.values.reshape(-1)
        res_var = f.residual_variance.values.reshape(-1)
        '''
        mp, _, _ = harx._parse_parameters(result.params)
        resids = harx.resids(mp)
        print(resids.shape) #resids(last_obs-22, )
        print(res_var.shape) #res_var(steps, )
        print(res_var[23] - (resids**2).mean())
        '''
        res_var[22] = res_var[23]
        mean = np.append(mean, np.nan)
        res_var = np.append(res_var, np.nan)
        return   {'params':result.params}, mean, res_var
    
    def _apply_model_harch(self, x, last_obs):
        x = x.squeeze(dim=1) #x:(steps, )
        x = x.data.cpu().numpy().astype(np.float64)
        harx = HARX(x, lags=[1, 5, 22], rescale=False)
        harx.distribution = Normal()
        harx.volatility = HARCH(lags=[1,5,22])
        result = harx.fit(last_obs=last_obs, disp='off', show_warning=False)
        f = harx.forecast(result.params, align='target',reindex=True, horizon=1, start=21)
        mean = f.mean.values.reshape(-1)
        res_var = f.residual_variance.values.reshape(-1)
        res_var[22] = res_var[23]
        mean = np.append(mean, np.nan)
        res_var = np.append(res_var, np.nan)
        return   {'params':result.params}, mean, res_var
    
    def _nll(self, x, last_obs):
        sz_batch, steps, dim = x.shape
        assert dim == 1 or sz_batch==1
        if last_obs == -1:
            last_obs = steps
        assert min(steps, last_obs) >= 10, 'Least timesteps for estimation is 10.'
        
        if dim == 1:
            means, vars, params = self._fit(x, last_obs)#means vars: (sz_batch, steps+1, dim)
            means = means[:, :-1].squeeze(dim=2).contiguous()
            vars = vars[:, :-1].squeeze(dim=2).contiguous()
            # means, vars: (sz_batch, steps)
            x = x.squeeze(dim=2)
            nll = torch.zeros_like(x)
            nll[:, :] = torch.nan
            if self.distribution == 'normal':
                nll[:, 22:] = - torchdist.Normal(means[:, 22:], vars[:, 22:]**0.5).log_prob(x[:, 22:])
            elif self.distribution == 't':
                nu = torch.tensor(params['nu'].values, device=x.device, dtype=x.dtype) \
                    .view(-1, 1).expand(-1, steps).contiguous()
                #print(x)
                nll[:, 22:]  = - torchdist.StudentsT(means[:, 22:] , vars[:, 22:] **0.5, nu[:, 22:]).log_prob(x[:, 22:])
            elif self.distribution == 'skewt':
                nu = torch.tensor(params['eta'].values, device=x.device, dtype=x.dtype) \
                    .view(-1, 1).expand(-1, steps).contiguous()
                lambd = torch.tensor(params['lambda'].values, device=x.device, dtype=x.dtype) \
                        .view(-1, 1).expand(-1, steps).contiguous()
                nll[:, 22:]  = - torchdist.SkewStudentsT(means[:, 22:], vars[:, 22:] **0.5, nu[:, 22:], lambd[:, 22:]).log_prob(x[:, 22:])
            elif self.distribution == 'ged':
                nu = torch.tensor(params['nu'].values, device=x.device, dtype=x.dtype) \
                    .view(-1, 1).expand(-1, steps).contiguous()
                nll[:, 22:]  = - torchdist.GeneralizedError(means[:, 22:] , vars[:, 22:] **0.5, nu[:, 22:]).log_prob(x[:, 22:])
        else:
            means, vars, _= self._mult_fit(x, last_obs)#means vars: (sz_batch, steps+1, dim)
            means = means[:, :-1, :].contiguous()
            vars = vars[:, :-1, :].contiguous()
            nll = torch.zeros_like(x)
            nll[:, :, :] = torch.nan
            if self.distribution == 'normal':
                nll[:, 22:, :] = - torchdist.Normal(means[:, 22:, :], vars[:, 22:, :]**0.5).log_prob(x[:, 22:, :])
            
                
            '''
            idx_list = []
            i = 1
            while int(i * (i + 1) / 2) - 1 < dim:
                idx_list.append(int(i * (i + 1) / 2) - 1)
                i += 1
            '''
            
            nll = nll[:, :, :].mean(dim=2)
            #print(means)
                
        return nll#nll: (sz_batch, steps)
    #first 22 elements in nll is NaN
            
    def _train_model(self, x, last_obs):
        #print(x.shape)
        x = x.squeeze(dim=1) #x:(steps, )
        x = x.data.cpu().numpy().astype(np.float64)
        harx = HARX(x[:last_obs], lags=[1, 5, 22], rescale=False)
        harx.distribution = Normal()
        result = harx.fit()
        #print(result)
        return {
            'params': result.params
        }
        
        
    def _train_model_tensor(self, x, last_obs):
        
        #x: (steps, dim=1)
        #x = x.squeeze(dim=1) #x:(steps, )
        d,w,m = self._preprocess_tensor(x)
        cons = torch.ones_like(d)
        regressors = torch.hstack((cons, d, w, m))
        #print(regressors.shape)
        params = torch.linalg.pinv(regressors[22:last_obs]).mm(x[22:last_obs])
        return {
            'params': params.squeeze(dim=1)
        }   
        
    def _mult_train_model(self, x):
        #x: (last_obs, dim)
        #print(x.shape)
        #print(x[:, 0] )
        last_obs, dim = x.shape
        x = x.data.cpu().numpy().astype(np.float64)
        d, w, m = self._preprocess(x)
        
        '''
        idx_list = []
        i = 1
        while int(i * (i + 1) / 2) - 1 < dim:
            idx_list.append(int(i * (i + 1) / 2) - 1)
            i += 1
        '''
        
        c, beta_d, beta_w, beta_m = [], [], [], []
        for i in range(dim):
            harx = HARX(x[:last_obs, i], lags=[1, 5, 22], rescale=False)
            harx.distribution = Normal()
            result = harx.fit()
            #print(i)
            c.append(result.params[0])
            beta_d.append(result.params[1])
            beta_w.append(result.params[2])
            beta_m.append(result.params[3])
            #print(result)
        
        
        '''
        k = 1 
        def NLL(theta):
            c, beta_d, beta_w, beta_m = theta[:k], theta[k:2*k], \
                theta[2*k:3*k], theta[3*k:4*k]
            #c, beta_d, beta_w, beta_m = theta
            pred_x = c + beta_d * d + beta_w * w + beta_m * m
            mse = ((x[22:, :] - pred_x[22, :])**2).mean()
            return mse
        
        
        #theta_zero = np.full((dim * 4, ), 0.1) # supposed that c is  (dim, )vector
        theta_zero = np.full((1+3, ), 0.01) #supposed that c is scalar  
        bounds = scipy.optimize.Bounds(np.full(theta_zero.shape, -0.2), \
            np.full(theta_zero.shape, 0.2))
        
        
        result = scipy.optimize.minimize(NLL, theta_zero, method='SLSQP', \
            bounds=bounds, \
            tol=1e-10)
        
        #print(result)
        '''
        
        return {
            'params': np.concatenate([c, beta_d, beta_w, beta_m]),
            #'params': result.x,
        }
        
    def _mult_simulate_model(self, model, x, last_obs):
        # x: (steps, dim)
        steps, dim = x.shape #x (steps, dim)
        device, dtype = x.device, x.dtype
        x = x.data.cpu().numpy().astype(np.float64)
        d, w, m = self._preprocess(x)
        #c, beta_d, beta_w, beta_m = result.x
        params = model['params']
        
        #-----------------------
        k = dim
        #k = 1# for scalar
        #-----------------------
        
        c, beta_d, beta_w, beta_m = params[:k], params[k:2*k], \
            params[2*k:3*k], params[3*k:4*k]
        pred_x = c + beta_d * d + beta_w * w + beta_m * m
        resids, vars, means = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        resids, vars[:, :], means = x - pred_x, np.nan, pred_x
        vars[22:, :] = (resids[22:last_obs, :]**2).mean(axis=0)
        #print(means.shape) 
        #means, vars: (steps, dim)
        means = np.vstack([means, np.zeros(dim)])
        vars = np.vstack([vars, np.ones(dim)])
        #means, vars: (steps+1, dim) add dummy 
        
        return means, vars
        
            
    def _preprocess(self, x):
        #x: (last_obs or steps, dim)
        #X_t = c + beta_d * d_t + beta_w * w_t + beta_m * m_t
        d, w, m = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        steps, _ = x.shape
        d[:, :], w[:, :], m[:, :] = np.nan, np.nan, np.nan
        
        d[1:, :] = x[:-1, :]
        w[5:, :] = np.array([x[i-5:i, :].mean(axis=0) for i in range(5, steps)])
        m[22:, :] = np.array([x[i-22:i, :].mean(axis=0) for i in range(22, steps)])
        return d, w, m
            
    def _preprocess_tensor(self, x):
        #x: (last_obs or steps, dim)
        #X_t = c + beta_d * d_t + beta_w * w_t + beta_m * m_t
        d, w, m = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
        steps, _ = x.shape
        d[:, :], w[:, :], m[:, :] = torch.nan, torch.nan, torch.nan
        
        d[1:, :] = x[:-1, :]
        
        w[5:, :] = torch.tensor([x[i-5:i, :].mean(dim=0) for i in range(5, steps)]).unsqueeze(dim=1)
        m[22:, :] = torch.tensor([x[i-22:i, :].mean(dim=0) for i in range(22, steps)]).unsqueeze(dim=1)
        return d, w, m
    
    def _simulate_model(self, model, x, last_obs):
        params = model['params']
        steps, dim= x.shape 
        #print(x.shape)
        device, dtype = x.device, x.dtype
        
        #x = x.squeeze(dim=1)
        x = x.data.cpu().numpy().astype(np.float64)
        
        resids, vars, means = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        resids[:], vars[:], means[:] = np.nan, np.nan, np.nan
        '''
        harx = HARX(x, lags=[1, 5, 22], rescale=False)
        harx.distribution = Normal()
        harx.fit()
        f = harx.forecast(params, align='target',reindex=True, horizon=1, start=21)
        resids[22:] = harx.resids(params[:4])
        means = f.mean['h.1']
        
        #print(means)
        means[22:] = x[22:] - resids[22:]
        #print((resids[22:last_obs]**2).mean(axis=0))
        
        #res_vars = harx.forecast(params, start=21, align='target', reindex=True, method='simulation').residual_variance
        vars[22:] = (resids[22:last_obs]**2).mean(axis=0) 
        #print(vars[22:])
        #means, vars: (steps, )
        means = np.append(means, np.nan)
        vars = np.append(vars, np.nan)
        #means, vars: (steps + 1, ) add dummy 
        '''
        
        
        d, w, m = self._preprocess(x)
        c, beta_d, beta_w, beta_m, _ = model['params']
        pred_x = c + beta_d * d + beta_w * w + beta_m * m
        resids, vars, means = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        resids, vars[:, :], means = x - pred_x, np.nan, pred_x
        #print(means)
        #vars[22:] = (resids[22:last_obs]**2).mean()
        
        term = 22
        vars[22:22+term] = (resids[22:22+term]**2).mean()
        for i in range(22+term, len(vars)):
            vars[i] = (resids[i-term :i]**2).mean()
        
        
        #means, vars: (steps, dim)
        means = np.vstack([means, np.zeros(dim)])
        vars = np.vstack([vars, np.ones(dim)])
        means = means.reshape(means.size)
        vars = vars.reshape(vars.size)
        return means, vars
        
        '''
        d, w, m = self._preprocess_tensor(x)
        c, beta_d, beta_w, beta_m, _ = model['params']
        pred_x = c + beta_d * d + beta_w * w + beta_m * m
        resids, vars, means = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
        resids, vars[:, :], means = x - pred_x, torch.nan, pred_x
        term = 5
        vars[22:last_obs] = (resids[22:22+term]**2).mean()
        #vars[22:] = (resids[22:last_obs]).var()
        
        for i in range(22+term, len(vars)):
            vars[i] = (resids[i-term :i]**2).mean()
        
        #means, vars: (steps, dim)
        #print(means.device)
        means = torch.vstack([means, torch.zeros(dim, device=device)])
        vars = torch.vstack([vars, torch.ones(dim, device=device)])
        means = means.squeeze(dim=1)
        vars = vars.squeeze(dim=1)
        return means, vars
        '''
        
    def _mult_fit(self, x, last_obs):
        sz_batch, steps, dim = x.shape
        device, dtype = x.device, x.dtype
        #print(x)
        params = []
        vars = []
        means = []
        for i in range(len(x)):
            sample = x[i]
            #print(sample)
            model = self._mult_train_model(sample[:last_obs].contiguous())
            mean, var = self._mult_simulate_model(model, sample.contiguous(), last_obs)
            params.append(model['params'])
            #print(mean.shape)
            vars.append(var)
            means.append(mean)
        #params: (sz_batch, 4), vars, means: (sz_batch, steps+1, dim)
        vars = torch.tensor(np.stack(vars), dtype=dtype, device=device) \
            if isinstance(vars[0], np.ndarray) else torch.stack(vars)
        means = torch.tensor(np.stack(means), dtype=dtype, device=device) \
            if isinstance(means[0], np.ndarray) else torch.stack(means)
        params = pd.DataFrame(params)
        return means, vars, params 
        
    def _simulate_model_tensor(self, model, x, last_obs):
        steps, dim= x.shape 
        device, dtype = x.device, x.dtype
        d, w, m = self._preprocess_tensor(x)
        c, beta_d, beta_w, beta_m = model['params']
        pred_x = torch.zeros_like(x)
        pred_x[:, :] = torch.nan
        pred_x[22:] = c + beta_d * d[22:] + beta_w * w[22:] + beta_m * m[22:]
        resids, vars, means = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
        resids, vars[:, :], means = x - pred_x, torch.nan, pred_x
        vars[22:] = (resids[22:last_obs]**2).mean()
        
        #means, vars: (steps, dim)
        #print(means.device)
        means = torch.vstack([means, torch.zeros(dim, device=device)])
        vars = torch.vstack([vars, torch.ones(dim, device=device)])
        means = means.squeeze(dim=1)
        vars = vars.squeeze(dim=1)
        return means, vars
        
    
    
    
    def _forecast_sampling(self,x, last_obs, n_sampling_paths):
        '''
        One-step ahead forecasting of `x` by sampling.

        Parameters
        ----------
        x: data of shape (sz_batch, steps, dim=1).

        last_obs: 
            Use x[:, :last_obs] for volatility inference.
            But if `last_obs`==-1, `last_obs` will be set to `steps`
                so that all timesteps of `x` are used for inference.
            `last_obs` must be at least 10.
        
        Returns
        ---------
        y: samples from the GARCH predmodel following time step `last_obs`
            of shape (n_sampling_paths, sz_batch, steps-last_obs+1, dim=1)
        '''
        sz_batch, steps, dim = x.shape
        #print(x.shape, last_obs)
        assert dim == 1
        if last_obs == -1:
            last_obs = steps
        assert min(steps, last_obs) >= 10, 'Least timesteps for estimation is 10.'

        means, vars, params = self._fit(x, last_obs)
        # means, vars: (sz_batch, steps+1, dim=1)
        means = means[:, last_obs:].squeeze(dim=2).contiguous()
        vars = vars[:, last_obs:].squeeze(dim=2).contiguous()
        if self.distribution == 'normal':
            samples = torchdist.Normal(means[:, 22:], vars[:, 22:]**0.5).sample(n_sampling_paths)
        
        samples = samples.unsqueeze(3)
        # samples: (n_sampling_paths, sz_batch, steps-last_obs+1, 1)
        return samples
    
    def _sampling(self, x, last_obs, n_sampling_paths):
        sz_batch, steps, dim = x.shape
        assert dim == 1
        if last_obs == -1:
            last_obs = steps
        assert min(steps, last_obs) >= 10, 'Least timesteps for estimation is 10.'

        means, vars, params = self._fit(x, last_obs)
        # means, vars: (sz_batch, steps+1, dim=1)
        means = means.squeeze(dim=2).contiguous()
        vars = vars.squeeze(dim=2).contiguous()
        samples = torch.zeros(n_sampling_paths, sz_batch, steps + 1, device=x.device)
        samples[:, :, :] = torch.nan
        #print(means.shape, vars.shape)
        if self.distribution == 'normal':
            samples[:, :, 22:] = torchdist.Normal(means[:, 22:], vars[:, 22:]**0.5).sample(n_sampling_paths)
        
        samples = samples.unsqueeze(3)
        # samples: (n_sampling_paths, sz_batch, steps+1, 1)
        return samples
    
    def _forecast_VaR(self, x, last_obs, epsilon):
        return super()._forecast_VaR(x, last_obs, epsilon)
        
        


