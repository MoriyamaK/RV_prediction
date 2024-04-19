from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import logging
logger = logging.getLogger()


class GCFPBase(metaclass=ABCMeta):

    @abstractmethod
    def nll(self, *args, **kwargs):
        return NotImplementedError
    
    def forward(self, *args, **kwargs):
        return self.nll(*args, **kwargs)

    

class GCFP1D(GCFPBase, nn.Module):

    def __init__(self, predmodel, flow):
        super(GCFP1D, self).__init__()
        self.predmodel = predmodel
        self.flow = flow
    
    def nll(self, x, phi, last_obs=-1, return_auxiliary_losses=False):
        '''
        Parameters
        ----------
        x: data of shape (sz_batch, steps, dim=1).

        phi: (sz_batch, extra_dim)

        last_obs: 
            Use x[:, :last_obs] for parameter re-estimation.
            But if `last_obs`==-1, `last_obs` will be set to `steps`
                so that all timesteps of `x` are used for re-estimation.
            But if `last_obs`==0, skip re-estimation.
        
        Return
        ---------
        nll: negative-loglikelihood (sz_batch, steps)
            Note that nlls for the training steps are also returned.
        logdet: log-determinant of the flow at `x`.
        '''
        sz_batch, steps, dim = x.shape
        #assert dim == 1
        if last_obs == -1:
            last_obs = steps
        total_logdet = 0
        phi = phi.unsqueeze(dim=1).expand(-1, x.shape[1], -1)
        # phi: (sz_batch, 0:steps, extra_dim)
        
        
        # ------------- compute nll in latent space ---------
        y, logdet, flow_aux_losses = self.flow.forward(x, phi, return_auxiliary_losses=return_auxiliary_losses)   
        nll, dyna_aux_losses = self.predmodel.nll(y, last_obs, return_auxiliary_losses=return_auxiliary_losses)
        #nll: (sz_batch, steps)
        
        logdet /= dim
        
        total_logdet = logdet #+logdet_scale +  logdet_invscale
              
        if return_auxiliary_losses:
            return nll, total_logdet, {**flow_aux_losses, **dyna_aux_losses}    
        else:
            return nll, total_logdet
    



        
    def se(self, x, phi, last_obs=-1, 
            return_auxiliary_losses=False, lag=22):
        _, steps, dim = x.shape
    
        sz_batch, steps, dim = x.shape
        assert dim == 1 or sz_batch == 1
        if last_obs == -1:
            last_obs = steps
        if phi.dim() == 2:
            phi = phi.unsqueeze(dim=1).expand(-1, x.shape[1], -1)
        # # phi: (sz_batch, 0:steps, extra_dim)
        
        y, logdet, _ = self.flow.forward(x, phi, return_auxiliary_losses=return_auxiliary_losses)               
        y_, _, _ = self.predmodel.fit(y, last_obs) 
        #y_: (sz_batch, steps+1, dim)
        y_ = y_[:, :-1, :].contiguous() 
        latent_resids = torch.zeros_like(y_)
        latent_resids[:, :lag, :] = torch.nan
        latent_resids[:, lag:, :] = y[:, lag:, :] - y_[:, lag:, :]
        latent_resids_2 = ((latent_resids[:, :, :]**2).mean(dim=2).contiguous()) 
        #y_: (sz_batch, steps, dim)
        
        
        x_ = torch.zeros_like(y_)
        x_[:, :lag, :] = torch.nan
        x_[:, lag:, :], logdet_inverse, _ = self.flow.forward(y_[:, lag:, :].contiguous(), phi[:, lag:, :], return_auxiliary_losses=return_auxiliary_losses, reverse=True)
        resids = x - x_
        resids_2 = ((resids[:, :, :]**2).mean(dim=2).contiguous()) 
        
        return resids_2,  latent_resids_2, logdet, logdet_inverse
    
    
    
    def ae(self, x, phi, last_obs=-1,
            return_auxiliary_losses=False,lag=22):
        _, steps, dim = x.shape
            
        sz_batch, steps, dim = x.shape
        assert dim == 1 or sz_batch == 1
        if last_obs == -1:
            last_obs = steps
        # x: (sz_batch, 0:steps, dim=1)
        if phi.dim() == 2:
            phi = phi.unsqueeze(dim=1).expand(-1, x.shape[1], -1)
        # phi: (sz_batch, 0:steps, extra_dim)

        y, logdet1, _ = self.flow.forward(x, phi, return_auxiliary_losses=return_auxiliary_losses)
            # y: (sz_batch, 0:steps, dim=1)
            

        y_, _, _ = self.predmodel.fit(y, last_obs) if dim ==1 \
            else self.predmodel.mult_fit(y, last_obs)
        #y_: (sz_batch, steps+1, dim)
        y_ = y_[:, :-1, :].contiguous() 
        latent_resids = torch.zeros_like(y_)
        latent_resids[:, :lag, :] = torch.nan
        latent_resids[:, lag:, :] = y[:, lag:, :] - y_[:, lag:, :]
        latent_resids_2 = ((latent_resids[:, :, :].abs()).mean(dim=2).contiguous()) 
        #y_: (sz_batch, steps, dim)
        
        
        x_ = torch.zeros_like(y_)
        x_[:, :lag, :] = torch.nan
        x_[:, lag:, :], logdet2, _ = self.flow.forward(y_[:, lag:, :].contiguous(), phi[:, lag:, :], return_auxiliary_losses=return_auxiliary_losses, reverse=True)
        resids = x - x_        
        resids_2 = ((resids[:, :, :].abs()).mean(dim=2).contiguous()) 
        #resids_2 = ((resids[:, :, idx_list]**2).mean(dim=2).contiguous()) 
        
        return resids_2,  latent_resids_2, logdet1, logdet2 
