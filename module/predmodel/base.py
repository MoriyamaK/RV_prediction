from abc import ABCMeta, abstractmethod
import inspect

class Dynamics(metaclass=ABCMeta):

    def nll(self, x, last_obs, return_auxiliary_losses: bool=False):
        spec = inspect.getfullargspec(self._nll)
        argnames = spec[0]


        if 'return_auxiliary_losses' not in argnames:
            nll = self._nll(x, last_obs)
            aux_losses = {}

        else:
            if return_auxiliary_losses:
                nll, aux_losses = self._nll(x, last_obs, return_auxiliary_losses=return_auxiliary_losses)
            else:
                nll = self._nll(x, last_obs, return_auxiliary_losses=return_auxiliary_losses)
                aux_losses = {}

        return nll, aux_losses

    def forecast_sampling(self, x, last_obs, n_sampling_paths):
        return self._forecast_sampling(x, last_obs, n_sampling_paths)
    
    def sampling(self, x, last_obs, n_sampling_paths):
        return self._sampling(x, last_obs, n_sampling_paths)

    def forecast_VaR(self, x, last_obs, epsilon):
        ''' epsilon: threshold of Value-at-Risk within (0, 1) '''
        assert 0 < epsilon < 1
        return self._forecast_VaR(x, last_obs, epsilon)
    
    def fit(self, x, last_obs):
        return self._fit(x, last_obs)
    
    def mult_fit(self, x, last_obs):
        return self._mult_fit(x, last_obs)

    @abstractmethod
    def _nll(self, *args, **kwargs):
        return NotImplementedError

    @abstractmethod
    def _forecast_sampling(self, *args, **kwargs):
        return NotImplementedError
    
    @abstractmethod
    def _sampling(self, x, last_obs, n_sampling_paths):
        return NotImplementedError

    @abstractmethod
    def _forecast_VaR(self, x, last_obs, epsilon):
        return NotImplementedError
    
    @abstractmethod
    def _fit(self, x, last_obs):
        return NotImplementedError