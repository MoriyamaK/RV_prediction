from abc import ABCMeta, abstractmethod
import inspect

# class Flow(metaclass=ABCMeta):

#     def forward(self, x, reverse=False, return_auxiliary_losses=False):
#         '''
#         Returns
#         -------
#         If reverse is False:
#             y: flow(x).
#             jacobian: absolute value of the jacobian generated from flow(x)

#         If reverse is True:
#             x: invflow(x).
#             jacobian: absolute value of the jacobian generated from invflow(x).
#         (Invflow is the inverse of flow)
#         '''
#         spec = inspect.getfullargspec(self._forward)
#         argnames = spec[0]

#         if 'return_auxiliary_losses' not in argnames:
#             y, logdet = self._forward(x, reverse=reverse)
#             aux_losses = {}

#         else:
#             if return_auxiliary_losses:
#                 y, logdet, aux_losses = self._forward(x, reverse=reverse, return_auxiliary_losses=return_auxiliary_losses)
#             else:
#                 y, logdet = self._forward(x, reverse=reverse, return_auxiliary_losses=return_auxiliary_losses)
#                 aux_losses = {}
        
#         return y, logdet, aux_losses

#     @abstractmethod
#     def _forward(self, *args, **kwargs):
#         return NotImplementedError
    
#     def __call__(self, *args, **kwargs):
#         self.forward(*args, **kwargs)

class Flow(metaclass=ABCMeta):
    
    def forward(self, x, phi, reverse=False, return_auxiliary_losses=False):
        spec = inspect.getfullargspec(self._forward)
        argnames = spec[0]

        if 'return_auxiliary_losses' not in argnames:
            y, logdet = self._forward(x, reverse=reverse)
            aux_losses = {}

        else:
            if return_auxiliary_losses:
                y, logdet, aux_losses = self._forward(x, reverse=reverse)
            else:
                y, logdet = self._forward(x, reverse=reverse)
                aux_losses = {}
        
        return y, logdet, aux_losses

class ExtendedFlow(metaclass=ABCMeta):

    def forward(self, x, phi, reverse=False, return_auxiliary_losses=False):
        '''
        Returns
        -------
        If reverse is False:
            y: flow(x).
            jacobian: absolute value of the jacobian generated from flow(x)

        If reverse is True:
            x: invflow(x).
            jacobian: absolute value of the jacobian generated from invflow(x).
        (Invflow is the inverse of flow)
        '''
        spec = inspect.getfullargspec(self._forward)
        argnames = spec[0]

        if 'return_auxiliary_losses' not in argnames:
            y, logdet = self._forward(x, phi, reverse=reverse)
            aux_losses = {}

        else:
            if return_auxiliary_losses:
                y, logdet, aux_losses = self._forward(x, phi, reverse=reverse, return_auxiliary_losses=return_auxiliary_losses)
            else:
                y, logdet = self._forward(x, phi, reverse=reverse, return_auxiliary_losses=return_auxiliary_losses)
                aux_losses = {}
        
        return y, logdet, aux_losses

    @abstractmethod
    def _forward(self, *args, **kwargs):
        return NotImplementedError
    
    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
