import torch
import torch.nn as nn
from overrides import overrides


class MtLoss(nn.Module):
    """
    Parameters:
        losses: a list of losses (nn.Module) that will be put together.
        mixing_coef:
            Ignored if there is only one loss.
            If a float (in [0;1]), will be the factor multiplying the firt loss.
            The other losses will be multiplied so that the sum of factors is 1.
            If an array of float is given, each loss will get multiplied by the factor
            corresponding to the normalized array.
    """

    @overrides
    def __init__(self, *losses, mixing_coef=.5):
        super(MtLoss, self).__init__()
        print(len(losses))
        print(type(losses))

        if len(losses) > 1:
            if isinstance(mixing_coef, (float, int)):
                assert (mixing_coef <= 1) and (mixing_coef >= 0), 'mixing_coef should be between 0 and 1.'
                rest_len = len(losses) - 1
                rest = (1 - mixing_coef) / rest_len
                coefs = [mixing_coef]
                coefs.extend([rest]*rest_len)
            elif isinstance(mixing_coef, list) and (len(mixing_coef) == len(losses)):
                coefs = [c / max(mixing_coef) for c in mixing_coef]
            else:
                raise ValueError('mixing_coef should be either a float or a list of floats of length equal to the number of losses number of losses')
        else:
            coefs = [1]

        self.coefs = coefs        
        self.losses = losses

    @overrides
    def forward(self, *inputs):
        '''
        inputs: tuple of the inputs for each losses, in the same order
        they were given to the constructor.
        '''
        assert len(inputs) == len(self.losses), 'There should be as many inputs as losses.'

        losses_val = [l(*i)*c for l, i, c in zip(self.losses, inputs, self.coefs)]
        losses_val = [e if (len(e.size()) > 0) else e.unsqueeze(0) for e in losses_val]
        return torch.cat(losses_val, dim=0).sum()

    