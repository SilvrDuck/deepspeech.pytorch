import torch
import torch.nn as nn
from overrides import overrides
from utils import AverageMeter

def name(loss):
    return type(loss).__name__

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
        self.mix_coefs = torch.tensor(coefs)

        self.losses = losses
        self.names = [type(l).__name__ for l in self.losses]
        self.num_loss = len(losses)
        self.normalizing_coefs = [AverageMeter() for __ in range(self.num_loss)]
        self.update_coefs = True

    @overrides
    def forward(self, *inputs):
        '''
        inputs: tuple of the inputs for each losses, in the same order
        they were given to the constructor.
        '''
        assert len(inputs) == self.num_loss, 'There should be as many inputs as losses.'

        vals = [l(*i) for l, i in zip(self.losses, inputs)]
        vals = [e if (len(e.size()) > 0) else e.unsqueeze(0) for e in vals] # because all losses doesn’t give the same dimension
        vals = torch.cat(vals)
        
        if self.update_coefs:
            self.update_coef_tensor(vals)

        vals = vals.mul(self.get_coef_tensor())
        self.current_losses = vals


        vals = vals.mul(self.mix_coefs)
        return vals.sum()

    def get_coef_tensor(self):
        return torch.tensor([100. / x.avg for x in self.normalizing_coefs])

    def update_coef_tensor(self, tensor_):
        assert len(tensor_) == len(self.normalizing_coefs)
        for e, x in zip(self.normalizing_coefs, tensor_):
            e.update(x)

    def get_sublosses(self):
        return self.current_losses

    def print_sublosses(self):
        return {n: f'{float(v):.3f}' for n, v in zip(self.names, self.current_losses)}

    def toggle_update_coefs(self, new_value):
        self.update_coefs = new_value