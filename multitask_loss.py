import torch
import torch.nn as nn
from overrides import overrides

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
        self.coefs = torch.tensor(coefs) 

        self.losses = losses
        self.num_loss = len(losses)
        self.normalizing_coefs = None

    @overrides
    def forward(self, *inputs):
        '''
        inputs: tuple of the inputs for each losses, in the same order
        they were given to the constructor.
        '''
        assert len(inputs) == self.num_loss, 'There should be as many inputs as losses.'


        losses_val = {name(l): l(*i) for l, i in zip(self.losses, inputs)}
        losses_val = {k: v if (len(v.size()) > 0) else v.unsqueeze(0) for k, v in losses_val.items()}
        self.current_losses = losses_val

        losses_val = torch.cat(list(losses_val.values()))

        if self.normalizing_coefs is None:
            with torch.no_grad():
                self.normalizing_coefs = self.num_loss * 100. / losses_val
                # the first combined loss will always have a value of 100

        losses_val = losses_val.mul(self.normalizing_coefs)
        losses_val = losses_val.mul(self.coefs)

        return losses_val.sum() / self.num_loss

    def get_sublosses(self):
        return torch.cat(list(self.current_losses.values())).mul(self.normalizing_coefs)

    def print_sublosses(self):
        return {k: f'{float(v):.3f}' for k, v in self.current_losses.items()}
