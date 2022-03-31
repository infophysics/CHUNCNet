"""
Wrapper for OutputL2 loss
"""
import torch
import torch.nn as nn

from chunc.losses import GenericLoss

class L2OutputLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='l2_loss',
        reduction:  str='mean'
    ):
        super(L2OutputLoss, self).__init__(name)
        self.alpha = alpha
        self.reduction = reduction
        self.l2_loss = nn.MSELoss(reduction=reduction)

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = self.l2_loss(outputs[0].to(self.device), data[0].to(self.device))
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss