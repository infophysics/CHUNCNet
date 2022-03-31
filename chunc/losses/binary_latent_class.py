"""
Wasserstein loss
"""
import numpy as np
import torch
import torch.nn as nn

from chunc.losses import GenericLoss

class BinaryLatentClass(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='binary_latent_class',
        reduction:  str='mean',
        binary_dim: int=6,
    ):
        super(BinaryLatentClass, self).__init__(name)
        self.alpha = alpha
        self.l2_loss = nn.MSELoss(reduction=reduction)
        self.binary_dim = binary_dim

    def set_device(self,
        device
    ):  
        self.device = device
        self.reset_batch()

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = self.l2_loss(outputs[1][self.binary_dim], data[1].to(self.device))
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss