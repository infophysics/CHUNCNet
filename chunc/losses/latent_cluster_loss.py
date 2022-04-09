"""
Cluster loss
"""
import numpy as np
from sklearn import cluster
import torch

from chunc.losses import GenericLoss

class LatentClusterLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='latent_cluster_loss',
        cluster_type:       str='normalized',
        latent_variables:   list=[],
    ):
        super(LatentClusterLoss, self).__init__(name)
        self.alpha = alpha
        self.latent_variables = latent_variables

        if cluster_type == 'normalized':
            self.__loss = self.__loss_normalized
        else:
            self.__loss = self.__loss_inverse

    def set_device(self,
        device
    ):  
        self.device = device
        self.reset_batch()
    
    def __loss_normalized(self,
        latent_output,
        labels,
        weights=None,
    ):
        """
        This loss attempts to push valid points towards the origin
        and invalid points away from the origin.
        For valid points, we want |x|^2 = 0, so the loss is simply the distance.
        For invalid points, we want |x|^2 -> inf, so the loss is: 1 - |x|^2/|x|^2_max,
        where |x|^2_max is the largest distance in the batch.
        """
        lengths = torch.norm(latent_output, p=2, dim=1)
        max_length = torch.max(lengths)

        # valid and invalid loss terms
        valid_loss = labels * lengths/max_length
        invalid_loss = (1 - labels) * (1. - lengths/max_length)
        
        loss = valid_loss + invalid_loss
        
        if weights != None:
            loss = (loss * weights/weights.sum()).sum()

        return loss.mean()
    
    def __loss_inverse(self,
        latent_output,
        labels,
        weights=None,
    ):
        """
        This loss attempts to push valid points towards the origin
        and invalid points away from the origin.
        For valid points, we want |x|^2 = 0, so the loss is simply the distance.
        For invalid points, we want |x|^2 -> inf, so the loss is: 1/|x|^2.
        """
        lengths = torch.norm(latent_output, p=2, dim=1)

        # valid and invalid loss terms
        valid_loss = labels * lengths
        invalid_loss = (1 - labels) * (1./(lengths + 1e-16))

        loss = valid_loss + invalid_loss
        
        if weights != None:
            loss = (loss * weights/weights.sum()).sum()

        return loss.mean()

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = self.__loss(outputs[1][:, self.latent_variables], data[1].to(self.device))
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss        