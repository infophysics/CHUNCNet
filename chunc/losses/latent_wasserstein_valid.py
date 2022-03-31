"""
Wasserstein loss
"""
import numpy as np
import torch

from chunc.losses import GenericLoss

class LatentWassersteinValidLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='latent_wasserstein_loss',
        distribution_type:  str='input',
        distribution:       list=[],
        encoded_type:       str='latent',
        num_projections:    int=100,
    ):
        super(LatentWassersteinValidLoss, self).__init__(name)
        self.alpha = alpha
        self.distribution_type = distribution_type
        self.distribution = distribution.float()
        self.encoded_type = encoded_type
        self.num_projections = num_projections

    def set_device(self,
        device
    ):  
        self.device = device
        if self.distribution != []:
            self.distribution.to(self.device)
        self.reset_batch()
    
    def __loss(self,
        encoded_samples,
        data,
    ):
        """
        We project our distribution onto a sphere and compute the Wasserstein
        distance between the distribution (encoded_samples) and our expected 
        distribution (distribution_samples).
        """
        encoded_samples = encoded_samples[(data[1].squeeze(1) == 1)]
        distribution_samples = self.distribution[
            torch.randint(high = self.distribution.size(0), size =(encoded_samples.size(0),))
        ].to(self.device)
        # first, generate a random sample on a sphere
        embedding_dimension = distribution_samples.size(1)
        normal_samples = np.random.normal(
            size=(self.num_projections, embedding_dimension)
        )
        normal_samples /= np.sqrt((normal_samples**2).sum())
        projections = torch.tensor(normal_samples).transpose(0, 1).to(self.device)

        # now project the embedded samples onto the sphere
        encoded_projections = encoded_samples.matmul(projections.float()).transpose(0, 1).to(self.device)
        distribution_projections = distribution_samples.float().matmul(projections.float()).transpose(0, 1).to(self.device)

        # calculate the distance between the distributions
        wasserstein_distance = (
            torch.sort(encoded_projections, dim=1)[0] -
            torch.sort(distribution_projections, dim=1)[0]
        )
        wasserstein_mean = (torch.pow(wasserstein_distance, 2))
        # if weights != None:
        #     wasserstein_mean = torch.sum(wasserstein_mean, dim=0)
        #     wasserstein_mean = (wasserstein_mean * weights/weights.sum()).sum()

        return wasserstein_mean.mean()

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = self.__loss(outputs[1], data)
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss