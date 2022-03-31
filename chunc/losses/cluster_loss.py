"""
Cluster loss
"""
import torch

from chunc.losses import GenericLoss

class ClusterLoss(GenericLoss):
    """
    """
    def __init__(self,
        alpha: float=1.0,
        name:   str='cluster_loss',
    ):
        super(ClusterLoss, self).__init__(name)
        self.alpha = alpha

    def __loss(self,
        outputs,
        data,
    ):
        """
        This loss attempts to push valid points towards the origin
        and invalid points away from the origin.
        For valid points, we want |x|^2 = 0, so the loss is simply the distance.
        For invalid points, we want |x|^2 -> inf, so the loss is: 1 - |x|^2/|x|^2_max,
        where |x|^2_max is the largest distance in the batch.
        """
        latent_representation = outputs[1]
        lengths = torch.norm(latent_representation, p=2, dim=1)
        max_length = torch.max(lengths)
        labels = data[1].squeeze(1).to(self.device)
        #loss = torch.cat((lengths[(labels == 1)]/max_length,(1. - lengths[(labels == 0)]/max_length)))
        valid_loss = labels * lengths / max_length
        invalid_loss = (1 - labels) * (1. - lengths/max_length)
        loss = valid_loss + invalid_loss
        # loss = torch.cat((lengths[(labels == 1)],1.0/(lengths[(labels == 0)] + 1e-16)))
        #print(loss)
        # if weights != None:
        #     loss = (loss * weights/weights.sum()).sum()
        # #print(loss)
        return loss.mean()

    def loss(self,
        outputs,
        data,
    ):
        """Computes and returns/saves loss information"""
        loss = self.__loss(outputs, data)
        self.batch_loss = torch.cat((self.batch_loss, torch.tensor([[loss]], device=self.device)), dim=0)
        return self.alpha * loss