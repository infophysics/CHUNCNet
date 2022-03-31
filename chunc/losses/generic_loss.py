"""
Generic losses for tpc_ml.
"""
import torch

class GenericLoss:
    """
    """
    def __init__(self,
        name:   str='generic',
    ):
        self.name = name
        self.alpha = 0.0
        # set device to cpu for now
        self.device = 'cpu'

        # create empty tensors for batching
        self.batch_loss = torch.empty(size=(0,1), dtype=torch.float, device=self.device)

    def set_device(self,
        device
    ):  
        self.device = device
        self.reset_batch()

    def reset_batch(self):
        self.batch_loss = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
    
    def loss(self,
        outputs,
        data,
    ):
        pass