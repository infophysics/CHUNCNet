"""
Optimizers for chunc.
"""
import torch.optim as optim

from chunc.utils.logger import Logger

class Optimizer:
    """
    A standard optimizer for pytorch models.
    """
    def __init__(self,
        model,
        optimizer:      str='Adam',
        learning_rate:  float=0.001,
        momentum:       float=0.9
    ):
        self.name = model.name + "_optimizer"
        self.logger = Logger(self.name, file_mode='w')
        # set learning rate and momentum
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.logger.info(f"learning rate set to {self.learning_rate}")
        self.logger.info(f"momentum value set to {self.momentum}")

        # set the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
            )
            self.logger.info(f"using the Adam optimizer")
        
    def zero_grad(self):
        return self.optimizer.zero_grad()
    
    def step(self):
        return self.optimizer.step()