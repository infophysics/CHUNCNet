"""
Container for generic losses
"""
from chunc.utils.logger import Logger
from chunc.losses import GenericLoss
from chunc.losses import L1Loss, L2Loss
from chunc.losses import WassersteinLoss
from chunc.losses import ClusterLoss, L2OutputLoss
from chunc.losses import LatentWassersteinLoss
from chunc.losses import LatentWassersteinValidLoss
from chunc.losses import LatentWassersteinInvalidLoss
from chunc.utils.utils import get_method_arguments

class LossHandler:
    """
    """
    def __init__(self,
        name:   str,
        cfg:    dict={},
        losses:  list=[],
    ):
        self.name = name
        self.logger = Logger(self.name, file_mode="w")
        if bool(cfg) and len(losses) != 0:
            self.logger.error(f"handler received both a config and a list of losses! The user should only provide one or the other!")
        else:
            if bool(cfg):
                self.cfg = cfg
                self.process_config()
            else:
                self.losses = {loss.name: loss for loss in losses}

        # set to whatever the last call of set_device was.
        self.device = 'None'
    
    def process_config(self):
        # list of available criterions
        # TODO: Make this automatic
        self.available_criterions = {
            'L1Loss':           L1Loss,
            'L2Loss':           L2Loss,
            'WassersteinLoss':  WassersteinLoss,
            'ClusterLoss':      ClusterLoss,
            'L2OutputLoss':     L2OutputLoss,
            'LatentWassersteinLoss':        LatentWassersteinLoss,
            'LatentWassersteinValidLoss':   LatentWassersteinValidLoss,
            'LatentWassersteinInvalidLoss': LatentWassersteinInvalidLoss,
        }
        # check config
        for item in self.cfg.keys():
            if item not in self.available_criterions.keys():
                self.logger.error(f"specified callback '{item}' is not an available type! Available types:\n{self.available_criterions}")
            argdict = get_method_arguments(self.available_criterions[item])
            for value in self.cfg[item].keys():
                if value not in argdict.keys():
                    self.logger.error(f"specified callback value '{item}:{value}' not a constructor parameter for '{item}'! Constructor parameters:\n{argdict}")
            for value in argdict.keys():
                if argdict[value] == None:
                    if value not in self.cfg[item].keys():
                        self.logger.error(f"required input parameters '{item}:{value}' not specified! Constructor parameters:\n{argdict}")
        self.losses = {}
        for item in self.cfg.keys():
            self.losses[item] = self.available_criterions[item](**self.cfg[item])

    def set_device(self,
        device
    ):  
        for name, loss in self.losses.items():
            loss.set_device(device)
            loss.reset_batch()
        self.device = device

    def reset_batch(self):  
        for name, loss in self.losses.items():
            loss.reset_batch()

    def add_loss(self,
        loss:   GenericLoss
    ):
        self.losses[loss.name] = loss
    
    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:   int,
    ):
        for name, loss in self.losses.items():
            loss.set_training_info(
                epochs,
                num_training_batches,
                num_validation_batches,
                num_test_batches
            )
            loss.reset_batch()

    def loss(self,
        outputs,
        data,
    ):
        losses = [loss.loss(outputs, data) for name, loss in self.losses.items()]
        return sum(losses)
    