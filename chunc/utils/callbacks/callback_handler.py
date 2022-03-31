"""
Container for generic callbacks
"""
from chunc.utils.logger import Logger
from chunc.utils.callbacks import GenericCallback
from chunc.utils.callbacks import LossCallback, MetricCallback
from chunc.utils.utils import get_method_arguments

class CallbackHandler:
    """
    """

    def __init__(self,
        name:   str,
        cfg:    dict={},
        callbacks:  list=[]
    ):
        self.name = name
        self.logger = Logger(self.name, file_mode="w")
        if bool(cfg) and len(callbacks) != 0:
            self.logger.error(f"handler received both a config and a list of callbacks! The user should only provide one or the other!")
        else:
            if bool(cfg):
                self.cfg = cfg
                self.process_config()
            else:
                self.callbacks = {callback.name: callback for callback in callbacks}

        # set to whatever the last call of set_device was.
        self.device = 'None'

    def process_config(self):
        # list of available callbacks
        self.available_callbacks = {
            'loss':                     LossCallback,
            'metric':                   MetricCallback,
        }
        # check config
        for item in self.cfg.keys():
            if item not in self.available_callbacks.keys():
                self.logger.error(f"specified callback '{item}' is not an available type! Available types:\n{self.available_callbacks}")
            argdict = get_method_arguments(self.available_callbacks[item])
            for value in self.cfg[item].keys():
                if value not in argdict.keys():
                    self.logger.error(f"specified callback value '{item}:{value}' not a constructor parameter for '{item}'! Constructor parameters:\n{argdict}")
            for value in argdict.keys():
                if argdict[value] == None:
                    if value not in self.cfg[item].keys():
                        self.logger.error(f"required input parameters '{item}:{value}' not specified! Constructor parameters:\n{argdict}")
        self.callbacks = {}
        for item in self.cfg.keys():
            self.callbacks[item] = self.available_callbacks[item](**self.cfg[item])
    
    def set_device(self,
        device
    ):  
        for name, callback in self.callbacks.items():
            callback.set_device(device)
            callback.reset_batch()
        self.device = device

    def add_callback(self,
        callback:   GenericCallback
    ):
        self.callbacks[callback.name] = callback
    
    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:   int,
    ):
        for name, callback in self.callbacks.items():
            callback.set_training_info(
                epochs,
                num_training_batches,
                num_validation_batches,
                num_test_batches
            )

    def evaluate_epoch(self,
        train_type='train',
    ):
        if train_type not in ['training', 'validation', 'test']:
            self.logger.error(f"specified train_type: '{train_type}' not allowed!")
        for name, callback in self.callbacks.items():
            callback.evaluate_epoch(train_type)

    def evaluate_training(self):
        for name, callback in self.callbacks.items():
            callback.evaluate_training()

    def evaluate_testing(self):
        for name, callback in self.callbacks.items():
            callback.evaluate_testing()
    
    def evaluate_inference(self):
        for name, callback in self.callbacks.items():
            callback.evaluate_inference()