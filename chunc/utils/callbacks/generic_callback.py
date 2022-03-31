"""
Functions for evaluating and storing training information.
"""

class GenericCallback:
    """
    """
    def __init__(self):
        self.epochs = None
        self.num_training_batches = None
        self.num_validation_batches = None
        self.num_test_batches = None
        self.plot_colors = ['b','g','r','c','m','y']

        self.device = 'cpu'

    def set_device(self,
        device
    ):  
        self.device = device
    
    def reset_batch(self):
        pass

    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:       int,
    ):
        self.epochs = epochs
        self.num_training_batches = num_training_batches
        self.num_validation_batches = num_validation_batches
        self.num_test_batches = num_test_batches
    
    def evaluate_epoch(self,
        train_type='training'
    ):
        pass

    def evaluate_training(self):
        pass

    def evaluate_testing(self):
        pass

    def evaluate_inference(self):
        pass