"""
Generic metric class for tpc_ml.
"""
import torch

class GenericMetric:
    """
    """
    def __init__(self,
        name:   str='generic',
        output_shape:   tuple=(),
        latent_shape:   tuple=(),
        target_shape:   tuple=(),
        input_shape:    tuple=(),
        when_compute:   str='all',
    ):
        self.name = name
        self.output_shape = output_shape
        self.latent_shape = latent_shape
        self.target_shape = target_shape
        self.input_shape = input_shape
        self.when_compute = when_compute
        # set device to none for now
        self.device = 'cpu'

        # create empty tensors for evaluation
        self.batch_metric = torch.empty(
            size=(0,1), 
            dtype=torch.float, device=self.device
        )

    def reset_batch(self):
        self.batch_metric = torch.empty(
            size=(0,1), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        pass

    def set_device(self,
        device
    ):  
        self.device = device
        self.reset_batch()

    def compute(self):
        pass