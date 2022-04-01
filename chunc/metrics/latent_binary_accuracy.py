"""
Binary accuracy metric class for tpc_ml.
"""
import torch
import torch.nn as nn

from chunc.metrics import GenericMetric

class LatentBinaryAccuracy(GenericMetric):
    
    def __init__(self,
        name:   str='binary_accuracy',
        output_shape:   tuple=(),
        latent_shape:   tuple=(),
        target_shape:   tuple=(),
        input_shape:    tuple=(),
        cutoff:         float=0.5,
        binary_variable:    int=5,
    ):
        """
        Binary accuracy metric which essentially computes
        the number of correct guesses defined by a single
        cut along the output dimension.
        """
        super(LatentBinaryAccuracy, self).__init__(
            name,
            output_shape,
            latent_shape,
            target_shape,
            input_shape,
        )
        self.cutoff = cutoff
        self.binary_variable = binary_variable

    def update(self,
        outputs,
        data,
    ):
        # set predictions using cutoff
        predictions = (outputs[1][:, self.binary_variable] > self.cutoff).unsqueeze(1)
        accuracy = (predictions == data[1].to(self.device)).float().mean()
        self.batch_metric = torch.cat(
            (self.batch_metric, torch.tensor([[accuracy]], device=self.device)), 
            dim=0
        )

    def compute(self):
        return self.batch_metric.mean()