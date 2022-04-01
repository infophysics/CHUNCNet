"""
Generic saver metric class for tpc_ml.
"""
import torch
import torch.nn as nn

from chunc.metrics import GenericMetric

class LatentSaver(GenericMetric):
    
    def __init__(self,
        name:   str='latent_saver',
        output_shape:   tuple=(),
        latent_shape:   tuple=(),
        target_shape:   tuple=(),
        input_shape:    tuple=(),
    ):
        """
        Latent Saver
        """
        super(LatentSaver, self).__init__(
            name,
            output_shape,
            latent_shape,
            target_shape,
            input_shape
        )
         # create empty tensors for epoch 
        self.batch_latent = torch.empty(
            size=(0,*self.latent_shape), 
            dtype=torch.float, device=self.device
        )
        
    def reset_batch(self):
        self.batch_latent = torch.empty(
            size=(0,*self.latent_shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        self.batch_latent = torch.cat((self.batch_latent, outputs[1]),dim=0)

    def compute(self):
        pass

class OutputSaver(GenericMetric):
    
    def __init__(self,
        name:   str='output_saver',
        output_shape:   tuple=(),
        latent_shape:   tuple=(),
        target_shape:   tuple=(),
        input_shape:    tuple=(),
    ):
        """
        output Saver
        """
        super(OutputSaver, self).__init__(
            name,
            output_shape,
            latent_shape,
            target_shape,
            input_shape
        )
         # create empty tensors for epoch 
        self.batch_output = torch.empty(
            size=(0,*self.output_shape), 
            dtype=torch.float, device=self.device
        )
        
    def reset_batch(self):
        self.batch_output = torch.empty(
            size=(0,*self.output_shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        self.batch_output = torch.cat((self.batch_output, outputs[0]),dim=0)

    def compute(self):
        pass

class TargetSaver(GenericMetric):
    
    def __init__(self,
        name:   str='target_saver',
        output_shape:   tuple=(),
        latent_shape:   tuple=(),
        target_shape:   tuple=(),
        input_shape:    tuple=(),
    ):
        """
        Target Saver
        """
        super(TargetSaver, self).__init__(
            name,
            output_shape,
            latent_shape,
            target_shape,
            input_shape
        )
         # create empty tensors for epoch 
        self.batch_target = torch.empty(
            size=(0,*self.target_shape), 
            dtype=torch.float, device=self.device
        )
        
    def reset_batch(self):
        self.batch_target = torch.empty(
            size=(0,*self.target_shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        self.batch_target = torch.cat((self.batch_target, data[1].to(self.device)),dim=0)

    def compute(self):
        pass

class InputSaver(GenericMetric):
    
    def __init__(self,
        name:   str='input_saver',
        output_shape:   tuple=(),
        latent_shape:   tuple=(),
        target_shape:   tuple=(),
        input_shape:    tuple=(),
    ):
        """
        Input Saver
        """
        super(InputSaver, self).__init__(
            name,
            output_shape,
            latent_shape,
            target_shape,
            input_shape
        )
         # create empty tensors for epoch 
        self.batch_input = torch.empty(
            size=(0,*self.input_shape), 
            dtype=torch.float, device=self.device
        )
        
    def reset_batch(self):
        self.batch_input = torch.empty(
            size=(0,*self.input_shape), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        self.batch_input = torch.cat((self.batch_input, data[0].to(self.device)),dim=0)

    def compute(self):
        pass
        