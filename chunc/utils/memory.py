"""
Classes for storing ML memory information.
"""
import torch

class MemoryTracker:
    """
    Internal class for recording memory information.
    """
    def __init__(self,
        name:   str,
        level:  str='epoch',
        type:   str='train',
        gpu:    bool=True,
    ):
        self.name = name
        self.level = level
        self.type = type
        self.gpu = gpu
        # initialized tensor
        self.memory_values = torch.empty(size=(0,1), dtype=torch.float)
        if self.gpu:
            self.memory_values.cuda()
            self.memory_start  = 0.0
            self.memory_end    = 0.0
            self.start = self._start_cuda
            self.end   = self._end_cuda
        else:
            self.memory_start = 0.0
            self.memory_end   = 0.0
            self.start = self._start_cpu
            self.end   = self._end_cpu
    
    def synchronize(self):
        torch.cuda.synchronize()

    def _start_cuda(self):
        self.memory_start = torch.cuda.memory_stats()['allocated_bytes.all.allocated']
    
    def _start_cpu(self):
        self.memory_start = 0.0
    
    def _end_cuda(self):
        self.memory_end = torch.cuda.memory_stats()['allocated_bytes.all.allocated']
        torch.cuda.synchronize()
        self.memory_values = torch.cat(
            (self.memory_values,
            torch.tensor([[self.memory_end - self.memory_start]])),
        )
    
    def _end_cpu(self):
        self.memory_end = 0.0

class MemoryTrackers:
    """
    Collection of memory_trackers for ML tasks.
    """
    def __init__(self,
        gpu:    bool=True,
    ):
        self.gpu = gpu
        self.train_batch_params = {
            'type': 'training',
            'level':'batch',
            'gpu':  'self.gpu'
        }
        self.validation_batch_params = {
            'type': 'validation',
            'level':'batch',
            'gpu':  'self.gpu'
        }
        self.memory_trackers = {
            'epoch_training':   MemoryTracker('epoch_training', type='training', level='epoch',  gpu=self.gpu),
            'epoch_validation': MemoryTracker('epoch_validation', type='validation', level='epoch', gpu=self.gpu),
            # individual training information
            'training_data':            MemoryTracker('training_data',         **self.train_batch_params),
            'training_zero_grad':       MemoryTracker('training_zero_grad',    **self.train_batch_params),
            'training_forward':         MemoryTracker('training_forward',      **self.train_batch_params),
            'training_loss':            MemoryTracker('training_loss',         **self.train_batch_params),
            'training_loss_backward':   MemoryTracker('training_loss_backward',**self.train_batch_params),
            'training_backprop':        MemoryTracker('training_backprop',     **self.train_batch_params),
            'training_metrics':         MemoryTracker('training_metrics',      **self.train_batch_params),
            'training_progress':        MemoryTracker('training_progress',     **self.train_batch_params),
            'training_callbacks':       MemoryTracker('training_callbacks',    type='training', level='epoch',  gpu=self.gpu),
            # individual validation information
            'validation_data':      MemoryTracker('validation_data',      **self.validation_batch_params),
            'validation_forward':   MemoryTracker('validation_forward',   **self.validation_batch_params),
            'validation_loss':      MemoryTracker('validation_loss',      **self.validation_batch_params),
            'validation_metrics':   MemoryTracker('validation_metrics',   **self.validation_batch_params),
            'validation_progress':  MemoryTracker('validation_progress',  **self.validation_batch_params),
            'validation_callbacks': MemoryTracker('validation_callbacks', type='validation', level='epoch',  gpu=self.gpu),
        }
    
    def reset_trackers(self):
        self.memory_trackers = {
            'epoch_training':   MemoryTracker('epoch_training', type='training', level='epoch',  gpu=self.gpu),
            'epoch_validation': MemoryTracker('epoch_validation', type='validation', level='epoch', gpu=self.gpu),
            # individual training information
            'training_data':            MemoryTracker('training_data',         **self.train_batch_params),
            'training_zero_grad':       MemoryTracker('training_zero_grad',    **self.train_batch_params),
            'training_forward':         MemoryTracker('training_forward',      **self.train_batch_params),
            'training_loss':            MemoryTracker('training_loss',         **self.train_batch_params),
            'training_loss_backward':   MemoryTracker('training_loss_backward',**self.train_batch_params),
            'training_backprop':        MemoryTracker('training_backprop',     **self.train_batch_params),
            'training_metrics':         MemoryTracker('training_metrics',      **self.train_batch_params),
            'training_progress':        MemoryTracker('training_progress',     **self.train_batch_params),
            'training_callbacks':       MemoryTracker('training_callbacks',    type='training', level='epoch',  gpu=self.gpu),
            # individual validation information
            'validation_data':      MemoryTracker('validation_data',      **self.validation_batch_params),
            'validation_forward':   MemoryTracker('validation_forward',   **self.validation_batch_params),
            'validation_loss':      MemoryTracker('validation_loss',      **self.validation_batch_params),
            'validation_metrics':   MemoryTracker('validation_metrics',   **self.validation_batch_params),
            'validation_progress':  MemoryTracker('validation_progress',  **self.validation_batch_params),
            'validation_callbacks': MemoryTracker('validation_callbacks', type='validation', level='epoch',  gpu=self.gpu),
        }

    def synchronize(self):
        torch.cuda.synchronize()