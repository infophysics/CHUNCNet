"""
Classes for storing ML timing information.
"""
import torch
import time

class Timer:
    """
    Internal class for recording timing information.
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
        self.timer_values = torch.empty(size=(0,1), dtype=torch.float)
        if self.gpu:
            self.timer_values.cuda()
            self.timer_start  = torch.cuda.Event(enable_timing=True)
            self.timer_end    = torch.cuda.Event(enable_timing=True)
            self.start = self._start_cuda
            self.end   = self._end_cuda
        else:
            self.timer_start = 0
            self.timer_end   = 0
            self.start = self._start_cpu
            self.end   = self._end_cpu
    
    def synchronize(self):
        torch.cuda.synchronize()

    def _start_cuda(self):
        self.timer_start.record()
    
    def _start_cpu(self):
        self.timer_start = time.time()
    
    def _end_cuda(self):
        self.timer_end.record()
        torch.cuda.synchronize()
        self.timer_values = torch.cat(
            (self.timer_values,
            torch.tensor([[self.timer_start.elapsed_time(self.timer_end)]])),
        )
    
    def _end_cpu(self):
        self.timer_end = time.time()

class Timers:
    """
    Collection of timers for ML tasks.
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
        self.timers = {
            'epoch_training':   Timer('epoch_training', type='training', level='epoch',  gpu=self.gpu),
            'epoch_validation': Timer('epoch_validation', type='validation', level='epoch', gpu=self.gpu),
            # individual training information
            'training_data':            Timer('training_data',         **self.train_batch_params),
            'training_zero_grad':       Timer('training_zero_grad',    **self.train_batch_params),
            'training_forward':         Timer('training_forward',      **self.train_batch_params),
            'training_loss':            Timer('training_loss',         **self.train_batch_params),
            'training_loss_backward':   Timer('training_loss_backward',**self.train_batch_params),
            'training_backprop':        Timer('training_backprop',     **self.train_batch_params),
            'training_metrics':         Timer('training_metrics',      **self.train_batch_params),
            'training_progress':        Timer('training_progress',     **self.train_batch_params),
            'training_callbacks':       Timer('training_callbacks',    type='training', level='epoch',  gpu=self.gpu),
            # individual validation information
            'validation_data':      Timer('validation_data',      **self.validation_batch_params),
            'validation_forward':   Timer('validation_forward',   **self.validation_batch_params),
            'validation_loss':      Timer('validation_loss',      **self.validation_batch_params),
            'validation_metrics':   Timer('validation_metrics',   **self.validation_batch_params),
            'validation_progress':  Timer('validation_progress',  **self.validation_batch_params),
            'validation_callbacks': Timer('validation_callbacks', type='validation', level='epoch',  gpu=self.gpu),
        }

    def synchronize(self):
        torch.cuda.synchronize()