"""
Timing callback
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from chunc.utils.timing import Timers
from chunc.utils.callbacks import GenericCallback

class TimingCallback(GenericCallback):
    """
    """
    def __init__(self,
        output_dir: str,
        timers: Timers
    ):
        self.name = "timing"
        super(TimingCallback, self).__init__()
        self.output_dir = output_dir
        self.timers = timers
    
    def evaluate_epoch(self,
        train_type='train'
    ):
        pass

    def evaluate_training(self):
        if self.epochs != None:
            if self.num_training_batches != None:
                self.__evaluate_training('training')
            if self.num_validation_batches != None:
                self.__evaluate_training('validation')
    
    def __evaluate_training(self, 
        train_type
    ):
        epoch_ticks = np.arange(1,self.epochs+1)
        if train_type == 'training':
            num_batches = self.num_training_batches
        else:
            num_batches = self.num_validation_batches

        batch_overhead = torch.tensor([0.0 for ii in range(self.epochs)])

        averages = {}
        stds = {}
        for item in self.timers.timers.keys():
            if len(self.timers.timers[item].timer_values) == 0:
                continue
            if self.timers.timers[item].type == train_type:
                if self.timers.timers[item].level == 'epoch':
                    temp_times = self.timers.timers[item].timer_values.squeeze()
                else:
                    temp_times = self.timers.timers[item].timer_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                averages[item] = temp_times.mean()
                stds[item] = temp_times.std()

        fig, axs = plt.subplots(figsize=(10,6))
        for item in self.timers.timers.keys():
            if len(self.timers.timers[item].timer_values) == 0:
                continue
            if self.timers.timers[item].type == train_type:
                if self.timers.timers[item].level == 'epoch':
                    temp_times = self.timers.timers[item].timer_values.squeeze()
                    linestyle='-'
                else:
                    temp_times = self.timers.timers[item].timer_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                    linestyle='--'
                axs.plot(
                    epoch_ticks, 
                    temp_times, 
                    linestyle=linestyle,  
                    label=f'{item.replace(f"{train_type}_","")}'
                )    
                axs.plot([], [],
                    marker='', linestyle='',
                    label=f"total: {temp_times.sum():.2f}ms"
                )
                if 'epoch' in item:
                    batch_overhead += temp_times
                elif 'callbacks' in item:
                    pass
                else:
                    batch_overhead -= temp_times
        axs.plot(epoch_ticks, batch_overhead, linestyle='-',  label='overhead')
        axs.plot([], [],
            marker='', linestyle='',
            label=f"total: {batch_overhead.sum():.2f}ms"
        )

        axs.set_xlabel("epoch")
        axs.set_ylabel(r"$\Delta t$ (ms)")
        axs.set_yscale('log')

        if train_type == 'training':
            plt.title(r"$\Delta t$ (ms) vs. epoch (training)")
        else:
            plt.title(r"$\Delta t$ (ms) vs. epoch (validation)")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        if train_type == 'training':
            plt.savefig(f"{self.output_dir}/batch_training_timing.png")
        else:
            plt.savefig(f"{self.output_dir}/batch_validation_timing.png")
        
        fig, axs = plt.subplots(figsize=(10,6))
        box_values = torch.empty(size=(0,self.epochs))
        labels = []
        axs.plot([], [],
            marker='x', linestyle='',
            label=f'epochs: {self.epochs}'
        )
        for item in self.timers.timers.keys():
            if len(self.timers.timers[item].timer_values) == 0:
                continue
            if self.timers.timers[item].type == train_type:
                if self.timers.timers[item].level == 'epoch':
                    temp_times = self.timers.timers[item].timer_values.squeeze()
                    linestyle='-'
                else:
                    temp_times = self.timers.timers[item].timer_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                    linestyle='--'
                box_values = torch.cat((box_values, temp_times.unsqueeze(0)), dim=0)
                axs.plot([], [],
                    marker='', linestyle=linestyle,
                    label=f'{item.replace(f"{train_type}_","")}\n({averages[item]:.2f} +/- {stds[item]:.2f})'
                )
                labels.append(f'{item.replace(f"{train_type}_","")}')
        axs.boxplot(
            box_values,
            vert=True,
            patch_artist=True,
            labels=labels
        )    
        #axs.set_xlabel("epoch")
        axs.set_ylabel(r"$\langle\Delta t\rangle$ (ms)")
        axs.set_xticklabels(labels, rotation=45, ha='right')
        axs.set_yscale('log')

        if train_type == 'training':
            plt.title(r"$\langle\Delta t\rangle$ (ms) vs. task (training)")
        else:
            plt.title(r"$\langle\Delta t\rangle$ (ms) vs. task (validation)")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        if train_type == 'training':
            plt.savefig(f"{self.output_dir}/batch_training_timing_avg.png")
        else:
            plt.savefig(f"{self.output_dir}/batch_validation_timing_avg.png")

    def evaluate_testing(self):
        pass
    
    def evaluate_inference(self):
        pass