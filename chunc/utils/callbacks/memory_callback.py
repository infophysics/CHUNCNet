"""
Timing callback
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from chunc.utils.memory import MemoryTrackers
from chunc.utils.callbacks import GenericCallback

class MemoryTrackerCallback(GenericCallback):
    """
    """
    def __init__(self,
        output_dir: str,
        memory_trackers: MemoryTrackers
    ):
        self.name = "memory"
        super(MemoryTrackerCallback, self).__init__()
        self.output_dir = output_dir
        self.memory_trackers = memory_trackers
    
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
        for item in self.memory_trackers.memory_trackers.keys():
            if len(self.memory_trackers.memory_trackers[item].memory_values) == 0:
                continue
            if self.memory_trackers.memory_trackers[item].type == train_type:
                if self.memory_trackers.memory_trackers[item].level == 'epoch':
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.squeeze()
                else:
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                averages[item] = temp_times.mean()
                stds[item] = temp_times.std()

        fig, axs = plt.subplots(figsize=(10,6))
        for item in self.memory_trackers.memory_trackers.keys():
            if len(self.memory_trackers.memory_trackers[item].memory_values) == 0:
                continue
            if self.memory_trackers.memory_trackers[item].type == train_type:
                if self.memory_trackers.memory_trackers[item].level == 'epoch':
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.squeeze()
                    linestyle='-'
                else:
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.reshape(
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
                    label=f"total: {temp_times.sum():.2e}bytes"
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
            label=f"total: {batch_overhead.sum():.2e}bytes"
        )

        axs.set_xlabel("epoch")
        axs.set_ylabel(r"$\Delta t$ (bytes)")
        axs.set_yscale('log')

        if train_type == 'training':
            plt.title(r"$\Delta m$ (bytes) vs. epoch (training)")
        else:
            plt.title(r"$\Delta m$ (bytes) vs. epoch (validation)")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        if train_type == 'training':
            plt.savefig(f"{self.output_dir}/batch_training_memory.png")
        else:
            plt.savefig(f"{self.output_dir}/batch_validation_memory.png")

        fig, axs = plt.subplots(figsize=(10,6))
        box_values = torch.empty(size=(0,self.epochs))
        labels = []
        axs.plot([], [],
            marker='x', linestyle='',
            label=f'epochs: {self.epochs}'
        )
        for item in self.memory_trackers.memory_trackers.keys():
            if len(self.memory_trackers.memory_trackers[item].memory_values) == 0:
                continue
            if self.memory_trackers.memory_trackers[item].type == train_type:
                if self.memory_trackers.memory_trackers[item].level == 'epoch':
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.squeeze()
                    linestyle='-'
                else:
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.reshape(
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
        axs.set_ylabel(r"$\langle\Delta m\rangle$ (bytes)")
        axs.set_xticklabels(labels, rotation=45, ha='right')
        axs.set_yscale('log')

        if train_type == 'training':
            plt.title(r"$\langle\Delta m\rangle$ (bytes) vs. task (training)")
        else:
            plt.title(r"$\langle\Delta m\rangle$ (bytes) vs. task (validation)")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        if train_type == 'training':
            plt.savefig(f"{self.output_dir}/batch_training_memory_avg.png")
        else:
            plt.savefig(f"{self.output_dir}/batch_validation_memory_avg.png")

    def evaluate_testing(self):
        pass
    
    def evaluate_inference(self):
        pass