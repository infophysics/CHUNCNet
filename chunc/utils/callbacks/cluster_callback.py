"""
Generic metric callback
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
import os

from chunc.metrics.savers import *
from chunc.utils.callbacks import GenericCallback
from chunc.utils import utils

class ClusterCallback(GenericCallback):
    """
    """
    def __init__(self,
        criterion_list,
        metrics_list,
        latent_variables:   list=[],
    ):  
        super(ClusterCallback, self).__init__()
        self.criterion_list = criterion_list
        self.loss_names = [loss.name for name, loss in self.criterion_list.losses.items()]
        self.metrics_list = metrics_list
        self.latent_name = None
        self.target_name = None
        self.latent_variables = latent_variables
        if metrics_list != None:
            for name, metric in self.metrics_list.metrics.items():
                if isinstance(metric, LatentSaver):
                    self.latent_name = name
                if isinstance(metric, TargetSaver):
                    self.target_name = name

        if not os.path.isdir("plots/latent/"):
            os.makedirs("plots/latent/")

        # containers for training metrics
        if self.latent_name != None:
            self.training_latent = None
            self.validation_latent = None
        if self.target_name != None:
            self.training_target = None
            self.validation_target = None
            # containers for training metric
            self.training_latent = torch.empty(
                size=(0,1), 
                dtype=torch.float, device=self.device
            )
            self.validation_latent = torch.empty(
                size=(0,1), 
                dtype=torch.float, device=self.device
            )
            self.test_latent = torch.empty(
                size=(0,1), 
                dtype=torch.float, device=self.device
            )

    def reset_batch(self):
        self.training_metrics = torch.empty(
            size=(0,1), 
            dtype=torch.float, device=self.device
        )
        self.validation_metrics = torch.empty(
            size=(0,1), 
            dtype=torch.float, device=self.device
        )
        self.test_metrics = torch.empty(
            size=(0,1), 
            dtype=torch.float, device=self.device
        )

    def evaluate_epoch(self,
        train_type='training'
    ):  
        if train_type == 'training':
            if self.latent_name != None:
                self.training_latent = self.metrics_list.metrics[self.latent_name].epoch_latent
                self.metrics_list.metrics[self.latent_name].reset_batch()
            if self.target_name != None:
                self.training_target = self.metrics_list.metrics[self.target_name].epoch_target
                self.metrics_list.metrics[self.target_name].reset_batch()
        else:
            if self.latent_name != None:
                self.validation_latent = self.metrics_list.metrics[self.latent_name].epoch_latent
                self.metrics_list.metrics[self.latent_name].reset_batch()
            if self.target_name != None:
                self.validation_target = self.metrics_list.metrics[self.target_name].epoch_target
                self.metrics_list.metrics[self.target_name].reset_batch()

    def evaluate_training(self):
        # plot the latent distributions  
        if self.latent_name != None and self.target_name != None:
            fig, axs = plt.subplots(figsize=(10,6))

            lengths = torch.norm(self.training_latent[:,self.latent_variables], p=2, dim=1)
            valid_mask = (self.training_target == 1).squeeze(1)

            axs.hist(
                lengths[valid_mask].cpu().numpy(),
                bins=100,
                label='valid',
                color='k',
                histtype='step',
                stacked=True,
                density=True
            )
            axs.hist(
                lengths[~valid_mask].cpu().numpy(),
                bins=100,
                label='invalid',
                color='r',
                histtype='step',
                stacked=True,
                density=True
            )
            axs.set_xlabel("latent distance")
            plt.legend()
            plt.title("Training latent distance")
            plt.tight_layout()
            plt.savefig(f"plots/latent/training_distance.png")
            
            # plot all variable masks
            fig, axs = utils.generate_plot_grid(
                len(self.latent_variables),
                figsize=(10, 6)
            )
            for ii in self.latent_variables:
                lengths = torch.abs(self.training_latent[:,ii])
                axs.flat[ii].hist(
                    lengths[valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='valid', 
                    color='k',
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    lengths[~valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='invalid', 
                    color='r',
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f'latent_{ii}')
            axs.flat[0].legend()
            plt.suptitle("Training latent variables distance by class")
            plt.tight_layout()
            plt.savefig(f"plots/latent/training_variables_distance.png")

            fig, axs = plt.subplots(figsize=(10,6))

            lengths = torch.norm(self.validation_latent[:,self.latent_variables], p=2, dim=1)
            valid_mask = (self.validation_target == 1).squeeze(1)

            axs.hist(
                lengths[valid_mask].cpu().numpy(),
                bins=100,
                label='valid',
                color='k',
                histtype='step',
                stacked=True,
                density=True
            )
            axs.hist(
                lengths[~valid_mask].cpu().numpy(),
                bins=100,
                label='invalid',
                color='r',
                histtype='step',
                stacked=True,
                density=True
            )
            axs.set_xlabel("latent distance")
            plt.legend()
            plt.title("Validation latent distance")
            plt.tight_layout()
            plt.savefig(f"plots/latent/validation_distance.png")

            # plot all variable masks
            fig, axs = utils.generate_plot_grid(
                len(self.latent_variables),
                figsize=(10, 6)
            )
            for ii in self.latent_variables:
                lengths = torch.abs(self.validation_latent[:,ii])
                axs.flat[ii].hist(
                    lengths[valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='valid', 
                    color='k',
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    lengths[~valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='invalid', 
                    color='r',
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f'latent_{ii}')
            axs.flat[0].legend()
            plt.suptitle("Validation latent variables distance by class")
            plt.tight_layout()
            plt.savefig(f"plots/latent/validation_variables_distance.png")


    def evaluate_testing(self):  
        pass

    def evaluate_inference(self):
        pass