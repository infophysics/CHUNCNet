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

class LatentCallback(GenericCallback):
    """
    """
    def __init__(self,
        criterion_list,
        metrics_list,
        latent_variables:   list=[],
        binary_variable:    int=-1,
        binary_bins:        int=0,
    ):  
        super(LatentCallback, self).__init__()
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
        self.binary_variable = binary_variable
        self.binary_bins = binary_bins
        
        if not os.path.isdir("plots/latent/"):
            os.makedirs("plots/latent/")
        if self.binary_bins > 0:
            if not os.path.isdir("plots/latent/bins/"):
                os.makedirs("plots/latent/bins/")

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
        if self.latent_name != None:
            # plot all variable inputs and outputs
            fig, axs = utils.generate_plot_grid(
                len(self.latent_variables),
                figsize=(10, 6)
            )
            for ii in self.latent_variables:
                axs.flat[ii].hist(
                    self.training_latent[:,ii].cpu().numpy(), 
                    bins=100, 
                    label='training', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    self.validation_latent[:,ii].cpu().numpy(), 
                    bins=100, 
                    label='validation', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f'latent_{ii}')
            axs.flat[0].legend()
            plt.suptitle("Training/Validation Latent variables")
            plt.tight_layout()
            plt.savefig(f"plots/latent/training_variables.png")
            
        if self.target_name != None and self.binary_variable != -1:
            fig, axs = plt.subplots(figsize=(10,6))
            valid_mask = (self.training_target == 1).squeeze(1)
            axs.hist(
                self.training_latent[:,self.binary_variable][valid_mask].cpu().numpy(),
                bins=100,
                label='valid',
                color='k',
                histtype='step',
                stacked=True,
                density=True
            )
            axs.hist(
                self.training_latent[:,self.binary_variable][~valid_mask].cpu().numpy(),
                bins=100,
                label='invalid',
                color='r',
                histtype='step',
                stacked=True,
                density=True
            )
            axs.set_xlabel("latent category")
            plt.legend()
            plt.title("Latent binary category")
            plt.tight_layout()
            plt.savefig(f"plots/latent/training_category.png")
            
            # plot all variable masks
            fig, axs = utils.generate_plot_grid(
                len(self.latent_variables),
                figsize=(10, 6)
            )
            for ii in self.latent_variables:
                axs.flat[ii].hist(
                    self.training_latent[:,ii][valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='valid', 
                    color='k',
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    self.training_latent[:,ii][~valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='invalid', 
                    color='r',
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f'latent_{ii}')
            axs.flat[0].legend()
            plt.suptitle("Training latent variables by class")
            plt.tight_layout()
            plt.savefig(f"plots/latent/training_variables_category.png")
            # plot all variable masks
            valid_mask = (self.validation_target == 1).squeeze(1)
            fig, axs = utils.generate_plot_grid(
                len(self.latent_variables),
                figsize=(10, 6)
            )
            for ii in self.latent_variables:
                axs.flat[ii].hist(
                    self.validation_latent[:,ii][valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='valid', 
                    color='k',
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    self.validation_latent[:,ii][~valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='invalid', 
                    color='r',
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f'latent_{ii}')
            axs.flat[0].legend()
            plt.suptitle("Validation latent variables by class")
            plt.tight_layout()
            plt.savefig(f"plots/latent/validation_variables_category.png")
        
            if self.binary_bins > 0:
                valid_mask = (self.training_target == 1).squeeze(1)
                # bin the binary variable
                hist, bin_edges = np.histogram(
                    self.training_latent[:,self.binary_variable].cpu().numpy(),
                    bins=self.binary_bins
                )
                valid_indices = np.digitize(
                    self.training_latent[:,self.binary_variable][valid_mask].cpu().numpy(),
                    bins=bin_edges
                )
                invalid_indices = np.digitize(
                    self.training_latent[:,self.binary_variable][~valid_mask].cpu().numpy(),
                    bins=bin_edges
                )
                for ii in self.latent_variables:
                    fig, axs = utils.generate_plot_grid(
                        self.binary_bins+1,
                        figsize=(10, 6)
                    )
                    for jj in range(self.binary_bins):
                        if sum((valid_indices == jj+1)) != 0:
                            # bin the binary variable
                            axs.flat[jj].hist(
                                self.training_latent[:,ii][valid_mask][(valid_indices == jj+1)].cpu().numpy(), 
                                bins=100, 
                                label='valid', 
                                color='k',
                                histtype='step', 
                            )
                        if sum((invalid_indices == jj+1)) != 0:
                            axs.flat[jj].hist(
                                self.training_latent[:,ii][~valid_mask][(invalid_indices == jj+1)].cpu().numpy(), 
                                bins=100, 
                                label='invalid', 
                                color='r',
                                histtype='step', 
                            )
                        axs.flat[jj].set_xlabel(f'bin {jj}: [{bin_edges[jj]:.2f},{bin_edges[jj+1]:.2f}]')
                    axs.flat[0].legend()
                    axs.flat[self.binary_bins].hist(
                        self.training_latent[:,self.binary_variable][valid_mask].cpu().numpy(),
                        bins=bin_edges,
                        label='valid',
                        color='k',
                        histtype='step',
                        stacked=True,
                        density=True
                    )
                    axs.flat[self.binary_bins].hist(
                        self.training_latent[:,self.binary_variable][~valid_mask].cpu().numpy(),
                        bins=bin_edges,
                        label='invalid',
                        color='r',
                        histtype='step',
                        stacked=True,
                        density=True
                    )
                    axs.flat[self.binary_bins].set_xlabel("latent category")
                    plt.suptitle(f"Training latent variable {ii} by binary bin")
                    plt.tight_layout()
                    plt.savefig(f"plots/latent/bins/latent_{ii}.png")

    def evaluate_testing(self):  
        pass

    def evaluate_inference(self):
        pass