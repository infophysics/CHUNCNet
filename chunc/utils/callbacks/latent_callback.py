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
                self.training_latent = self.metrics_list.metrics[self.latent_name].batch_latent
                self.metrics_list.metrics[self.latent_name].reset_batch()
            if self.target_name != None:
                self.training_target = self.metrics_list.metrics[self.target_name].batch_target
                self.metrics_list.metrics[self.target_name].reset_batch()
        else:
            if self.latent_name != None:
                self.validation_latent = self.metrics_list.metrics[self.latent_name].batch_latent
                self.metrics_list.metrics[self.latent_name].reset_batch()
            if self.target_name != None:
                self.validation_target = self.metrics_list.metrics[self.target_name].batch_target
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
                histtype='step',
                stacked=True,
                density=True
            )
            axs.hist(
                self.training_latent[:,self.binary_variable][~valid_mask].cpu().numpy(),
                bins=100,
                label='invalid',
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
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    self.training_latent[:,ii][~valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='invalid', 
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
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    self.validation_latent[:,ii][~valid_mask].cpu().numpy(), 
                    bins=100, 
                    label='invalid', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f'latent_{ii}')
            axs.flat[0].legend()
            plt.suptitle("Validation latent variables by class")
            plt.tight_layout()
            plt.savefig(f"plots/latent/validation_variables_category.png")
            
    def evaluate_testing(self):  
        pass
        # # evaluate metrics from training and validation
        # if self.metrics_list == None:
        #     return
        # epoch_ticks = np.arange(1,self.epochs+1)
        # # training plot
        # fig, axs = plt.subplots(figsize=(10,5))
        # if len(self.training_metrics) != 0:
        #     for ii, metric in enumerate(self.metric_names):
        #         temp_metric = self.training_metrics[:,ii]
        #         final_metric_value = f"(final={temp_metric[-1]:.2e})"
        #         axs.plot(
        #             epoch_ticks,
        #             temp_metric.cpu().numpy(),
        #             c=self.plot_colors[-(ii+1)],
        #             label=rf"{metric}"
        #         )
        #         axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_metric_value}"
        #         )
        #     axs.set_xlabel("epoch")
        #     axs.set_ylabel("metric")
        #     axs.set_yscale('log')
        #     plt.title("metric vs. epoch (training)")
        #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        #     plt.tight_layout()
        #     plt.savefig("plots/epoch_training_metrics.png")
        
        # if len(self.validation_metrics) != 0:
        #     fig, axs = plt.subplots(figsize=(10,5))
        #     for ii, metric in enumerate(self.metric_names):
        #         temp_metric = self.validation_metrics[:,ii]
        #         final_metric_value = f"(final={temp_metric[-1]:.2e})"
        #         axs.plot(
        #             epoch_ticks,
        #             temp_metric.cpu().numpy(),
        #             c=self.plot_colors[-(ii+1)],
        #             label=rf"{metric}"
        #         )
        #         axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_metric_value}"
        #         )
        #     axs.set_xlabel("epoch")
        #     axs.set_ylabel("metric")
        #     axs.set_yscale('log')
        #     plt.title("metric vs. epoch (validation)")
        #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        #     plt.tight_layout()
        #     plt.savefig("plots/epoch_validation_metrics.png")

        # if len(self.training_metrics) != 0 and len(self.validation_metrics) != 0:
        #     fig, axs = plt.subplots(figsize=(10,5))
        #     for ii, metric in enumerate(self.metric_names):
        #         temp_training_metric = self.training_metrics[:,ii]
        #         temp_validation_metric = self.validation_metrics[:,ii]
        #         final_training_metric_value = f"(final={temp_training_metric[-1]:.2e})"
        #         final_validation_metric_value = f"(final={temp_validation_metric[-1]:.2e})"
        #         axs.plot(
        #             epoch_ticks,
        #             temp_training_metric.cpu().numpy(),
        #             c=self.plot_colors[-(ii+1)],
        #             linestyle='-',
        #             label=rf"{metric}"
        #         )
        #         axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_training_metric_value}"
        #         )
        #         axs.plot(
        #             epoch_ticks,
        #             temp_validation_metric.cpu().numpy(),
        #             c=self.plot_colors[-(ii+1)],
        #             linestyle='--',
        #             label=rf"{metric}"
        #         )
        #         axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_validation_metric_value}"
        #         )
        #     if len(self.test_metrics) != 0:
        #         for ii, metric in enumerate(self.metric_names):
        #             temp_metric = self.test_metrics[:,ii]
        #             final_metric_value = f"(final={temp_metric[-1]:.2e})"
        #             axs.plot([],[],
        #                 marker='x',
        #                 linestyle='',
        #                 c=self.plot_colors[-(ii+1)],
        #                 label=rf"(test) {metric}"
        #             )
        #             axs.plot([],[],
        #             marker='',
        #             linestyle='',
        #             label=rf"{final_metric_value}"
        #         )
        #     axs.set_xlabel("epoch")
        #     axs.set_ylabel("metric")
        #     axs.set_yscale('log')
        #     plt.title("metric vs. epoch")
        #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        #     plt.tight_layout()
        #     plt.savefig("plots/epoch_metrics.png")

    def evaluate_inference(self):
        pass