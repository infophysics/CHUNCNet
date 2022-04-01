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

class OutputCallback(GenericCallback):
    """
    """
    def __init__(self,
        criterion_list,
        metrics_list,
        input_variables:   list=[],
    ):  
        super(OutputCallback, self).__init__()
        self.criterion_list = criterion_list
        self.loss_names = [loss.name for name, loss in self.criterion_list.losses.items()]
        self.metrics_list = metrics_list
        self.output_name = None
        self.input_name = None
        self.input_variables = input_variables
        if metrics_list != None:
            for name, metric in self.metrics_list.metrics.items():
                if isinstance(metric, OutputSaver):
                    self.output_name = name
                if isinstance(metric, InputSaver):
                    self.input_name = name
        
        if not os.path.isdir("plots/output/"):
            os.makedirs("plots/output/")

        # containers for training metrics
        if self.output_name != None:
            self.training_output = None
            self.validation_output = None
        if self.input_name != None:
            self.training_input = None
            self.validation_input = None
            # # containers for training metric
            # self.training_output = torch.empty(
            #     size=(0,1), 
            #     dtype=torch.float, device=self.device
            # )
            # self.validation_output = torch.empty(
            #     size=(0,1), 
            #     dtype=torch.float, device=self.device
            # )
            # self.test_output = torch.empty(
            #     size=(0,1), 
            #     dtype=torch.float, device=self.device
            # )

    def reset_batch(self):
        pass
        # self.training_metrics = torch.empty(
        #     size=(0,1), 
        #     dtype=torch.float, device=self.device
        # )
        # self.validation_metrics = torch.empty(
        #     size=(0,1), 
        #     dtype=torch.float, device=self.device
        # )
        # self.test_metrics = torch.empty(
        #     size=(0,1), 
        #     dtype=torch.float, device=self.device
        # )

    def evaluate_epoch(self,
        train_type='training'
    ):  
        if train_type == 'training':
            if self.output_name != None:
                self.training_output = self.metrics_list.metrics[self.output_name].batch_output
                self.metrics_list.metrics[self.output_name].reset_batch()
            if self.input_name != None:
                self.training_input = self.metrics_list.metrics[self.input_name].batch_input
                self.metrics_list.metrics[self.input_name].reset_batch()
        else:
            if self.output_name != None:
                self.validation_output = self.metrics_list.metrics[self.output_name].batch_output
                self.metrics_list.metrics[self.output_name].reset_batch()
            if self.input_name != None:
                self.validation_input = self.metrics_list.metrics[self.input_name].batch_input
                self.metrics_list.metrics[self.input_name].reset_batch()

    def evaluate_training(self):
        # plot the latent distributions
        if self.output_name != None:
            # plot all variable inputs and outputs
            fig, axs = utils.generate_plot_grid(
                len(self.input_variables),
                figsize=(10, 6)
            )
            for ii in range(len(self.input_variables)):
                axs.flat[ii].hist(
                    self.training_output[:,ii].cpu().numpy(), 
                    bins=100, 
                    label='output_training', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    self.training_input[:,ii].cpu().numpy(), 
                    bins=100, 
                    label='input_training', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f'{self.input_variables[ii]}')
            axs.flat[0].legend()
            plt.suptitle("Training input/output variables")
            plt.tight_layout()
            plt.savefig(f"plots/output/training_variables.png")
            fig, axs = utils.generate_plot_grid(
                len(self.input_variables),
                figsize=(10, 6)
            )
            for ii in range(len(self.input_variables)):
                axs.flat[ii].hist(
                    self.validation_output[:,ii].cpu().numpy(), 
                    bins=100, 
                    label='output_validation', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    self.validation_input[:,ii].cpu().numpy(), 
                    bins=100, 
                    label='input_validation', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f'{self.input_variables[ii]}')
            axs.flat[0].legend()
            plt.suptitle("Validation input/output variables")
            plt.tight_layout()
            plt.savefig(f"plots/output/validation_variables.png")
            
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