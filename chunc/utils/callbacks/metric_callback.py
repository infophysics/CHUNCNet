"""
Generic metric callback
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from chunc.metrics.savers import *
from chunc.utils.callbacks import GenericCallback

class MetricCallback(GenericCallback):
    """
    """
    def __init__(self,
        metrics_list
    ):  
        super(MetricCallback, self).__init__()
        self.metrics_list = metrics_list
        if metrics_list != None:
            self.metric_names = [
                name for name, metric in self.metrics_list.metrics.items()
                if not sum([
                    isinstance(metric, LatentSaver),
                    isinstance(metric, OutputSaver),
                    isinstance(metric, TargetSaver),
                    isinstance(metric, InputSaver)
                ])
            ]

        # containers for training metrics
        if metrics_list != None:
            # containers for training metric
            self.training_metrics = torch.empty(
                size=(0,len(self.metric_names)), 
                dtype=torch.float, device=self.device
            )
            self.validation_metrics = torch.empty(
                size=(0,len(self.metric_names)), 
                dtype=torch.float, device=self.device
            )
            self.test_metrics = torch.empty(
                size=(0,len(self.metric_names)), 
                dtype=torch.float, device=self.device
            )

    def reset_batch(self):
        self.training_metrics = torch.empty(
            size=(0,len(self.metric_names)), 
            dtype=torch.float, device=self.device
        )
        self.validation_metrics = torch.empty(
            size=(0,len(self.metric_names)), 
            dtype=torch.float, device=self.device
        )
        self.test_metrics = torch.empty(
            size=(0,len(self.metric_names)), 
            dtype=torch.float, device=self.device
        )

    def evaluate_epoch(self,
        train_type='training'
    ):  
        temp_metrics = torch.empty(
            size=(1,0), 
            dtype=torch.float, device=self.device
        )
        for name in self.metric_names:
            temp_metric = torch.tensor(
                [[self.metrics_list.metrics[name].compute()]], 
                device=self.device
            )
            temp_metrics = torch.cat(
                (temp_metrics, temp_metric),
                dim=1
            )
            self.metrics_list.metrics[name].reset_batch()
        # run through criteria
        if train_type == 'training':
            self.training_metrics = torch.cat(
                (self.training_metrics, temp_metrics),
                dim=0
            )
        elif train_type == 'validation':
            self.validation_metrics = torch.cat(
                (self.validation_metrics, temp_metrics),
                dim=0
            )
        else:
            self.test_metrics = torch.cat(
                (self.test_metrics, temp_metrics),
                dim=0
            )

    def evaluate_training(self):
        pass

    def evaluate_testing(self):  
        # evaluate metrics from training and validation
        if self.metrics_list == None:
            return
        epoch_ticks = np.arange(1,self.epochs+1)
        # training plot
        fig, axs = plt.subplots(figsize=(10,5))
        if len(self.training_metrics) != 0:
            for ii, metric in enumerate(self.metric_names):
                temp_metric = self.training_metrics[:,ii]
                final_metric_value = f"(final={temp_metric[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_metric_value}"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("metric")
            axs.set_yscale('log')
            plt.title("metric vs. epoch (training)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("plots/epoch_training_metrics.png")
        
        if len(self.validation_metrics) != 0:
            fig, axs = plt.subplots(figsize=(10,5))
            for ii, metric in enumerate(self.metric_names):
                temp_metric = self.validation_metrics[:,ii]
                final_metric_value = f"(final={temp_metric[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_metric_value}"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("metric")
            axs.set_yscale('log')
            plt.title("metric vs. epoch (validation)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("plots/epoch_validation_metrics.png")

        if len(self.training_metrics) != 0 and len(self.validation_metrics) != 0:
            fig, axs = plt.subplots(figsize=(10,5))
            for ii, metric in enumerate(self.metric_names):
                temp_training_metric = self.training_metrics[:,ii]
                temp_validation_metric = self.validation_metrics[:,ii]
                final_training_metric_value = f"(final={temp_training_metric[-1]:.2e})"
                final_validation_metric_value = f"(final={temp_validation_metric[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_training_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    linestyle='-',
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_training_metric_value}"
                )
                axs.plot(
                    epoch_ticks,
                    temp_validation_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    linestyle='--',
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_validation_metric_value}"
                )
            if len(self.test_metrics) != 0:
                for ii, metric in enumerate(self.metric_names):
                    temp_metric = self.test_metrics[:,ii]
                    final_metric_value = f"(final={temp_metric[-1]:.2e})"
                    axs.plot([],[],
                        marker='x',
                        linestyle='',
                        c=self.plot_colors[-(ii+1)],
                        label=rf"(test) {metric}"
                    )
                    axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_metric_value}"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("metric")
            axs.set_yscale('log')
            plt.title("metric vs. epoch")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("plots/epoch_metrics.png")

    def evaluate_inference(self):
        pass