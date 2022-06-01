"""
Code for training on different constraints
"""
# CHUNC imports
from chunc.dataset.chunc import CHUNCDataset
from chunc.dataset.mapper import MSSMMapper
from chunc.utils.loader import Loader
from chunc.losses import LossHandler
from chunc.optimizers import Optimizer
from chunc.metrics import MetricHandler
from chunc.trainer import Trainer
from chunc.utils.callbacks import CallbackHandler
from chunc.utils.distributions import generate_concentric_spheres
from chunc.utils.utils import get_files, save_model
from chunc.models import CHUNC
import numpy as np
import torch
import os
import shutil
from datetime import datetime


if __name__ == "__main__":
    # clean up folders
    save_model()

    constraints = ["higgs_dm_lsp"]
    epochs = [100]

    for ii, constraint in enumerate(constraints):
        """
        Now we load our dataset as a torch dataset (chuncDataset),
        and then feed that into a dataloader.
        """
        features = [
            'gut_m1', 'gut_m2', 
            'gut_m3', 'gut_mmu', 
            'gut_mA', 'gut_At', 
            'gut_Ab', 'gut_Atau', 
            'gut_mL1','gut_mL3', 
            'gut_me1','gut_mtau1', 
            'gut_mQ1','gut_mQ3', 
            'gut_mu1','gut_mu3', 
            'gut_md1','gut_md3', 
            'gut_tanb'
        ]
        chunc_dataset = CHUNCDataset(
            name="chunc_dataset",
            input_file=f'datasets/pmssm_{constraint}_symmetric_no_gap.npz',
            features = features,
            classes = ['valid']
        )
        chunc_loader = Loader(
            chunc_dataset, 
            batch_size=64,
            test_split=0.1,
            test_seed=100,
            validation_split=0.1,
            validation_seed=100,
            num_workers=4
        )
        """
        Construct the chunc Model, specify the loss and the 
        optimizer and metrics.
        """
        chunc_pmssm_config = {
            # dimension of the input variables
            'input_dimension':      19,
            # encoder parameters
            'encoder_dimensions':   [50, 100, 200, 100, 50],
            'encoder_activation':   'leaky_relu',
            'encoder_activation_params':    {'negative_slope': 0.02},
            'encoder_normalization':'bias',
            # desired dimension of the latent space
            'latent_dimension':     19,
            # decoder parameters
            'decoder_dimensions':   [50, 100, 200, 100, 50],
            'decoder_activation':   'leaky_relu',
            'decoder_activation_params':    {'negative_slope': 0.02},
            'decoder_normalization':'bias',
            # output activation
            'output_activation':    'linear',
            'output_activation_params':     {},
        }
        chunc_model = CHUNC(
            name = f'chunc_pmssm_{constraint}_no_gap',
            cfg  = chunc_pmssm_config
        ) 

        # create loss, optimizer and metrics
        chunc_optimizer = Optimizer(
            model=chunc_model,
            optimizer='Adam'
        )

        # create criterions
        chunc_loss_config = {
            'L2OutputLoss':   {
                'alpha':    1.0,
                'reduction':'mean',
            },
            'LatentWassersteinLoss': {
                'alpha':    1.0,
                'latent_variables': [ii for ii in range(19)],
                'distribution':     generate_concentric_spheres(
                    number_of_samples=10000,
                    dimension=19,
                    inner_radius=0.3,
                    outer_radius=1.0,
                    thickness=0.3,
                    save_plot=True,
                ),
                'num_projections':  1000,
            },
            'LatentClusterLoss':    {
                'alpha':    1.0,
                'latent_variables': [ii for ii in range(19)],
                'cluster_type': 'fixed',
                'fixed_value':  1.0,
            }
            
        }
        chunc_loss = LossHandler(
            name="chunc_loss",
            cfg=chunc_loss_config,
        )
        
        # create metrics
        chunc_metric_config = {
            'LatentSaver':  {},
            'TargetSaver':  {},
            'InputSaver':   {},
            'OutputSaver':  {},
        }
        chunc_metrics = MetricHandler(
            "chunc_metric",
            cfg=chunc_metric_config,
        )

        # create callbacks
        callback_config = {
            'loss':   {'criterion_list': chunc_loss},
            'metric': {'metrics_list':   chunc_metrics},
            'latent': {
                'criterion_list':   chunc_loss,
                'metrics_list':     chunc_metrics,
                'latent_variables': [ii for ii in range(19)],
            },
            'cluster': {
                'criterion_list':   chunc_loss,
                'metrics_list':     chunc_metrics,
                'latent_variables': [ii for ii in range(19)],
            },
            'output':   {
                'criterion_list':   chunc_loss,
                'metrics_list':     chunc_metrics,
                'input_variables':  features,
            }
        }
        chunc_callbacks = CallbackHandler(
            "chunc_callbacks",
            callback_config
        )

        # create trainer
        chunc_trainer = Trainer(
            model=chunc_model,
            criterion=chunc_loss,
            optimizer=chunc_optimizer,
            metrics=chunc_metrics,
            callbacks=chunc_callbacks,
            metric_type='test',
            gpu=True,
            gpu_device=0
        )
        
        chunc_trainer.train(
            chunc_loader,
            epochs=epochs[ii],
            checkpoint=25
        )

        # # run mapper
        # chunc_mapper = MSSMMapper(
        #     chunc_dataset,
        #     chunc_model
        # )
        # chunc_mapper.run_mapper(
        #     num_covers=50
        # )

        # clean up
        save_model(f"pmssm_{constraint}_no_gap")