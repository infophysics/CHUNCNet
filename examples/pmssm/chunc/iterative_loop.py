"""
Iterative training/sampling/training/...
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import torch
import csv
import os
import shutil
from datetime import datetime

from chunc.dataset.parameters import *
from chunc.dataset.pmssm import pMSSMDataset
from chunc.dataset.chunc import CHUNCDataset
from chunc.utils.loader import Loader
from chunc.losses import LossHandler
from chunc.optimizers import Optimizer
from chunc.metrics import MetricHandler
from chunc.trainer import IterativeTrainer
from chunc.utils.callbacks import CallbackHandler
from chunc.generator import CHUNCGenerator
from chunc.utils.distributions import generate_sphere
from chunc.utils.distributions import generate_concentric_spheres
from chunc.utils.distributions import generate_gaussian
from chunc.models import CHUNC
from chunc.utils.mssm import MSSMGenerator
from chunc.sampler import CHUNCSampler
from chunc.utils.utils import concatenate_csv, get_files, save_model


if __name__ == "__main__":
    # clean up directories first
    save_model()
    NUM_ITERATIONS = 10
    INIT_EPOCHS = 100
    ITER_EPOCHS = 25
    """
    First, we create the initial dataset from the parameter files.
    Then, generate the constrained and unconstrained subspaces
    and save the labeled training set to numpy files.
    """
    apply_constraints = False
    dataset = pMSSMDataset(
        input_dir = '../../../pmssm/pmssm_random_new/',
    )
    if apply_constraints:
        dataset.generate_constrained_dataset()
        dataset.generate_unconstrained_dataset(max_num_files = 100)
    dataset.generate_training_set(
        constrained_file    = 'constraints/higgs_dm_lsp/constrained_data.csv',
        unconstrained_file  = 'constraints/higgs_dm_lsp/unconstrained_data.csv',
        symmetric_events    = True,
        labeling    = 'binary',
        save        = True,
        output_file = 'pmssm_dataset_symmetric.npz'
    )
    """
    Then we do an initial training for 100 epochs.
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
        input_file='datasets/pmssm_dataset_symmetric.npz',
        features = features,
        classes = ['valid']
    )
    chunc_loader = Loader(
        chunc_dataset, 
        batch_size=32,
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
        'encoder_dimensions':   [25, 50, 100, 50, 25],
        'encoder_activation':   'leaky_relu',
        'encoder_activation_params':    {'negative_slope': 0.02},
        'encoder_normalization':'bias',
        # desired dimension of the latent space
        'latent_dimension':     19,
        # decoder parameters
        'decoder_dimensions':   [25, 50, 100, 50, 25],
        'decoder_activation':   'leaky_relu',
        'decoder_activation_params':    {'negative_slope': 0.02},
        'decoder_normalization':'bias',
        # output activation
        'output_activation':    'linear',
        'output_activation_params':     {},
    }
    chunc_model = CHUNC(
        name = 'chunc_pmssm',
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
                inner_radius=0.1,
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

    """Generate samples from the latent variables"""
    chunc_sampler = CHUNCSampler(
        model=chunc_model,
        latent_variables=[ii for ii in range(19)],
    )
    mssm = MSSMGenerator(
        microemgas_dir='~/physics/micromegas/micromegas_5.2.13/MSSM/', 
        softsusy_dir='~/physics/softsusy/softsusy-4.1.10/',
        param_space='pmssm',
    )
    chunc_generator_config = {
        'loader':       chunc_loader,
        'sampler':      chunc_sampler,
        'mssm_generator':mssm,
        'subspace':     'pmssm',
        'num_events':   1000,
        'num_workers':  16,
        'sample_mean':  0.0,
        'sample_sigma': 0.001
    }
    chunc_generator = CHUNCGenerator(chunc_generator_config)
    # create trainer
    chunc_trainer = IterativeTrainer(
        model=chunc_model,
        criterion=chunc_loss,
        optimizer=chunc_optimizer,
        generator=chunc_generator,
        metrics=chunc_metrics,
        callbacks=chunc_callbacks,
        metric_type='test',
        gpu=True,
        gpu_device=0
    )
    
    chunc_trainer.train(
        chunc_loader,
        iterations=NUM_ITERATIONS,
        init_epochs=INIT_EPOCHS,
        iter_epochs=ITER_EPOCHS,
        checkpoint=25
    )