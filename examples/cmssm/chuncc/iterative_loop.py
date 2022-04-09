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
from chunc.dataset.cmssm import cMSSMDataset
from chunc.dataset.chunc import CHUNCDataset
from chunc.utils.loader import Loader
from chunc.losses import LossHandler
from chunc.optimizers import Optimizer
from chunc.metrics import MetricHandler
from chunc.trainer import IterativeTrainer
from chunc.utils.callbacks import CallbackHandler
from chunc.generator import Generator
from chunc.utils.distributions import generate_sphere
from chunc.utils.distributions import generate_concentric_spheres
from chunc.utils.distributions import generate_gaussian
from chunc.models import CHUNCC
from chunc.utils.mssm import MSSMGenerator
from chunc.sampler import Sampler
from chunc.utils.utils import concatenate_csv


if __name__ == "__main__":

    NUM_ITERATIONS = 10
    INIT_EPOCHS = 10
    ITER_EPOCHS = 10
    """
    First, we create the initial dataset from the parameter files.
    Then, generate the constrained and unconstrained subspaces
    and save the labeled training set to numpy files.
    """
    apply_constraints = False
    dataset = cMSSMDataset(
        input_dir = '../../../cmssm/cmssm_random_new/',
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
        output_file = 'cmssm_dataset_symmetric.npz'
    )
    """
    Then we do an initial training for 100 epochs.
    Now we load our dataset as a torch dataset (chunccDataset),
    and then feed that into a dataloader.
    """
    features = [
            'gut_m0', 
            'gut_m12', 
            'gut_A0', 
            'gut_tanb', 
            'sign_mu'
    ]
    chuncc_dataset = CHUNCDataset(
        name="chuncc_dataset",
        input_file='datasets/cmssm_dataset_symmetric.npz',
        features = features,
        classes = ['valid']
    )
    chuncc_loader = Loader(
        chuncc_dataset, 
        batch_size=64,
        test_split=0.3,
        test_seed=100,
        validation_split=0.3,
        validation_seed=100,
        num_workers=4
    )
    """
    Construct the chuncc Model, specify the loss and the 
    optimizer and metrics.
    """
    chuncc_cmssm_config = {
        # dimension of the input variables
        'input_dimension':      5,
        # encoder parameters
        'encoder_dimensions':   [25, 50, 100, 50, 25],
        'encoder_activation':   'leaky_relu',
        'encoder_activation_params':    {'negative_slope': 0.02},
        'encoder_normalization':'bias',
        # desired dimension of the latent space
        'latent_dimension':     5,
        'latent_binary':        1,
        'latent_binary_activation': 'sigmoid',
        'latent_binary_activation_params':  {},
        # decoder parameters
        'decoder_dimensions':   [25, 50, 100, 50, 25],
        'decoder_activation':   'leaky_relu',
        'decoder_activation_params':    {'negative_slope': 0.02},
        'decoder_normalization':'bias',
        # output activation
        'output_activation':    'linear',
        'output_activation_params':     {},
    }
    chuncc_model = CHUNCC(
        name = 'chuncc_cmssm_init',
        cfg  = chuncc_cmssm_config
    ) 

    # create loss, optimizer and metrics
    chuncc_optimizer = Optimizer(
        model=chuncc_model,
        optimizer='Adam'
    )

    # create criterions
    chuncc_loss_config = {
        'L2OutputLoss':   {
            'alpha':    1.0,
            'reduction':'mean',
        },
        'LatentWassersteinLoss': {
            'alpha':    1.0,
            'latent_variables': [0,1,2,3,4],
            'distribution':     generate_gaussian(dimension=5),
            'num_projections':  1000,
        },
        'LatentBinaryLoss': {
            'alpha':    1.0,
            'binary_variable':  5,
            'reduction':    'mean',
        }
    }
    chuncc_loss = LossHandler(
        name="chuncc_loss",
        cfg=chuncc_loss_config,
    )
    
    # create metrics
    chuncc_metric_config = {
        'LatentBinaryAccuracy': {
            'cutoff':   0.5,
            'binary_variable':  5,
        },
        'LatentSaver':  {},
        'TargetSaver':  {},
        'InputSaver':   {},
        'OutputSaver':  {},
    }
    chuncc_metrics = MetricHandler(
        "chuncc_metric",
        cfg=chuncc_metric_config,
    )

    # create callbacks
    callback_config = {
        'loss':   {'criterion_list': chuncc_loss},
        'metric': {'metrics_list':   chuncc_metrics},
        'latent': {
            'criterion_list':   chuncc_loss,
            'metrics_list':     chuncc_metrics,
            'latent_variables': [0,1,2,3,4],
            'binary_variable':  5,
            'binary_bins':      10,
        },
        'output':   {
            'criterion_list':   chuncc_loss,
            'metrics_list':     chuncc_metrics,
            'input_variables':  features,
        }
    }
    chuncc_callbacks = CallbackHandler(
        "chuncc_callbacks",
        callback_config
    )
    """Generate samples from the latent variables"""
    chuncc_sampler = Sampler(
        model=chuncc_model,
        latent_variables=[0,1,2,3,4],
        binary_variable=5,
        num_latent_bins=25,
        num_binary_bins=10
    )
    mssm = MSSMGenerator(
        microemgas_dir='~/physics/micromegas/micromegas_5.2.13/MSSM/', 
        softsusy_dir='~/physics/softsusy/softsusy-4.1.10/',
        param_space='cmssm',
    )
    chuncc_generator_config = {
        'sampler':      chuncc_sampler,
        'mssm_generator':mssm,
        'subspace':     'cmssm',
        'num_events':   1000,
        'num_workers':  16,
        'binary_bin':   9,
    }
    chuncc_generator = Generator(chuncc_generator_config)
    # create trainer
    chuncc_trainer = IterativeTrainer(
        model=chuncc_model,
        criterion=chuncc_loss,
        optimizer=chuncc_optimizer,
        generator=chuncc_generator,
        metrics=chuncc_metrics,
        callbacks=chuncc_callbacks,
        metric_type='test',
        gpu=True,
        gpu_device=0
    )
    
    chuncc_trainer.train(
        chuncc_loader,
        iterations=NUM_ITERATIONS,
        init_epochs=INIT_EPOCHS,
        iter_epochs=ITER_EPOCHS,
        checkpoint=25
    )