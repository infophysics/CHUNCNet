"""
Example script which runs SoftSUSY/micrOMEGAs
for cMSSM models.
"""
from random import sample
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import torch
import csv

from chunc.utils.mssm import MSSMGenerator
from chunc.dataset.chunc import CHUNCDataset
from chunc.dataset.cmssm import cMSSMDataset
from chunc.models import CHUNC
from chunc.utils.loader import Loader
from chunc.sampler import CHUNCSampler
from chunc.sampler import CHUNCSampler
from chunc.generator import CHUNCGenerator
from chunc.utils.mssm import MSSMGenerator
import os
import shutil
from datetime import datetime

if __name__ == "__main__":

    num_events = 10
    model = "chunc_cmssm/chunc_cmssm_trained_params.ckpt"
    sigmas = [1.0,0.5,0.1,0.01,0.001,0.0001]
    num_iterations = 10

    """
    Now we load our dataset as a torch dataset,
    and then feed that into a dataloader.
    """
    features = [
            'gut_m0', 
            'gut_m12', 
            'gut_A0', 
            'gut_tanb', 
            'sign_mu'
    ]
    chunc_dataset = CHUNCDataset(
        name="chunc_dataset",
        input_file='datasets/cmssm_higgs_dm_lsp_symmetric.npz',
        features = features,
        classes = ['valid']
    )
    chunc_loader = Loader(
        chunc_dataset, 
        batch_size=64,
        test_split=0.3,
        test_seed=100,
        validation_split=0.3,
        validation_seed=100,
        num_workers=4
    )

    # load the trained model
    model_file = f"models/{model}"
    chunc_model = CHUNC(name = 'chunc_cmssm')
    chunc_model.load_model(model_file)

    """Generate samples from the latent variables"""
    chunc_sampler = CHUNCSampler(
        model=chunc_model,
        latent_variables=[0,1,2,3,4],
    )
    mssm = MSSMGenerator(
        microemgas_dir='~/physics/micromegas/micromegas_5.2.13/MSSM/', 
        softsusy_dir='~/physics/softsusy/softsusy-4.1.10/',
        param_space='cmssm',
    )
    chunc_generator_config = {
        'loader':       chunc_loader,
        'sampler':      chunc_sampler,
        'mssm_generator':mssm,
        'subspace':     'cmssm',
        'num_events':   num_events,
        'num_workers':  16,
        'sample_mean':  0.0,
        'sample_sigma': 0.01,
        'variables':    features,
    }
    chunc_generator = CHUNCGenerator(chunc_generator_config)
    """
    We want to scan over different sigma values to see how
    it correlates with validities.
    """
    
    validities = [["sigma","total","higgs","dm","higgs_dm","higgs_dm_lsp"]]

    now = datetime.now()
    if not os.path.isdir(f"old_outputs/{now}"):
        os.makedirs(f"old_outputs/{now}")
    
    for sigma in sigmas:
        for iteration in range(num_iterations):
            chunc_generator.generate(
                chunc_model,
                chunc_loader,
                mean=0.0,
                sigma=sigma,
                iteration=iteration,
            )
            num_valid = chunc_generator.check_validities()
            temp_valid = [sigma, num_events]
            for valid in num_valid:
                temp_valid.append(valid)
            validities.append(temp_valid)

            shutil.move(
                f"mssm_output/cmssm_generated_{0.0}_{sigma}_{iteration}.txt", 
                f"old_outputs/{now}/"
            )

    with open("validities.csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerows(validities)