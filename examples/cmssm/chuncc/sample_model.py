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
from chunc.models import CHUNCC
from chunc.utils.loader import Loader
from chunc.sampler import CHUNCCSampler
from chunc.generator import CHUNCCGenerator
from chunc.utils.mssm import MSSMGenerator
import os
import shutil
from datetime import datetime

if __name__ == "__main__":

    num_events = 10000
    models = [
        "sample_models/cmssm_higgs_dm/models/chuncc_cmssm_higgs_dm/chuncc_cmssm_higgs_dm_trained_params.ckpt",
        "sample_models/cmssm_higgs_dm_lsp/models/chuncc_cmssm_higgs_dm_lsp/chuncc_cmssm_higgs_dm_lsp_trained_params.ckpt"
    ]
    model_names = ["cmssm_higgs_dm", "cmssm_higgs_dm_lsp"]
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
    chuncc_dataset = CHUNCDataset(
        name="chuncc_dataset",
        input_file='datasets/cmssm_higgs_dm_lsp_symmetric.npz',
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
    for ii, model in enumerate(models):
        # load the trained model
        chuncc_model = CHUNCC(name = 'chuncc_cmssm')
        chuncc_model.load_model(model)

        """Generate samples from the latent variables"""
        chuncc_sampler = CHUNCCSampler(
            model=chuncc_model,
            latent_variables=[0,1,2,3,4],
            binary_variable=5,
            num_latent_bins=25,
            num_binary_bins=10,
        )
        mssm = MSSMGenerator(
            microemgas_dir='~/physics/micromegas/micromegas_5.2.13/MSSM/', 
            softsusy_dir='~/physics/softsusy/softsusy-4.1.10/',
            param_space='cmssm',
        )
        chuncc_generator_config = {
            'loader':       chuncc_loader,
            'sampler':      chuncc_sampler,
            'mssm_generator':mssm,
            'subspace':     'cmssm',
            'num_events':   num_events,
            'num_workers':  16,
            'binary_bin':   9,
        }
        chuncc_generator = CHUNCCGenerator(chuncc_generator_config)
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
                chuncc_generator.generate(
                    chuncc_model,
                    chuncc_loader,
                    mean=0.0,
                    sigma=sigma,
                    iteration=iteration,
                )
                num_valid = chuncc_generator.check_validities()
                temp_valid = [sigma, num_events]
                for valid in num_valid:
                    temp_valid.append(valid)
                validities.append(temp_valid)

                shutil.move(
                    f"mssm_output/cmssm_generated_{0.0}_{sigma}_{iteration}.txt", 
                    f"old_outputs/{now}/"
                )

        with open(f"{model_names[ii]}_validities.csv", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(validities)