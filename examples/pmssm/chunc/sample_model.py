"""
Example script which runs SoftSUSY/micrOMEGAs
for pmssm models.
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
from chunc.dataset.pmssm import pMSSMDataset
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

    num_events = 1000
    models = [
        #"sample_models/pmssm_higgs_dm/models/chunc_pmssm_higgs_dm/chunc_pmssm_higgs_dm_trained_params.ckpt",
        #"sample_models/pmssm_higgs_dm_lsp/models/chunc_pmssm_higgs_dm_lsp/chunc_pmssm_higgs_dm_lsp_trained_params.ckpt"
        #"sample_models/pmssm_higgs_dm_lsp/models/chunc_pmssm_higgs_dm_lsp/chunc_pmssm_higgs_dm_lsp_trained_params.ckpt"
        "sample_models/pmssm_higgs_dm_lsp_no_gap/models/chunc_pmssm_higgs_dm_lsp_no_gap/chunc_pmssm_higgs_dm_lsp_no_gap_trained_params.ckpt"
    ]
    model_names = ["pmssm_higgs_dm_lsp_no_gap"]
    sigmas = [1.0,0.5,0.1,0.03,0.01,0.001,0.0001]
    num_iterations = 10

    """
    Now we load our dataset as a torch dataset,
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
        input_file='datasets/pmssm_higgs_dm_lsp_symmetric_no_gap.npz',
        features = features,
        classes = ['valid'],
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
    for ii, model in enumerate(models):
        # load the trained model
        chunc_model = CHUNC(name = 'chunc_pmssm')
        chunc_model.load_model(model)

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
                    f"mssm_output/pmssm_generated_{0.0}_{sigma}_{iteration}.txt", 
                    f"old_outputs/{now}/"
                )

        with open(f"{model_names[ii]}_validities.csv", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(validities)