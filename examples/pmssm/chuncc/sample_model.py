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
from chunc.models import CHUNCC
from chunc.utils.loader import Loader
from chunc.sampler import CHUNCCSampler
from chunc.generator import CHUNCCGenerator
from chunc.utils.mssm import MSSMGenerator
import os
import shutil
from datetime import datetime

if __name__ == "__main__":

    num_events = 1000
    models = [
        #"sample_models/pmssm_higgs_dm/models/chuncc_pmssm_higgs_dm/chuncc_pmssm_higgs_dm_trained_params.ckpt",
        "sample_models/pmssm_higgs_dm_lsp_no_gap/models/chuncc_pmssm_higgs_dm_lsp_no_gap/chuncc_pmssm_higgs_dm_lsp_no_gap_trained_params.ckpt"
    ]
    model_names = ["pmssm_higgs_dm_lsp_no_gap"]
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
    chuncc_dataset = CHUNCDataset(
        name="chuncc_dataset",
        input_file='datasets/pmssm_higgs_dm_lsp_symmetric_no_gap.npz',
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
        chuncc_model = CHUNCC(name = 'chuncc_pmssm')
        chuncc_model.load_model(model)

        """Generate samples from the latent variables"""
        chuncc_sampler = CHUNCCSampler(
            model=chuncc_model,
            latent_variables=[ii for ii in range(19)],
            binary_variable=19,
            num_latent_bins=25,
            num_binary_bins=25,
        )
        mssm = MSSMGenerator(
            microemgas_dir='~/physics/micromegas/micromegas_5.2.13/MSSM/', 
            softsusy_dir='~/physics/softsusy/softsusy-4.1.10/',
            param_space='pmssm',
        )
        chuncc_generator_config = {
            'loader':       chuncc_loader,
            'sampler':      chuncc_sampler,
            'mssm_generator':mssm,
            'subspace':     'pmssm',
            'num_events':   num_events,
            'num_workers':  16,
            'binary_bin':   24,
            'variables':    features,
        }
        chuncc_generator = CHUNCCGenerator(chuncc_generator_config)
        """
        We want to scan over different sigma values to see how
        it correlates with validities.
        """
        
        validities = [["total","higgs","dm","higgs_dm","higgs_dm_lsp"]]

        now = datetime.now()
        if not os.path.isdir(f"old_outputs/{now}"):
            os.makedirs(f"old_outputs/{now}")
        
        for iteration in range(num_iterations):
            chuncc_generator.generate(
                chuncc_model,
                chuncc_loader,   
                iteration=iteration,
            )
            num_valid = chuncc_generator.check_validities()
            temp_valid = [num_events]
            for valid in num_valid:
                temp_valid.append(valid)
            validities.append(temp_valid)

            shutil.move(
                f"mssm_output/pmssm_generated_{iteration}.txt", 
                f"old_outputs/{now}/"
            )

        with open(f"{model_names[ii]}_validities.csv", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(validities)