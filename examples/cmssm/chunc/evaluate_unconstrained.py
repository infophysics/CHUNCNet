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
from chunc.sampler import CHUNCKDESampler
from chunc.sampler import CHUNCSampler
from chunc.generator import CHUNCGenerator
from chunc.utils.mssm import MSSMGenerator
import os
import shutil
from datetime import datetime
from tqdm import tqdm

if __name__ == "__main__":

    models = [
        #"sample_models/cmssm_higgs_dm2/models/chunc_cmssm_higgs_dm/chunc_cmssm_higgs_dm_trained_params.ckpt",
        "sample_models/cmssm_higgs_dm_lsp2/models/chunc_cmssm_higgs_dm_lsp/chunc_cmssm_higgs_dm_lsp_trained_params.ckpt"
    ]
    model_names = [
        #"cmssm_higgs_dm", 
        "cmssm_higgs_dm_lsp"
    ]
    constrained_datasets = ["datasets/cmssm_higgs_dm_lsp_symmetric.npz"]
    unconstrained_datasets = [f"datasets/cmssm_higgs_dm_lsp_unconstrained_{ii}.npz" for ii in range(10)]
    constrained_output = []
    unconstrained_training = []
    unconstrained_output = []
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
        input_file=constrained_datasets[0],
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
    for ii, model in enumerate(models):
        # load the trained model
        chunc_model = CHUNC(name = 'chunc_cmssm')
        chunc_model.load_model(model)
        
        inference_loop = tqdm(
            enumerate(chunc_loader.inference_loader, 0), 
            total=len(chunc_loader.inference_loader), 
            leave=False,
            colour='magenta'
        )

        predictions = torch.empty(size=(0,5), dtype=torch.float)
        latent = torch.empty(size=(0,5), dtype=torch.float)
        uncon_latent = torch.empty(size=(0,5), dtype=torch.float)
    
        chunc_model.eval()
        with torch.no_grad():
            for ii, data in inference_loop:
                # get the network output
                outputs, latent_outputs = chunc_model(data)
                                
                predictions = torch.cat((predictions, outputs),dim=0)
                if data[1][0] == 1.0:
                    latent = torch.cat((latent, latent_outputs),dim=0)
                else:
                    uncon_latent = torch.cat((uncon_latent, latent_outputs),dim=0)
                
        constrained_output.append(torch.norm(latent, dim=1).numpy())
        unconstrained_training.append(torch.norm(uncon_latent, dim=1).numpy())

    for kk in range(len(unconstrained_datasets)):
        chunc_dataset = CHUNCDataset(
            name="chunc_dataset",
            input_file=unconstrained_datasets[kk],
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
        for ii, model in enumerate(models):
            # load the trained model
            chunc_model = CHUNC(name = 'chunc_cmssm')
            chunc_model.load_model(model)
            
            inference_loop = tqdm(
                enumerate(chunc_loader.inference_loader, 0), 
                total=len(chunc_loader.inference_loader), 
                leave=False,
                colour='magenta'
            )

            predictions = torch.empty(size=(0,5), dtype=torch.float)
            latent = torch.empty(size=(0,5), dtype=torch.float)
        
            chunc_model.eval()
            with torch.no_grad():
                for ii, data in inference_loop:
                    # get the network output
                    outputs, latent_outputs = chunc_model(data)
                    
                    predictions = torch.cat((predictions, outputs),dim=0)
                    latent = torch.cat((latent, latent_outputs),dim=0)
                    
            unconstrained_output.append(torch.norm(latent, dim=1).numpy())

    fig, axs = plt.subplots(figsize=(10,6))
    axs.hist(
        constrained_output,
        bins=100,
        density=True,
        stacked=True,
        histtype='step',
        label=f'constrained training'
    )
    axs.hist(
        unconstrained_training,
        bins=100,
        density=True,
        stacked=True,
        histtype='step',
        label=f'unconstrained training'
    )
    for kk in range(len(unconstrained_output)):
        axs.hist(
            unconstrained_output[kk],
            bins=100,
            density=True,
            stacked=True,
            histtype='step',
            label=f'unconstrained subset {kk}'
        )
    axs.set_xlabel(r"Distance from the origin $|\vec{r}|$")
    axs.set_title("cMSSM CHUNC latent representation")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/latent_unconstrained_subsets_distance.png")

