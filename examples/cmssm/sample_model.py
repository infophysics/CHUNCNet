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
from chunc.sampler import Sampler
"""
Now we load our dataset as a torch dataset (SWAEDataset),
and then feed that into a dataloader.
"""
features = [
        'gut_m0', 
        'gut_m12', 
        'gut_A0', 
        'gut_tanb', 
        'sign_mu'
]
swae_dataset = CHUNCDataset(
    name="swae_dataset",
    input_file='datasets/cmssm_dataset_symmetric.npz',
    features = features,
    classes = ['valid']
)
swae_loader = Loader(
    swae_dataset, 
    batch_size=64,
    test_split=0.3,
    test_seed=100,
    validation_split=0.3,
    validation_seed=100,
    num_workers=4
)

# load the trained model
model_file = "models/swae_cmssm/swae_cmssm_trained_params.ckpt"
swae_model = CHUNC(name = 'swae_cmssm')
swae_model.load_model(model_file)

"""Generate samples from the latent variables"""
swae_sampler = Sampler(
    model=swae_model,
    latent_variables=[0,1,2,3,4],
    binary_variable=5
)
swae_sampler.build_histograms(
    swae_loader,
    num_latent_bins=100,
    num_binary_bins=10,
)
valid_samples = swae_sampler.sample_histograms(
    num_samples=1000,
    binary_bin=9,
    sample_type='valid'
)

"""Generate the samples from the model"""
generated_samples = swae_model.sample(valid_samples)
generated_samples = swae_dataset.unnormalize(generated_samples, detach=True)

for ii, sample in enumerate(generated_samples):
    if sample[4] < 0:
        generated_samples[ii][4] = -1
    else:
        generated_samples[ii][4] = 1
with open("cmssm_samples.txt", "w") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerows(generated_samples)

# create the MSSM object which takes the 
# path to micromegas, softsusy and the
# parameter subspace
mssm = MSSMGenerator(
    microemgas_dir='~/physics/micromegas/micromegas_5.2.13/MSSM/', 
    softsusy_dir='~/physics/softsusy/softsusy-4.1.10/',
    param_space='cmssm',
)
# run the parameters and dump the output
# to the standard folder mssm_output.
mssm.run_parameters(
    input_file="cmssm_samples.txt",
    num_events=1000,
    num_workers=16,
    save_super_invalid=False,
)

dataset = cMSSMDataset(
    input_dir = 'mssm_output/',
)
num_valid = dataset.get_number_of_valid(
    apply_dm=True,
    apply_higgs=True,
    apply_lsp=False,
)
print(num_valid)