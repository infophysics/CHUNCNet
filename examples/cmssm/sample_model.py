"""
Example script which runs SoftSUSY/micrOMEGAs
for cMSSM models.
"""
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

# load the trained model
model_file = "models/swae_cmssm/swae_cmssm_trained_params.ckpt"
swae_model = CHUNC(name = 'swae_cmssm')
swae_model.load_model(model_file)

"""Generate samples from the latent variables"""
N = 100
mean = 0.0
sigma = 0.1
latent_samples = torch.normal(mean, sigma, size=(N, 5))

"""Generate samples from the class variable"""
class_samples = torch.full(size=(N,1), fill_value=.995)

"""Generate the samples from the model"""
samples = torch.cat((latent_samples, class_samples), dim=1)
generated_samples = swae_model.sample(samples)
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
    num_events=10000,
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