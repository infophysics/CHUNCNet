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

if __name__ == "__main__":

    """
    First, we create the dataset from the parameter files.
    Then, generate the constrained and unconstrained subspaces
    and save the labeled training set to numpy files.
    """
    dataset = cMSSMDataset(
        input_dir = 'old_outputs/2022-07-18 07:37:44.250010',
    )
    dataset.generate_constrained_dataset(
        output_file="constrained_output",
        #input_file="cmssm_generated_0.0_0.0001_0.txt",
        #max_num_files=1000,
        apply_dm=True,
        apply_higgs=True,
        apply_lsp=True
    )
    dataset.generate_unconstrained_dataset(
        output_file="unconstraint_output",
        #input_file="cmssm_generated_0.0_0.0001_0.txt",
        max_num_files=100,
        apply_dm=True,
        apply_higgs=True,
        apply_lsp=True
    )

    output_1 = []
    output_2 = []
    output_3 = []
    output_4 = []
    output_5 = []

    input_1 = []
    input_2 = []
    input_3 = []
    input_4 = []
    input_5 = []

    with open("constraints/higgs_dm_lsp/constrained_output.csv","r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            output_1.append(float(row[0]))
            output_2.append(float(row[1]))
            output_3.append(float(row[2]))
            output_4.append(float(row[3]))
            output_5.append(float(row[4]))
    
    with open("constraints/higgs_dm_lsp/constrained_data.csv","r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            input_1.append(float(row[0]))
            input_2.append(float(row[1]))
            input_3.append(float(row[2]))
            input_4.append(float(row[3]))
            input_5.append(float(row[4]))

    fig, axs = plt.subplots(2,3,figsize=(10,6))
    axs[0][0].hist(input_1, bins=25, label="input data", histtype='step', density=True, stacked=True, color='k')
    axs[0][0].hist(output_1, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[0][0].set_yticks([])
    axs[0][0].set_yticklabels([])
    axs[0][0].set_xlabel("gut_m0")

    axs[0][1].hist(input_2, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[0][1].hist(output_2, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[0][1].set_yticks([])
    axs[0][1].set_yticklabels([])
    axs[0][1].set_xlabel("gut_m1")

    axs[0][2].hist(input_3, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[0][2].hist(output_3, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[0][2].set_yticks([])
    axs[0][2].set_yticklabels([])
    axs[0][2].set_xlabel("gut_A0")

    axs[1][0].hist(input_4, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[1][0].hist(output_4, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[1][0].set_yticks([])
    axs[1][0].set_yticklabels([])
    axs[1][0].set_xlabel("gut_tanb")

    axs[1][1].hist(input_5, bins=25, label="input data", histtype='step', density=True, stacked=True, color='k')
    axs[1][1].hist(output_5, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[1][1].set_yticks([])
    axs[1][1].set_yticklabels([])
    axs[1][1].set_xlabel("gut_signmu")

    axs[1][2].set_visible(False)

    axs[0][0].legend(loc='upper left')
    plt.suptitle(r"CHUNC Valid Points (Tophat kernel: $\lambda = $" + f"{0.0001})")
    plt.tight_layout()
    plt.show()

