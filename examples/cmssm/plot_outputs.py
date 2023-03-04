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
    # dataset = cMSSMDataset(
    #     input_dir = 'old_outputs/2023-03-04 07:28:47.900202',
    # )
    # dataset.generate_constrained_dataset(
    #     output_file="constrained_output",
    #     #input_file="cmssm_generated_0.0_0.0001_0.txt",
    #     #max_num_files=1000,
    #     apply_dm=True,
    #     apply_higgs=True,
    #     apply_lsp=True
    # )
    # dataset.generate_unconstrained_dataset(
    #     output_file="unconstraint_output",
    #     #input_file="cmssm_generated_0.0_0.0001_0.txt",
    #     max_num_files=100,
    #     apply_dm=True,
    #     apply_higgs=True,
    #     apply_lsp=True
    # )

    chunc_1 = []
    chunc_2 = []
    chunc_3 = []
    chunc_4 = []
    chunc_5 = []

    chuncc_1 = []
    chuncc_2 = []
    chuncc_3 = []
    chuncc_4 = []
    chuncc_5 = []

    input_1 = []
    input_2 = []
    input_3 = []
    input_4 = []
    input_5 = []

    with open("chunc/constraints/higgs_dm_lsp/constrained_output.csv","r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            chunc_1.append(float(row[0]))
            chunc_2.append(float(row[1]))
            chunc_3.append(float(row[2]))
            chunc_4.append(float(row[3]))
            chunc_5.append(float(row[4]))
    
    with open("chuncc/constraints/higgs_dm_lsp/constrained_output.csv","r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            chuncc_1.append(float(row[0]))
            chuncc_2.append(float(row[1]))
            chuncc_3.append(float(row[2]))
            chuncc_4.append(float(row[3]))
            chuncc_5.append(float(row[4]))
    
    with open("chunc/constraints/higgs_dm_lsp/constrained_data.csv","r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            input_1.append(float(row[0]))
            input_2.append(float(row[1]))
            input_3.append(float(row[2]))
            input_4.append(float(row[3]))
            input_5.append(float(row[4]))


    fig, axs = plt.subplots(2,3,figsize=(10,6))
    line_labels = ["Valid training data", "Valid generated data (CHUNC)", "Valid generated data (CHUNC2)"]

    axs[0][0].hist(input_1, bins=25, label="input data", histtype='step', density=True, stacked=True, color='b')
    axs[0][0].hist(chunc_1, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[0][0].hist(chuncc_1, bins=25, label="chunc2 output", histtype='step', density=True, stacked=True, color='orange')
    axs[0][0].set_yscale("log")
    #axs[0][0].set_yticks([])
    #axs[0][0].set_yticklabels([])
    axs[0][0].set_xlabel(r"$m_0$")

    axs[0][1].hist(input_2, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[0][1].hist(chunc_2, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[0][1].hist(chuncc_2, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[0][1].set_yscale("log")
    #axs[0][1].set_yticks([])
    #axs[0][1].set_yticklabels([])
    axs[0][1].set_xlabel(r"$m_{1/2}$")

    axs[0][2].hist(input_3, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[0][2].hist(chunc_3, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[0][2].hist(chuncc_3, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[0][2].set_yscale("log")
    #axs[0][2].set_yticks([])
    #axs[0][2].set_yticklabels([])
    axs[0][2].set_xlabel(r"$A_0$")

    axs[1][0].hist(input_4, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[1][0].hist(chunc_4, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[1][0].hist(chuncc_4, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[1][0].set_yscale("log")
    #axs[1][0].set_yticks([])
    #axs[1][0].set_yticklabels([])
    axs[1][0].set_xlabel(r"$\tan \beta$")

    axs[1][1].hist(input_5, bins=25, label="input data", histtype='step', density=True, stacked=True, color='b')
    axs[1][1].hist(chunc_5, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[1][1].hist(chuncc_5, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='orange')
    axs[1][1].set_yscale("log")
    #axs[1][1].set_yticks([])
    #axs[1][1].set_yticklabels([])
    axs[1][1].set_xlabel(r"$\mathrm{sgn}(\mu)$")

    axs[1][2].set_visible(False)

    #axs[0][0].legend(loc='upper left')
    #plt.suptitle(r" Valid Points (Tophat kernel: $\lambda = $" + f"{0.0001})")
    plt.figlegend(labels=line_labels, loc='lower right', bbox_to_anchor=(.95, 0.3))
    fig.supylabel("Counts (normalized)")
    plt.tight_layout()
    plt.show()

