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
from chunc.dataset.pmssm import pMSSMDataset
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
    dataset = pMSSMDataset(
        input_dir = 'old_outputs/2023-03-04 09:02:47.886157',
    )
    dataset.generate_constrained_dataset(
        output_file="constrained_output",
        #input_file="pmssm_generated_0.0_0.0001_0.txt",
        #max_num_files=1000,
        apply_dm=True,
        apply_higgs=True,
        apply_lsp=True
    )
    dataset.generate_unconstrained_dataset(
        output_file="unconstraint_output",
        #input_file="pmssm_generated_0.0_0.0001_0.txt",
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
    output_6 = []
    output_7 = []
    output_8 = []
    output_9 = []
    output_10 = []
    output_11 = []
    output_12 = []
    output_13 = []
    output_14 = []
    output_15 = []
    output_16 = []
    output_17 = []
    output_18 = []
    output_19 = []

    input_1 = []
    input_2 = []
    input_3 = []
    input_4 = []
    input_5 = []
    input_6 = []
    input_7 = []
    input_8 = []
    input_9 = []
    input_10 = []
    input_11 = []
    input_12 = []
    input_13 = []
    input_14 = []
    input_15 = []
    input_16 = []
    input_17 = []
    input_18 = []
    input_19 = []

    with open("constraints/higgs_dm_lsp/constrained_output.csv","r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            output_1.append(float(row[0]))
            output_2.append(float(row[1]))
            output_3.append(float(row[2]))
            output_4.append(float(row[3]))
            output_5.append(float(row[4]))
            output_6.append(float(row[5]))
            output_7.append(float(row[6]))
            output_8.append(float(row[7]))
            output_9.append(float(row[8]))
            output_10.append(float(row[9]))
            output_11.append(float(row[10]))
            output_12.append(float(row[11]))
            output_13.append(float(row[12]))
            output_14.append(float(row[13]))
            output_15.append(float(row[14]))
            output_16.append(float(row[15]))
            output_17.append(float(row[16]))
            output_18.append(float(row[17]))
            output_19.append(float(row[18]))
    
    with open("constraints/higgs_dm_lsp/constrained_data.csv","r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            input_1.append(float(row[0]))
            input_2.append(float(row[1]))
            input_3.append(float(row[2]))
            input_4.append(float(row[3]))
            input_5.append(float(row[4]))
            input_6.append(float(row[5]))
            input_7.append(float(row[6]))
            input_8.append(float(row[7]))
            input_9.append(float(row[8]))
            input_10.append(float(row[9]))
            input_11.append(float(row[10]))
            input_12.append(float(row[11]))
            input_13.append(float(row[12]))
            input_14.append(float(row[13]))
            input_15.append(float(row[14]))
            input_16.append(float(row[15]))
            input_17.append(float(row[16]))
            input_18.append(float(row[17]))
            input_19.append(float(row[18]))

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

    fig, axs = plt.subplots(5,4,figsize=(10,8))
    axs[0][0].hist(input_1, bins=25, label="input data", histtype='step', density=True, stacked=True, color='k')
    axs[0][0].hist(output_1, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[0][0].set_yticks([])
    axs[0][0].set_yticklabels([])
    axs[0][0].set_xlabel("gut_m1")

    axs[0][1].hist(input_2, bins=25, label="input data", histtype='step', density=True, stacked=True, color='k')
    axs[0][1].hist(output_2, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[0][1].set_yticks([])
    axs[0][1].set_yticklabels([])
    axs[0][1].set_xlabel("gut_m2")

    axs[0][2].hist(input_3, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[0][2].hist(output_3, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[0][2].set_yticks([])
    axs[0][2].set_yticklabels([])
    axs[0][2].set_xlabel("gut_m3")

    axs[0][3].hist(input_4, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[0][3].hist(output_4, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[0][3].set_yticks([])
    axs[0][3].set_yticklabels([])
    axs[0][3].set_xlabel("gut_mmu")

    axs[1][0].hist(input_5, bins=25, label="input data", histtype='step', density=True, stacked=True, color='k')
    axs[1][0].hist(output_5, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[1][0].set_yticks([])
    axs[1][0].set_yticklabels([])
    axs[1][0].set_xlabel("gut_mA")

    axs[1][1].hist(input_6, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[1][1].hist(output_6, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[1][1].set_yticks([])
    axs[1][1].set_yticklabels([])
    axs[1][1].set_xlabel("gut_At")

    axs[1][2].hist(input_7, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[1][2].hist(output_7, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[1][2].set_yticks([])
    axs[1][2].set_yticklabels([])
    axs[1][2].set_xlabel("gut_Ab")

    axs[1][3].hist(input_8, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[1][3].hist(output_8, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[1][3].set_yticks([])
    axs[1][3].set_yticklabels([])
    axs[1][3].set_xlabel("gut_Atau")

    axs[2][0].hist(input_9, bins=25, label="input data", histtype='step', density=True, stacked=True, color='k')
    axs[2][0].hist(output_9, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[2][0].set_yticks([])
    axs[2][0].set_yticklabels([])
    axs[2][0].set_xlabel("gut_mL1")

    axs[2][1].hist(input_10, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[2][1].hist(output_10, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[2][1].set_yticks([])
    axs[2][1].set_yticklabels([])
    axs[2][1].set_xlabel("gut_mL3")

    axs[2][2].hist(input_11, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[2][2].hist(output_11, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[2][2].set_yticks([])
    axs[2][2].set_yticklabels([])
    axs[2][2].set_xlabel("gut_me1")

    axs[2][3].hist(input_12, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[2][3].hist(output_12, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[2][3].set_yticks([])
    axs[2][3].set_yticklabels([])
    axs[2][3].set_xlabel("gut_mtau1")

    axs[3][0].hist(input_13, bins=25, label="input data", histtype='step', density=True, stacked=True, color='k')
    axs[3][0].hist(output_13, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[3][0].set_yticks([])
    axs[3][0].set_yticklabels([])
    axs[3][0].set_xlabel("gut_Q1")

    axs[3][1].hist(input_14, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[3][1].hist(output_14, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[3][1].set_yticks([])
    axs[3][1].set_yticklabels([])
    axs[3][1].set_xlabel("gut_Q3")

    axs[3][2].hist(input_15, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[3][2].hist(output_15, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[3][2].set_yticks([])
    axs[3][2].set_yticklabels([])
    axs[3][2].set_xlabel("gut_mu1")

    axs[3][3].hist(input_16, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[3][3].hist(output_16, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[3][3].set_yticks([])
    axs[3][3].set_yticklabels([])
    axs[3][3].set_xlabel("gut_mu3")

    axs[4][0].hist(input_17, bins=25, label="input data", histtype='step', density=True, stacked=True, color='k')
    axs[4][0].hist(output_17, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[4][0].set_yticks([])
    axs[4][0].set_yticklabels([])
    axs[4][0].set_xlabel("gut_md1")

    axs[4][1].hist(input_18, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[4][1].hist(output_18, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[4][1].set_yticks([])
    axs[4][1].set_yticklabels([])
    axs[4][1].set_xlabel("gut_md3")

    axs[4][2].hist(input_19, bins=25, histtype='step', density=True, stacked=True, color='k')
    axs[4][2].hist(output_19, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[4][2].set_yticks([])
    axs[4][2].set_yticklabels([])
    axs[4][2].set_xlabel("gut_tanb")



    axs[4][3].set_visible(False)

    axs[0][1].legend(loc='upper left')
    plt.suptitle(r"CHUNCC Valid Points")
    plt.tight_layout()
    plt.show()

