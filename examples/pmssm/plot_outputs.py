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
    chunc_6 = []
    chunc_7 = []
    chunc_8 = []
    chunc_9 = []
    chunc_10 = []
    chunc_11 = []
    chunc_12 = []
    chunc_13 = []
    chunc_14 = []
    chunc_15 = []
    chunc_16 = []
    chunc_17 = []
    chunc_18 = []
    chunc_19 = []

    chuncc_1 = []
    chuncc_2 = []
    chuncc_3 = []
    chuncc_4 = []
    chuncc_5 = []
    chuncc_6 = []
    chuncc_7 = []
    chuncc_8 = []
    chuncc_9 = []
    chuncc_10 = []
    chuncc_11 = []
    chuncc_12 = []
    chuncc_13 = []
    chuncc_14 = []
    chuncc_15 = []
    chuncc_16 = []
    chuncc_17 = []
    chuncc_18 = []
    chuncc_19 = []

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

    with open("chunc/constraints/higgs_dm_lsp/constrained_output.csv","r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            chunc_1.append(float(row[0]))
            chunc_2.append(float(row[1]))
            chunc_3.append(float(row[2]))
            chunc_4.append(float(row[3]))
            chunc_5.append(float(row[4]))
            chunc_6.append(float(row[5]))
            chunc_7.append(float(row[6]))
            chunc_8.append(float(row[7]))
            chunc_9.append(float(row[8]))
            chunc_10.append(float(row[9]))
            chunc_11.append(float(row[10]))
            chunc_12.append(float(row[11]))
            chunc_13.append(float(row[12]))
            chunc_14.append(float(row[13]))
            chunc_15.append(float(row[14]))
            chunc_16.append(float(row[15]))
            chunc_17.append(float(row[16]))
            chunc_18.append(float(row[17]))
            chunc_19.append(float(row[18]))
    
    with open("chuncc/constraints/higgs_dm_lsp/constrained_output.csv","r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            chuncc_1.append(float(row[0]))
            chuncc_2.append(float(row[1]))
            chuncc_3.append(float(row[2]))
            chuncc_4.append(float(row[3]))
            chuncc_5.append(float(row[4]))
            chuncc_6.append(float(row[5]))
            chuncc_7.append(float(row[6]))
            chuncc_8.append(float(row[7]))
            chuncc_9.append(float(row[8]))
            chuncc_10.append(float(row[9]))
            chuncc_11.append(float(row[10]))
            chuncc_12.append(float(row[11]))
            chuncc_13.append(float(row[12]))
            chuncc_14.append(float(row[13]))
            chuncc_15.append(float(row[14]))
            chuncc_16.append(float(row[15]))
            chuncc_17.append(float(row[16]))
            chuncc_18.append(float(row[17]))
            chuncc_19.append(float(row[18]))
    
    with open("chunc/constraints/higgs_dm_lsp/constrained_data.csv","r") as file:
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

    

    fig, axs = plt.subplots(5,4,figsize=(11,8))
    line_labels = ["Valid training data", "Valid generated data (CHUNC)", "Valid generated data (CHUNC2)"]

    axs[0][0].hist(input_1, bins=25, label="input data", histtype='step', density=True, stacked=True, color='b')
    axs[0][0].hist(chunc_1, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[0][0].hist(chuncc_1, bins=25, label="chunc2 output", histtype='step', density=True, stacked=True, color='orange')
    axs[0][0].set_yticks([])
    axs[0][0].set_yticklabels([])
    axs[0][0].set_xlabel(r"$m_1$")

    axs[0][1].hist(input_2, bins=25, label="input data", histtype='step', density=True, stacked=True, color='b')
    axs[0][1].hist(chunc_2, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[0][1].hist(chuncc_2, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='orange')
    axs[0][1].set_yticks([])
    axs[0][1].set_yticklabels([])
    axs[0][1].set_xlabel(r"$m_2$")

    axs[0][2].hist(input_3, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[0][2].hist(chunc_3, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[0][2].hist(chuncc_3, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[0][2].set_yticks([])
    axs[0][2].set_yticklabels([])
    axs[0][2].set_xlabel(r"m_3")

    axs[0][3].hist(input_4, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[0][3].hist(chunc_4, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[0][3].hist(chuncc_4, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[0][3].set_yticks([])
    axs[0][3].set_yticklabels([])
    axs[0][3].set_xlabel(r"$m_{\mu}$")

    axs[1][0].hist(input_5, bins=25, label="input data", histtype='step', density=True, stacked=True, color='b')
    axs[1][0].hist(chunc_5, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[1][0].hist(chuncc_5, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='orange')
    axs[1][0].set_yticks([])
    axs[1][0].set_yticklabels([])
    axs[1][0].set_xlabel(r"$m_A$")

    axs[1][1].hist(input_6, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[1][1].hist(chunc_6, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[1][1].hist(chuncc_6, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[1][1].set_yticks([])
    axs[1][1].set_yticklabels([])
    axs[1][1].set_xlabel(r"$A_t$")

    axs[1][2].hist(input_7, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[1][2].hist(chunc_7, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[1][2].hist(chuncc_7, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[1][2].set_yticks([])
    axs[1][2].set_yticklabels([])
    axs[1][2].set_xlabel(r"$A_b$")

    axs[1][3].hist(input_8, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[1][3].hist(chunc_8, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[1][3].hist(chuncc_8, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[1][3].set_yticks([])
    axs[1][3].set_yticklabels([])
    axs[1][3].set_xlabel(r"$A_{\tau}$")

    axs[2][0].hist(input_9, bins=25, label="input data", histtype='step', density=True, stacked=True, color='b')
    axs[2][0].hist(chunc_9, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[2][0].hist(chuncc_9, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='orange')
    axs[2][0].set_yticks([])
    axs[2][0].set_yticklabels([])
    axs[2][0].set_xlabel(r"$m_{\tilde{L}_1}$")

    axs[2][1].hist(input_10, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[2][1].hist(chunc_10, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[2][1].hist(chuncc_10, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[2][1].set_yticks([])
    axs[2][1].set_yticklabels([])
    axs[2][1].set_xlabel(r"$m_{\tilde{L}_3}$")

    axs[2][2].hist(input_11, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[2][2].hist(chunc_11, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[2][2].hist(chuncc_11, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[2][2].set_yticks([])
    axs[2][2].set_yticklabels([])
    axs[2][2].set_xlabel(r"$m_{\tilde{e}_1}$")

    axs[2][3].hist(input_12, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[2][3].hist(chunc_12, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[2][3].hist(chuncc_12, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[2][3].set_yticks([])
    axs[2][3].set_yticklabels([])
    axs[2][3].set_xlabel(r"$m_{\tilde{e}_3}$")

    axs[3][0].hist(input_13, bins=25, label="input data", histtype='step', density=True, stacked=True, color='b')
    axs[3][0].hist(chunc_13, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[3][0].hist(chuncc_13, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='orange')
    axs[3][0].set_yticks([])
    axs[3][0].set_yticklabels([])
    axs[3][0].set_xlabel(r"$m_{\tilde{Q}_1}$")

    axs[3][1].hist(input_14, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[3][1].hist(chunc_14, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[3][1].hist(chuncc_14, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[3][1].set_yticks([])
    axs[3][1].set_yticklabels([])
    axs[3][1].set_xlabel(r"$m_{\tilde{Q}_3}$")

    axs[3][2].hist(input_15, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[3][2].hist(chunc_15, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[3][2].hist(chuncc_15, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[3][2].set_yticks([])
    axs[3][2].set_yticklabels([])
    axs[3][2].set_xlabel(r"m_{\tilde{u}_1}$")

    axs[3][3].hist(input_16, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[3][3].hist(chunc_16, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[3][3].hist(chuncc_16, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[3][3].set_yticks([])
    axs[3][3].set_yticklabels([])
    axs[3][3].set_xlabel(r"$m_{\tilde{u}_3}$")

    axs[4][0].hist(input_17, bins=25, label="input data", histtype='step', density=True, stacked=True, color='b')
    axs[4][0].hist(chunc_17, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='r')
    axs[4][0].hist(chuncc_17, bins=25, label="chunc output", histtype='step', density=True, stacked=True, color='orange')
    axs[4][0].set_yticks([])
    axs[4][0].set_yticklabels([])
    axs[4][0].set_xlabel(r"$m_{\tilde{d}_1}$")

    axs[4][1].hist(input_18, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[4][1].hist(chunc_18, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[4][1].hist(chuncc_18, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[4][1].set_yticks([])
    axs[4][1].set_yticklabels([])
    axs[4][1].set_xlabel(r"$m_{\tilde{d}_3}$")

    axs[4][2].hist(input_19, bins=25, histtype='step', density=True, stacked=True, color='b')
    axs[4][2].hist(chunc_19, bins=25, histtype='step', density=True, stacked=True, color='r')
    axs[4][2].hist(chuncc_19, bins=25, histtype='step', density=True, stacked=True, color='orange')
    axs[4][2].set_yticks([])
    axs[4][2].set_yticklabels([])
    axs[4][2].set_xlabel(r"$\tan \beta$")



    axs[4][3].set_visible(False)

    #axs[0][0].legend(loc='upper left')
    #plt.suptitle(r" Valid Points (Tophat kernel: $\lambda = $" + f"{0.0001})")
    plt.figlegend(labels=line_labels, loc='lower right', bbox_to_anchor=(1.0, 0.1))
    fig.supylabel("Counts (normalized)")
    plt.tight_layout()
    plt.show()

