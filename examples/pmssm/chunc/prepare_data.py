"""
Create datasets for CMSSM
"""
import os
from datetime import datetime

from chunc.dataset.parameters import *
from chunc.dataset.pmssm import pMSSMDataset

if __name__ == "__main__":

    """
    First, we create the dataset from the parameter files.
    Then, generate the constrained and unconstrained subspaces
    and save the labeled training set to numpy files.
    """
    apply_constraints = False
    dataset = pMSSMDataset(
        input_dir = '../../../pmssm/pmssm_random_new/',
    )
    if apply_constraints:
        dataset.generate_constrained_dataset(
            #max_num_files=1000,
            apply_dm=True,
            apply_higgs=True,
            apply_lsp=True
        )
        dataset.generate_unconstrained_dataset(
            max_num_files=100,
            apply_dm=True,
            apply_higgs=True,
            apply_lsp=True
        )
    dataset.generate_training_set(
        constrained_file    = 'constraints/higgs_dm_lsp/constrained_data.csv',
        unconstrained_file  = 'constraints/higgs_dm_lsp/unconstrained_data.csv',
        close_gap           = True,
        symmetric_events    = True,
        labeling    = 'binary',
        save        = True,
        output_file = 'pmssm_higgs_dm_lsp_symmetric_no_gap.npz'
    )