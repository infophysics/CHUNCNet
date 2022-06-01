"""
Code for generating new samples from a model
and running them through the running coupling code
"""
import csv
import os
from matplotlib import pyplot as plt
from chunc.dataset.cmssm import cMSSMDataset
from chunc.dataset.pmssm import pMSSMDataset
from chunc.dataset.chunc import CHUNCDataset
from chunc.utils.loader import Loader
from chunc.utils.logger import Logger
from chunc.utils.mssm import MSSMGenerator
from chunc.sampler import CHUNCSampler, CHUNCCSampler
from chunc.utils import utils

chunc_generator_config = {
    'loader':       Loader,
    'sampler':      CHUNCSampler,
    'mssm_generator':MSSMGenerator,
    'subspace':     'cmssm',
    'num_events':   1000,
    'num_workers':  16,
    'sample_mean':  0.0,
    'sample_sigma': 0.01,
    'variables':    [],
}

chuncc_generator_config = {
    'loader':       Loader,
    'sampler':      CHUNCCSampler,
    'mssm_generator':MSSMGenerator,
    'subspace':     'cmssm',
    'num_events':   1000,
    'num_workers':  16,
    'binary_bin':   9,
    'variables':    [],
}

class CHUNCGenerator:
    """
    """
    def __init__(self,
        cfg=chunc_generator_config,
    ):
        self.cfg = cfg
        self.logger = Logger(name='generator', file_mode='w')

        if not os.path.isdir('mssm_input/'):
            os.makedirs('mssm_input/')
        self.device = None
        self.gut_m1_gap = 50.0
        self.gut_m2_gap = 100.0
        self.gut_mmu_gap = 100.0
        
        self.constraints = ['higgs','dm','higgs_dm','higgs_dm_lsp']
        self.num_valid = []
        self.num_invalid = []
 
    def set_device(self,
        device, 
    ):
        self.device = device
        self.cfg['sampler'].device = device
    
    def generate(self,
        model,
        loader,
        mean=-999,
        sigma=-999,
        iteration=-1,
        plot_samples=True,
    ):
        if (mean == -999) and (sigma == -999):
            mean = self.cfg['sample_mean']
            sigma = self.cfg['sample_sigma']
        valid_samples = self.cfg['sampler'].sample_latent(
            loader, 
            self.cfg['num_events']
        )
        # valid_samples = self.cfg['sampler'].sample_latent(
        #     self.cfg['num_events'],
        #     mean,
        #     sigma,
        # )
        """Generate the samples from the model"""
        generated_samples = model.sample(valid_samples).cpu()
        generated_samples = loader.dataset.unnormalize(generated_samples, detach=True)
        if self.cfg['subspace'] == 'cmssm':
            for ii, sample in enumerate(generated_samples):
                if sample[4] < 0:
                    generated_samples[ii][4] = -1
                else:
                    generated_samples[ii][4] = 1
        else:
            if loader.dataset.meta['close_gap']:
                for ii, sample in enumerate(generated_samples):
                    if sample[0] < 0:
                        generated_samples[ii][0] -= self.gut_m1_gap
                    else:
                        generated_samples[ii][0] += self.gut_m1_gap
                    if sample[1] < 0:
                        generated_samples[ii][1] -= self.gut_m2_gap
                    else:
                        generated_samples[ii][1] += self.gut_m2_gap
                    if sample[3] < 0:
                        generated_samples[ii][3] -= self.gut_mmu_gap
                    else:
                        generated_samples[ii][3] += self.gut_mmu_gap
        if plot_samples:
            if not os.path.isdir(f"plots/samples_{mean}_{sigma}/"):
                os.makedirs(f"plots/samples_{mean}_{sigma}/")
            training_input = loader.dataset.event_features
            # plot all variable inputs and outputs
            fig, axs = utils.generate_plot_grid(
                len(self.cfg['variables']),
                figsize=(10, 6)
            )
            for ii in range(len(self.cfg['variables'])):
                axs.flat[ii].hist(
                    generated_samples[:,ii], 
                    bins=25, 
                    label='samples', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    training_input[:,ii].cpu().numpy(), 
                    bins=25, 
                    label='input_training', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f"{self.cfg['variables'][ii]}")
            axs.flat[0].legend()
            plt.suptitle(f"CHUNC training input/sample variables (mean: {mean}, sigma: {sigma})")
            plt.tight_layout()
            plt.savefig(f"plots/samples_{mean}_{sigma}/training_sample_variables_{iteration}.png")

        with open(f"mssm_input/samples_{mean}_{sigma}_{iteration}.txt", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(generated_samples)
        # run the parameters and dump the output
        # to the standard folder mssm_output.
        self.cfg['mssm_generator'].run_parameters(
            input_file=f"mssm_input/samples_{mean}_{sigma}_{iteration}.txt",
            num_events=self.cfg['num_events'],
            num_workers=self.cfg['num_workers'],
            save_super_invalid=False,
            output_flag=f"{mean}_{sigma}_{iteration}"
        )
        
    def check_validities(self):
        if self.cfg['subspace'] == 'cmssm':
            mssm_dataset = cMSSMDataset(
                input_dir = 'mssm_output/',
            )
        else:
            mssm_dataset = pMSSMDataset(
                input_dir = 'mssm_output/'
            )
        validities = [
            mssm_dataset.get_number_of_valid(
                apply_dm=False,
                apply_higgs=True,
                apply_lsp=False,
            ),
            mssm_dataset.get_number_of_valid(
                apply_dm=True,
                apply_higgs=False,
                apply_lsp=False,
            ),
            mssm_dataset.get_number_of_valid(
                apply_dm=True,
                apply_higgs=True,
                apply_lsp=False,
            ),
            mssm_dataset.get_number_of_valid(
                apply_dm=True,
                apply_higgs=True,
                apply_lsp=True,
            ),
        ]
        self.num_valid.append(validities)
        return validities

    def generate_iterative(self,
        iteration,
        model,
        loader,
    ):
        valid_samples = self.cfg['sampler'].sample_latent(
            self.cfg['num_events'],
            self.cfg['sample_mean'],
            self.cfg['sample_sigma']
        )

        """Generate the samples from the model"""
        generated_samples = model.sample(valid_samples).cpu()
        generated_samples = loader.dataset.unnormalize(generated_samples, detach=True)
        if self.cfg['subspace'] == 'cmssm':
            for ii, sample in enumerate(generated_samples):
                if sample[4] < 0:
                    generated_samples[ii][4] = -1
                else:
                    generated_samples[ii][4] = 1
        with open(f"mssm_input/samples_{iteration}.txt", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(generated_samples)
        # run the parameters and dump the output
        # to the standard folder mssm_output.
        self.cfg['mssm_generator'].run_parameters(
            input_file=f"mssm_input/samples_{iteration}.txt",
            num_events=self.cfg['num_events'],
            num_workers=self.cfg['num_workers'],
            save_super_invalid=False,
            output_flag=f"{iteration}"
        )
        if self.cfg['subspace'] == 'cmssm':
            mssm_dataset = cMSSMDataset(
                input_dir = 'mssm_output/',
            )
        else:
            mssm_dataset = pMSSMDataset(
                input_dir = 'mssm_output/'
            )
        self.num_valid.append(
            mssm_dataset.get_number_of_valid(
                apply_dm=True,
                apply_higgs=True,
                apply_lsp=True,
        ))
        mssm_dataset.generate_constrained_dataset(
            output_file=f"iterative_constrained_{iteration}",
            input_file=f"{self.cfg['subspace']}_generated_{iteration}.txt"
        )
        mssm_dataset.generate_unconstrained_dataset(
            output_file=f"iterative_unconstrained_{iteration}",
            input_file=f"{self.cfg['subspace']}_generated_{iteration}.txt"
        )

        # """Add the new events to the constrained and unconstrained files"""
        new_constrained = []
        with open(f"constraints/higgs_dm_lsp/iterative_constrained_{iteration}.csv", "r") as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                new_constrained.append([r for r in row])
        with open('constraints/higgs_dm_lsp/constrained_data.csv', "a") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(new_constrained)

        new_unconstrained = []
        with open(f"constraints/higgs_dm_lsp/iterative_unconstrained_{iteration}.csv", "r") as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                new_unconstrained.append([r for r in row])
        with open('constraints/higgs_dm_lsp/unconstrained_data.csv', "a") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(new_unconstrained)

        # """Reload training set"""
        mssm_dataset.generate_training_set(
            constrained_file    = 'constraints/higgs_dm_lsp/constrained_data.csv',
            unconstrained_file  = 'constraints/higgs_dm_lsp/unconstrained_data.csv',
            symmetric_events    = True,
            labeling    = 'binary',
            save        = True,
            output_file = 'cmssm_dataset_symmetric.npz'
        )
        dataset = CHUNCDataset(
            name=f"dataset_{iteration}",
            input_file='datasets/cmssm_dataset_symmetric.npz',
            features = self.cfg['loader'].dataset.features,
            classes = self.cfg['loader'].dataset.classes
        )
        loader = Loader(
            dataset, 
            batch_size=64,
            test_split=0.3,
            test_seed=100,
            validation_split=0.3,
            validation_seed=100,
            num_workers=4
        )
        return loader

    

class CHUNCCGenerator:
    """
    """
    def __init__(self,
        cfg=chuncc_generator_config,
    ):
        self.cfg = cfg
        self.logger = Logger(name='generator', file_mode='w')

        if not os.path.isdir('mssm_input/'):
            os.makedirs('mssm_input/')
        self.device = None
        self.gut_m1_gap = 50.0
        self.gut_m2_gap = 100.0
        self.gut_mmu_gap = 100.0
        
        self.num_valid = []
        self.num_invalid = []
    
    def set_device(self,
        device, 
    ):
        self.device = device
        self.cfg['sampler'].device = device

    def generate(self,
        model,
        loader,
        iteration=-1,
        plot_samples=True,
    ):
        self.cfg['sampler'].build_histograms(
            loader
        )
        valid_samples = self.cfg['sampler'].sample_histograms(
            num_samples=self.cfg['num_events'],
            binary_bin=self.cfg['binary_bin'],
            sample_type='valid'
        )
        """Generate the samples from the model"""
        generated_samples = model.sample(valid_samples).cpu()
        generated_samples = loader.dataset.unnormalize(generated_samples, detach=True)
        if self.cfg['subspace'] == 'cmssm':
            for ii, sample in enumerate(generated_samples):
                if sample[4] < 0:
                    generated_samples[ii][4] = -1
                else:
                    generated_samples[ii][4] = 1
        else:
            if loader.dataset.meta['close_gap']:
                for ii, sample in enumerate(generated_samples):
                    if sample[0] < 0:
                        generated_samples[ii][0] -= self.gut_m1_gap
                    else:
                        generated_samples[ii][0] += self.gut_m1_gap
                    if sample[1] < 0:
                        generated_samples[ii][1] -= self.gut_m2_gap
                    else:
                        generated_samples[ii][1] += self.gut_m2_gap
                    if sample[3] < 0:
                        generated_samples[ii][3] -= self.gut_mmu_gap
                    else:
                        generated_samples[ii][3] += self.gut_mmu_gap
        if plot_samples:
            if not os.path.isdir(f"plots/samples_{iteration}/"):
                os.makedirs(f"plots/samples_{iteration}/")
            training_input = loader.dataset.event_features
            # plot all variable inputs and outputs
            fig, axs = utils.generate_plot_grid(
                len(self.cfg['variables']),
                figsize=(10, 6)
            )
            for ii in range(len(self.cfg['variables'])):
                axs.flat[ii].hist(
                    generated_samples[:,ii], 
                    bins=25, 
                    label='samples', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].hist(
                    training_input[:,ii].cpu().numpy(), 
                    bins=25, 
                    label='input_training', 
                    histtype='step', 
                    stacked=True,
                    density=True
                )
                axs.flat[ii].set_xlabel(f"{self.cfg['variables'][ii]}")
            axs.flat[0].legend()
            plt.suptitle(f"CHUNCC training input/sample variables")
            plt.tight_layout()
            plt.savefig(f"plots/samples_{iteration}/training_sample_variables_{iteration}.png")

        with open(f"mssm_input/samples_{iteration}_{iteration}.txt", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(generated_samples)
        # run the parameters and dump the output
        # to the standard folder mssm_output.
        self.cfg['mssm_generator'].run_parameters(
            input_file=f"mssm_input/samples_{iteration}_{iteration}.txt",
            num_events=self.cfg['num_events'],
            num_workers=self.cfg['num_workers'],
            save_super_invalid=False,
            output_flag=f"{iteration}"
        )

    def generate_iterative(self,
        iteration,
        model,
        loader,
    ):
        self.cfg['sampler'].build_histograms(
            loader
        )
        valid_samples = self.cfg['sampler'].sample_histograms(
            num_samples=self.cfg['num_events'],
            binary_bin=self.cfg['binary_bin'],
            sample_type='valid'
        )

        """Generate the samples from the model"""
        generated_samples = model.sample(valid_samples).cpu()
        generated_samples = loader.dataset.unnormalize(generated_samples, detach=True)
        if self.cfg['subspace'] == 'cmssm':
            for ii, sample in enumerate(generated_samples):
                if sample[4] < 0:
                    generated_samples[ii][4] = -1
                else:
                    generated_samples[ii][4] = 1
        with open(f"mssm_input/samples_{iteration}.txt", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(generated_samples)
        # run the parameters and dump the output
        # to the standard folder mssm_output.
        self.cfg['mssm_generator'].run_parameters(
            input_file=f"mssm_input/samples_{iteration}.txt",
            num_events=self.cfg['num_events'],
            num_workers=self.cfg['num_workers'],
            save_super_invalid=False,
            output_flag=f"{iteration}"
        )
        if self.cfg['subspace'] == 'cmssm':
            mssm_dataset = cMSSMDataset(
                input_dir = 'mssm_output/',
            )
        else:
            mssm_dataset = pMSSMDataset(
                input_dir = 'mssm_output/'
            )
        self.num_valid.append(
            mssm_dataset.get_number_of_valid(
                apply_dm=True,
                apply_higgs=True,
                apply_lsp=True,
        ))
        mssm_dataset.generate_constrained_dataset(
            output_file=f"iterative_constrained_{iteration}",
            input_file=f"{self.cfg['subspace']}_generated_{iteration}.txt"
        )
        mssm_dataset.generate_unconstrained_dataset(
            output_file=f"iterative_unconstrained_{iteration}",
            input_file=f"{self.cfg['subspace']}_generated_{iteration}.txt"
        )

        # """Add the new events to the constrained and unconstrained files"""
        new_constrained = []
        with open(f"constraints/higgs_dm_lsp/iterative_constrained_{iteration}.csv", "r") as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                new_constrained.append([r for r in row])
        with open('constraints/higgs_dm_lsp/constrained_data.csv', "a") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(new_constrained)

        new_unconstrained = []
        with open(f"constraints/higgs_dm_lsp/iterative_unconstrained_{iteration}.csv", "r") as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                new_unconstrained.append([r for r in row])
        with open('constraints/higgs_dm_lsp/unconstrained_data.csv', "a") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(new_unconstrained)

        # """Reload training set"""
        mssm_dataset.generate_training_set(
            constrained_file    = 'constraints/higgs_dm_lsp/constrained_data.csv',
            unconstrained_file  = 'constraints/higgs_dm_lsp/unconstrained_data.csv',
            symmetric_events    = True,
            labeling    = 'binary',
            save        = True,
            output_file = 'cmssm_dataset_symmetric.npz'
        )
        dataset = CHUNCDataset(
            name=f"dataset_{iteration}",
            input_file='datasets/cmssm_dataset_symmetric.npz',
            features = self.cfg['loader'].dataset.features,
            classes = self.cfg['loader'].dataset.classes
        )
        loader = Loader(
            dataset, 
            batch_size=64,
            test_split=0.3,
            test_seed=100,
            validation_split=0.3,
            validation_seed=100,
            num_workers=4
        )
        return loader
    
    def check_validities(self):
        if self.cfg['subspace'] == 'cmssm':
            mssm_dataset = cMSSMDataset(
                input_dir = 'mssm_output/',
            )
        else:
            mssm_dataset = pMSSMDataset(
                input_dir = 'mssm_output/'
            )
        validities = [
            mssm_dataset.get_number_of_valid(
                apply_dm=False,
                apply_higgs=True,
                apply_lsp=False,
            ),
            mssm_dataset.get_number_of_valid(
                apply_dm=True,
                apply_higgs=False,
                apply_lsp=False,
            ),
            mssm_dataset.get_number_of_valid(
                apply_dm=True,
                apply_higgs=True,
                apply_lsp=False,
            ),
            mssm_dataset.get_number_of_valid(
                apply_dm=True,
                apply_higgs=True,
                apply_lsp=True,
            ),
        ]
        self.num_valid.append(validities)
        return validities