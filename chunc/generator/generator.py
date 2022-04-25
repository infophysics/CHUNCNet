"""
Code for generating new samples from a model
and running them through the running coupling code
"""
import csv
import os
from chunc.dataset.cmssm import cMSSMDataset
from chunc.dataset.pmssm import pMSSMDataset
from chunc.dataset.chunc import CHUNCDataset
from chunc.utils.loader import Loader
from chunc.utils.logger import Logger
from chunc.utils.mssm import MSSMGenerator
from chunc.sampler import CHUNCSampler, CHUNCCSampler

chunc_generator_config = {
    'loader':       Loader,
    'sampler':      CHUNCSampler,
    'mssm_generator':MSSMGenerator,
    'subspace':     'cmssm',
    'num_events':   1000,
    'num_workers':  16,
    'sample_mean':  0.0,
    'sample_sigma': 0.01,
}

chuncc_generator_config = {
    'loader':       Loader,
    'sampler':      CHUNCCSampler,
    'mssm_generator':MSSMGenerator,
    'subspace':     'cmssm',
    'num_events':   1000,
    'num_workers':  16,
    'binary_bin':   9,
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
        
        self.num_valid = []
        self.num_invalid = []
    
    def set_device(self,
        device, 
    ):
        self.device = device
        self.cfg['sampler'].device = device

    def generate(self,
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
        
        self.num_valid = []
        self.num_invalid = []
    
    def set_device(self,
        device, 
    ):
        self.device = device
        self.cfg['sampler'].device = device

    def generate(self,
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