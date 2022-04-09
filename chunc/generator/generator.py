"""
Code for generating new samples from a model
and running them through the running coupling code
"""
import csv
import os
from chunc.dataset.cmssm import cMSSMDataset
from chunc.dataset.pmssm import pMSSMDataset
from chunc.utils.loader import Loader
from chunc.utils.logger import Logger
from chunc.utils.mssm import MSSMGenerator
from chunc.sampler import Sampler

chuncc_generator_config = {
    'sampler':      Sampler,
    'mssm_generator':MSSMGenerator,
    'subspace':     'cmssm',
    'num_events':   1000,
    'num_workers':  16,
    'binary_bin':   9,
}

class Generator:
    """
    """
    def __init__(self,
        cfg=chuncc_generator_config,
    ):
        self.cfg = cfg
        self.logger = Logger(name='generator', file_mode='w')

        if not os.path.isdir('mssm_input/'):
            os.makdirs('mssm_input/')
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
        )
        if self.cfg['subspace'] == 'cmssm':
            dataset = cMSSMDataset(
                input_dir = 'mssm_output/',
            )
        else:
            dataset = pMSSMDataset(
                input_dir = 'mssm_output/'
            )
        self.num_valid.append(
            dataset.get_number_of_valid(
                apply_dm=True,
                apply_higgs=True,
                apply_lsp=True,
        ))
        dataset.generate_constrained_dataset(
            output_file=f"iterative_constrained_{iteration}"
        )
        dataset.generate_unconstrained_dataset(
            output_file=f"iterative_unconstrained_{iteration}"
        )
        # """Add the new events to the constrained and unconstrained files"""
        # constrained_files = [f"constraints/higgs_dm_lsp/iterative_constrained_{jj}.csv" for jj in range(iteration+1)]
        # constrained_files.append('constraints/higgs_dm_lsp/constrained_data.csv')
        # unconstrained_files = [f"constraints/higgs_dm_lsp/iterative_unconstrained_{jj}.csv" for jj in range(iteration+1)]
        # unconstrained_files.append('constraints/higgs_dm_lsp/unconstrained_data.csv')

        # concatenate_csv(constrained_files, 'constraints/higgs_dm_lsp/constrained_data.csv')
        # concatenate_csv(unconstrained_files, 'constraints/higgs_dm_lsp/unconstrained_data.csv')
        # """Reload training set"""
        # dataset.generate_training_set(
        #     constrained_file    = 'constraints/higgs_dm_lsp/constrained_data.csv',
        #     unconstrained_file  = 'constraints/higgs_dm_lsp/unconstrained_data.csv',
        #     symmetric_events    = True,
        #     labeling    = 'binary',
        #     save        = True,
        #     output_file = 'cmssm_dataset_symmetric.npz'
        # )
        # dataset = CHUNCDataset(
        #     name="dataset",
        #     input_file='datasets/cmssm_dataset_symmetric.npz',
        #     features = features,
        #     classes = ['valid']
        # )
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