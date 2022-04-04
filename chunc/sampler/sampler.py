"""
Class for sampling from latent space distributions of CHUNC models
"""
import numpy as np
import torch

from chunc.utils.logger import Logger

class Sampler:
    """
    """
    def __init__(self,
        model,
        latent_variables:   list=[],
        binary_variable:    int=-1,
    ):
        self.name = model.name + "_sampler"
        self.logger = Logger(self.name, output='both', file_mode='w')
        self.logger.info(f"constructing model sampler.")
        self.model = model
        self.latent_variables = latent_variables
        self.binary_variable = binary_variable

        # set up latent dictionaries
        self.latent_valid_hist = {}
        self.latent_invalid_hist = {}
        self.latent_binary_hist = {}
    
    def build_histograms(self,
        loader,
        num_latent_bins:    int=25,
        num_binary_bins:    int=10,
    ):
        inference_loader = loader.all_loader
        inference_loop = enumerate(inference_loader, 0)
        
        # set up array for latent
        latent = [
            torch.empty(size=(0,1), dtype=torch.float)
            for ii in range(len(self.latent_variables))
        ]
        binary = torch.empty(size=(0,1), dtype=torch.float)
        targets = torch.empty(size=(0,1), dtype=torch.float)
        
        self.logger.info(f"running inference on dataset '{loader.dataset.name}'.")
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            for ii, data in inference_loop:
                # get the network output
                outputs = self.model(data)

                latent = [
                    torch.cat((latent[jj], outputs[1][jj]), dim=0)
                    for jj in range(len(self.latent_variables))
                ]
                binary = torch.cat((binary, outputs[1][self.binary_variable]), dim=0)
                targets = torch.cat((targets,data[1]), dim=0)
        
        # create bins for binary variable
        binary_hist, binary_edges = np.histogram(binary, bins=num_binary_bins)

        # create bins for all latent variables

    def sample_histograms(self,
        num_samples:    int=100,
    ):
        pass

