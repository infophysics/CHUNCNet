"""
Class for sampling from latent space distributions of CHUNC models
"""
import numpy as np
import torch
from sklearn.neighbors import KernelDensity

from chunc.utils.logger import Logger

class CHUNCSampler:
    """
    """
    def __init__(self,
        model,
        latent_variables:   list=[],
    ):
        self.name = model.name + "_sampler"
        self.logger = Logger(self.name, output='both', file_mode='w')
        self.logger.info(f"constructing model sampler.")
        self.model = model
        self.latent_variables = latent_variables
        self.device = model.device

    def sample_latent(self,
        num_samples:    int=100,
        sample_mean:    float=0.0,
        sample_sigma:   float=0.01
    ):
        valid_samples = np.random.normal(
            sample_mean,sample_sigma,
            size=(num_samples,len(self.latent_variables))
        )
        return torch.tensor(valid_samples, dtype=torch.float)

class CHUNCKDESampler:
    """
    """
    def __init__(self,
        model,
        latent_variables:   list=[],
        bandwidth:  float=1.0,
        kernel:     str='gaussian'
    ):
        self.name = model.name + "_sampler"
        self.logger = Logger(self.name, output='both', file_mode='w')
        self.logger.info(f"constructing model sampler.")
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.model = model
        self.latent_variables = latent_variables
        self.kder = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.device = model.device

    def sample_latent(self,
        loader,     
        num_samples:    int=100,
    ):
        inference_loader = loader.all_loader
        inference_loop = enumerate(inference_loader, 0)
        
        # set up array for latent
        latent = torch.empty(size=(0,len(self.latent_variables)), dtype=torch.float, device=self.device)
        targets = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        
        self.logger.info(f"running inference on dataset '{loader.dataset.name}'.")
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            for ii, data in inference_loop:
                # get the network output
                outputs = self.model(data)
                latent = torch.cat((latent, outputs[1][:,self.latent_variables]), dim=0)
                targets = torch.cat((targets,data[1].to(self.device)), dim=0)
        
        latent = latent.cpu()
        targets = targets.cpu()
        valid_mask = (targets == 1)
        
        kder = self.kder.fit(latent[valid_mask.flatten()])

        valid_samples = kder.sample(n_samples=num_samples)

        return torch.tensor(valid_samples, dtype=torch.float)

        
class CHUNCCSampler:
    """
    """
    def __init__(self,
        model,
        latent_variables:   list=[],
        binary_variable:    int=-1,
        num_latent_bins:    int=25,
        num_binary_bins:    int=10,
    ):
        self.name = model.name + "_sampler"
        self.logger = Logger(self.name, output='both', file_mode='w')
        self.logger.info(f"constructing model sampler.")
        self.model = model
        self.latent_variables = latent_variables
        self.binary_variable = binary_variable
        self.num_latent_bins = num_latent_bins
        self.num_binary_bins = num_binary_bins
        self.device = model.device

        # set up latent dictionaries
        self.latent_valid_hist = {}
        self.latent_invalid_hist = {}
        self.latent_binary_hist = {}
    
    def build_histograms(self,
        loader,
    ):
        inference_loader = loader.all_loader
        inference_loop = enumerate(inference_loader, 0)
        
        # set up array for latent
        latent = torch.empty(size=(0,len(self.latent_variables)), dtype=torch.float, device=self.device)
        binary = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        targets = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        
        self.logger.info(f"running inference on dataset '{loader.dataset.name}'.")
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            for ii, data in inference_loop:
                # get the network output
                outputs = self.model(data)
                latent = torch.cat((latent, outputs[1][:,self.latent_variables]), dim=0)
                binary = torch.cat((binary, outputs[1][:,self.binary_variable].unsqueeze(1)), dim=0)
                targets = torch.cat((targets,data[1].to(self.device)), dim=0)
        
        latent = latent.cpu()
        binary = binary.cpu()
        targets = targets.cpu()
        # create bins for binary variable
        self.latent_binary_hist['binary_hist'], self.latent_binary_hist['binary_edges'] = np.histogram(
            binary, bins=self.num_binary_bins
        )
        binary_masks = [
            (binary >= self.latent_binary_hist['binary_edges'][i]) & (binary < self.latent_binary_hist['binary_edges'][i+1])
            for i in range(len(self.latent_binary_hist['binary_hist']))
        ]
        # create bins for all latent variables
        for ii, mask in enumerate(binary_masks):
            valid_mask = (targets[mask] == 1)
            for jj in range(len(self.latent_variables)):
                temp_latent = latent[:,jj].unsqueeze(1)
                self.latent_valid_hist[f'{jj}_hist_{ii}'], self.latent_valid_hist[f'{jj}_bin_edges_{ii}'] = np.histogram(
                    temp_latent[mask][valid_mask], bins=self.num_latent_bins
                )
                self.latent_invalid_hist[f'{jj}_hist_{ii}'], self.latent_invalid_hist[f'{jj}_bin_edges_{ii}'] = np.histogram(
                    temp_latent[mask][~valid_mask], bins=self.num_latent_bins
                )
        for ii in range(self.num_binary_bins):
            for jj in range(len(self.latent_variables)):
                self.latent_valid_hist[f'{jj}_hist_{ii}'] = self.latent_valid_hist[f'{jj}_hist_{ii}'].astype(float)
                self.latent_invalid_hist[f'{jj}_hist_{ii}'] = self.latent_invalid_hist[f'{jj}_hist_{ii}'].astype(float)
                self.latent_valid_hist[f'{jj}_hist_{ii}'] /= sum(self.latent_valid_hist[f'{jj}_hist_{ii}'])
                self.latent_invalid_hist[f'{jj}_hist_{ii}'] /= sum(self.latent_invalid_hist[f'{jj}_hist_{ii}'])
                self.latent_valid_hist[f'{jj}_chist_{ii}'] = self.latent_valid_hist[f'{jj}_hist_{ii}']
                self.latent_invalid_hist[f'{jj}_chist_{ii}'] = self.latent_invalid_hist[f'{jj}_hist_{ii}']
                for kk in range(1,self.num_latent_bins):
                    self.latent_valid_hist[f'{jj}_chist_{ii}'][kk] += self.latent_valid_hist[f'{jj}_chist_{ii}'][kk-1]
                    self.latent_invalid_hist[f'{jj}_chist_{ii}'][kk] += self.latent_invalid_hist[f'{jj}_chist_{ii}'][kk-1]

    def sample_histograms(self,
        num_samples:    int=100,
        binary_bin:     int=0,
        sample_type:    str='valid',
    ):
        if binary_bin >= self.num_binary_bins:
            self.logger.error(f"specified binary bin: {binary_bin} is out of bounds!")
        # first get bin samples
        bin_samples = []
        sample_probs = np.random.uniform(0,1.0,num_samples)
        for ii in range(len(sample_probs)):
            bin_samples.append(np.random.uniform(
                low=self.latent_binary_hist['binary_edges'][binary_bin],
                high=self.latent_binary_hist['binary_edges'][binary_bin+1],
                size=1
            )[0])

        # now generate valid samples
        valid_samples = []
        if sample_type == 'valid' or sample_type == 'all':
            for jj in range(len(self.latent_variables)):
                latent_samples = []
                sample_probs = np.random.uniform(0,1.0,num_samples)
                for ii in range(len(sample_probs)):
                    for kk in range(self.num_latent_bins):
                        if self.latent_valid_hist[f'{jj}_chist_{binary_bin}'][kk] >= sample_probs[ii]:
                            latent_samples.append(np.random.uniform(
                                low=self.latent_valid_hist[f'{jj}_bin_edges_{binary_bin}'][kk],
                                high=self.latent_valid_hist[f'{jj}_bin_edges_{binary_bin}'][kk+1],
                                size=1
                            )[0])
                            break
                valid_samples.append(latent_samples)
            valid_samples.append(bin_samples)
            valid_samples = np.vstack(valid_samples).T
            if sample_type == 'valid':
                return torch.tensor(valid_samples, dtype=torch.float)

        # generate invalid samples
        invalid_samples = []
        if sample_type == 'invalid' or sample_type == 'all':
            for jj in range(len(self.latent_variables)):
                latent_samples = []
                sample_probs = np.random.uniform(0,1.0,num_samples)
                for ii in range(len(sample_probs)):
                    for kk in range(self.num_latent_bins):
                        if self.latent_invalid_hist[f'{jj}_chist_{binary_bin}'][kk] >= sample_probs[ii]:
                            latent_samples.append(np.random.uniform(
                                low=self.latent_invalid_hist[f'{jj}_bin_edges_{binary_bin}'][kk],
                                high=self.latent_invalid_hist[f'{jj}_bin_edges_{binary_bin}'][kk+1],
                                size=1
                            )[0])
                            break
                invalid_samples.append(latent_samples)
            invalid_samples.append(bin_samples)
            invalid_samples = np.vstack(invalid_samples).T
            if sample_type == 'invalid':    
                return torch.tensor(invalid_samples, dtype=torch.float)
        if sample_type == 'all':
            return torch.tensor(np.concatenate((valid_samples, invalid_samples)), dtype=torch.float)
    
    def sample_kde(self,
        num_samples:    int=100,
        binary_bin:     int=0,
        sample_type:    str='valid',
    ):
        if binary_bin >= self.num_binary_bins:
            self.logger.error(f"specified binary bin: {binary_bin} is out of bounds!")
        # first get bin samples
        bin_samples = []
        sample_probs = np.random.uniform(0,1.0,num_samples)
        for ii in range(len(sample_probs)):
            bin_samples.append(np.random.uniform(
                low=self.latent_binary_hist['binary_edges'][binary_bin],
                high=self.latent_binary_hist['binary_edges'][binary_bin+1],
                size=1
            )[0])
    
    def sample_latent(self,
        loader,     
        num_samples:    int=100,
        binary_bin: int=0,
    ):
        inference_loader = loader.all_loader
        inference_loop = enumerate(inference_loader, 0)
        
        # set up array for latent
        latent = torch.empty(size=(0,len(self.latent_variables)), dtype=torch.float, device=self.device)
        targets = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        
        self.logger.info(f"running inference on dataset '{loader.dataset.name}'.")
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            for ii, data in inference_loop:
                # get the network output
                outputs = self.model(data)
                latent = torch.cat((latent, outputs[1][:,self.latent_variables]), dim=0)
                targets = torch.cat((targets,data[1].to(self.device)), dim=0)
        
        latent = latent.cpu()
        targets = targets.cpu()
        valid_mask = (targets == 1)
        
        kder = self.kder.fit(latent[valid_mask.flatten()])

        valid_samples = kder.sample(n_samples=num_samples)

        return torch.tensor(valid_samples, dtype=torch.float)

class CHUNCCKDESampler:
    """
    """
    def __init__(self,
        model,
        latent_variables:   list=[],
        binary_variable:    int=-1,
        num_latent_bins:    int=25,
        num_binary_bins:    int=10,
        bandwidth:      float=0.01,
        kernel:     str='gaussian',
        method:     str='bin',
    ):
        self.name = model.name + "_sampler"
        self.logger = Logger(self.name, output='both', file_mode='w')
        self.logger.info(f"constructing model sampler.")
        self.model = model
        self.latent_variables = latent_variables
        self.binary_variable = binary_variable
        self.num_latent_bins = num_latent_bins
        self.num_binary_bins = num_binary_bins
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.method = method
        self.kder = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.device = model.device

        # set up latent dictionaries
        self.latent_valid_hist = {}
        self.latent_invalid_hist = {}
        self.latent_binary_hist = {}
    
    def build_histograms(self,
        loader,
    ):
        inference_loader = loader.all_loader
        inference_loop = enumerate(inference_loader, 0)
        
        # set up array for latent
        latent = torch.empty(size=(0,len(self.latent_variables)), dtype=torch.float, device=self.device)
        binary = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        targets = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        
        self.logger.info(f"running inference on dataset '{loader.dataset.name}'.")
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            for ii, data in inference_loop:
                # get the network output
                outputs = self.model(data)
                latent = torch.cat((latent, outputs[1][:,self.latent_variables]), dim=0)
                binary = torch.cat((binary, outputs[1][:,self.binary_variable].unsqueeze(1)), dim=0)
                targets = torch.cat((targets,data[1].to(self.device)), dim=0)
        
        latent = latent.cpu()
        binary = binary.cpu()
        targets = targets.cpu()
        # create bins for binary variable
        self.latent_binary_hist['binary_hist'], self.latent_binary_hist['binary_edges'] = np.histogram(
            binary, bins=self.num_binary_bins
        )
        binary_masks = [
            (binary >= self.latent_binary_hist['binary_edges'][i]) & (binary < self.latent_binary_hist['binary_edges'][i+1])
            for i in range(len(self.latent_binary_hist['binary_hist']))
        ]
        # create bins for all latent variables
        for ii, mask in enumerate(binary_masks):
            valid_mask = (targets[mask] == 1)
            for jj in range(len(self.latent_variables)):
                temp_latent = latent[:,jj].unsqueeze(1)
                self.latent_valid_hist[f'{jj}_hist_{ii}'], self.latent_valid_hist[f'{jj}_bin_edges_{ii}'] = np.histogram(
                    temp_latent[mask][valid_mask], bins=self.num_latent_bins
                )
                self.latent_invalid_hist[f'{jj}_hist_{ii}'], self.latent_invalid_hist[f'{jj}_bin_edges_{ii}'] = np.histogram(
                    temp_latent[mask][~valid_mask], bins=self.num_latent_bins
                )
        for ii in range(self.num_binary_bins):
            for jj in range(len(self.latent_variables)):
                self.latent_valid_hist[f'{jj}_hist_{ii}'] = self.latent_valid_hist[f'{jj}_hist_{ii}'].astype(float)
                self.latent_invalid_hist[f'{jj}_hist_{ii}'] = self.latent_invalid_hist[f'{jj}_hist_{ii}'].astype(float)
                self.latent_valid_hist[f'{jj}_hist_{ii}'] /= sum(self.latent_valid_hist[f'{jj}_hist_{ii}'])
                self.latent_invalid_hist[f'{jj}_hist_{ii}'] /= sum(self.latent_invalid_hist[f'{jj}_hist_{ii}'])
                self.latent_valid_hist[f'{jj}_chist_{ii}'] = self.latent_valid_hist[f'{jj}_hist_{ii}']
                self.latent_invalid_hist[f'{jj}_chist_{ii}'] = self.latent_invalid_hist[f'{jj}_hist_{ii}']
                for kk in range(1,self.num_latent_bins):
                    self.latent_valid_hist[f'{jj}_chist_{ii}'][kk] += self.latent_valid_hist[f'{jj}_chist_{ii}'][kk-1]
                    self.latent_invalid_hist[f'{jj}_chist_{ii}'][kk] += self.latent_invalid_hist[f'{jj}_chist_{ii}'][kk-1]

    def sample_histograms(self,
        loader,     
        num_samples:    int=100,
        binary_bin: int=0,
    ):
        if binary_bin >= self.num_binary_bins:
            self.logger.error(f"specified binary bin: {binary_bin} is out of bounds!")
        low=self.latent_binary_hist['binary_edges'][binary_bin]
        high=self.latent_binary_hist['binary_edges'][binary_bin+1]
        inference_loader = loader.all_loader
        inference_loop = enumerate(inference_loader, 0)
        
        # set up array for latent
        latent = torch.empty(size=(0,len(self.latent_variables)+1), dtype=torch.float, device=self.device)
        targets = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        
        self.logger.info(f"running inference on dataset '{loader.dataset.name}'.")
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            for ii, data in inference_loop:
                # get the network output
                outputs = self.model(data)
                if self.method == 'bin':
                    latent = torch.cat(
                    (
                        latent, 
                        outputs[1][
                            (outputs[1][:,self.binary_variable] >= low) &
                            (outputs[1][:,self.binary_variable] <= high)
                        ]), 
                        dim=0
                    )
                    targets = torch.cat(
                        (
                            targets,
                            data[1].to(self.device)[
                                (outputs[1][:,self.binary_variable] >= low) &
                                (outputs[1][:,self.binary_variable] <= high)
                            ]
                        ), 
                        dim=0
                    )
                else:
                    latent = torch.cat((latent, outputs[1]), dim=0)
                    targets = torch.cat((targets,data[1].to(self.device)), dim=0)
        
        latent = latent.cpu()
        targets = targets.cpu()
        valid_mask = (targets == 1)
        
        kder = self.kder.fit(latent[valid_mask.flatten()])

        valid_samples = kder.sample(n_samples=num_samples)

        return torch.tensor(valid_samples, dtype=torch.float)

        # if binary_bin >= self.num_binary_bins:
        #     self.logger.error(f"specified binary bin: {binary_bin} is out of bounds!")
        # low=self.latent_binary_hist['binary_edges'][binary_bin]
        # high=self.latent_binary_hist['binary_edges'][binary_bin+1]

        # inference_loader = loader.all_loader
        # inference_loop = enumerate(inference_loader, 0)
        
        # # set up array for latent
        # latent = torch.empty(size=(0,len(self.latent_variables)), dtype=torch.float, device=self.device)
        # binary = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        # targets = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
        
        # self.logger.info(f"running inference on dataset '{loader.dataset.name}'.")
        # # make sure to set model to eval() during validation!
        # self.model.eval()


        # with torch.no_grad():
        #     for ii, data in inference_loop:
        #         # get the network output
        #         outputs = self.model(data)
        #         latent = torch.cat(
        #             (
        #                 latent, 
        #                 outputs[1][:,self.latent_variables][
        #                     (outputs[1][:,self.binary_variable] >= low) &
        #                     (outputs[1][:,self.binary_variable] <= high)
        #                 ]), 
        #             dim=0
        #         )
        #         binary = torch.cat(
        #             (
        #                 binary, 
        #                 outputs[1][:,self.binary_variable][
        #                     (outputs[1][:,self.binary_variable] >= low) &
        #                     (outputs[1][:,self.binary_variable] <= high)
        #                 ]), 
        #             dim=0
        #         )
        #         targets = torch.cat((targets,data[1].to(self.device)), dim=0)
        
        # latent = latent.cpu()
        # targets = targets.cpu()
        # valid_mask = (targets == 1)
        
        # kder = self.kder.fit(latent[valid_mask.flatten()])

        # valid_samples = kder.sample(n_samples=num_samples)

        # return torch.tensor(valid_samples, dtype=torch.float)