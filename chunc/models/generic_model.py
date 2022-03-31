"""
Generic model code.
"""
import torch
import os
import csv
import getpass
from torch import nn
from time import time
from datetime import datetime

from chunc.utils.logger import Logger

generic_config = {
    "no_params":    "no_values"
}

class GenericModel(nn.Module):
    """
    Wrapper of torch nn.Module that generates a GenericModel
    """
    def __init__(self,
        name:   str,
        cfg:    dict=generic_config,
    ):
        super(GenericModel, self).__init__()
        self.name = name
        self.cfg = cfg
        self.logger = Logger(self.name, file_mode='w')
        self.logger.info(f"configuring model.")
        # now define the model
        self.forward_views      = {}
        self.forward_view_map   = {}
        self.input_shape = None
        self.output_shape = None
        # device for the model
        self.device = None

    def set_device(self,
        device
    ):
        self.device = device
        self.to(device)

    def forward_hook(self, m, i, o):
        """
        A forward hook for a particular module.
        It assigns the output to the views dictionary.
        """
        self.forward_views[self.forward_view_map[m]] = o

    def register_forward_hooks(self):
        """
        This function registers all forward hooks for the modules
        in ModuleDict.  
        """
        for name, module in self._modules.items():
            if isinstance(module, nn.ModuleDict):
                for name, layer in module.items():
                    self.forward_view_map[layer] = name
                    layer.register_forward_hook(self.forward_hook)
                    
    def forward(self, x):
        pass
    
    def save_model(self,
        flag:   str=''
    ):
        # save meta information
        if not os.path.isdir(f"models/{self.name}/"):
            os.makedirs(f"models/{self.name}/")
        output = f"models/{self.name}/" + self.name
        if flag != '':
            output += "_" + flag
        if not os.path.exists("models/"):
            os.makedirs("models/")
        meta_info = [[f'Meta information for model {self.name}']]
        meta_info.append(['date:',datetime.now().strftime("%m/%d/%Y %H:%M:%S")])
        meta_info.append(['user:', getpass.getuser()])
        meta_info.append(['user_id:',os.getuid()])
        system_info = self.logger.get_system_info()
        if len(system_info) > 0:
            meta_info.append(['System information:'])
            for item in system_info:
                meta_info.append([item,system_info[item]])
            meta_info.append([])
        meta_info.append(['Model configuration:'])
        meta_info.append([])
        for item in self.cfg:
            meta_info.append([item, self.cfg[item]])
        meta_info.append([])
        meta_info.append(['Model dictionary:'])
        for item in self.state_dict():
            meta_info.append([item, self.state_dict()[item].size()])
        meta_info.append([])
        with open(output + "_meta.csv", "w") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerows(meta_info)
        # save config
        cfg = [[item, self.cfg[item]] for item in self.cfg]
        with open(output+".cfg", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(cfg)
        # save parameters
        torch.save(
            {
            'model_state_dict': self.state_dict(), 
            'model_config': self.cfg
            }, 
            output + "_params.ckpt"
        )
    
    def load_model(self,
        model_file:   str=''
    ):
        self.logger.info(f"Attempting to load model checkpoint from file {model_file}.")
        try:
            checkpoint = torch.load(model_file)
            self.cfg = checkpoint['model_config']
            self.construct_model()
            # register hooks
            self.register_forward_hooks()
            self.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            self.logger.error(f"Unable to load model file {model_file}: {e}.")
            raise ValueError(f"Unable to load model file {model_file}: {e}.")
        self.logger.info(f"Successfully loaded model checkpoint.")