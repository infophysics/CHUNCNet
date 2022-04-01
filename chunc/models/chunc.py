"""
Implementation of the chunc model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from chunc.models.common import activations, normalizations
from chunc.models import GenericModel

chunc_cmssm_config = {
    # dimension of the input variables
    'input_dimension':      5,
    # encoder parameters
    'encoder_dimensions':   [10, 25, 50, 25, 10],
    'encoder_activation':   'leaky_relu',
    'encoder_activation_params':    {'negative_slope': 0.02},
    'encoder_normalization':'bias',
    # desired dimension of the latent space
    'latent_dimension':     5,
    'latent_binary':        1,
    'latent_binary_activation': 'sigmoid',
    'latent_binary_activation_params':  {},
    # decoder parameters
    'decoder_dimensions':   [10, 25, 50, 25, 10],
    'decoder_activation':   'leaky_relu',
    'decoder_activation_params':    {'negative_slope': 0.02},
    'decoder_normalization':'bias',
    # output activation
    'output_activation':    'linear',
    'output_activation_params':     {},
}

class CHUNC(GenericModel):
    """
    
    """
    def __init__(self,
        name:   str='chunc_cmssm',
        cfg:    dict=chunc_cmssm_config,
    ):
        super(CHUNC, self).__init__(name, cfg)
        self.cfg = cfg
        # check cfg
        self.logger.info(f"checking CHUNC architecture using cfg: {self.cfg}")
        for item in chunc_cmssm_config.keys():
            if item not in self.cfg:
                self.logger.error(f"parameter {item} was not specified in config file {self.cfg}")
                raise AttributeError(f"parameter {item} was not specified in config file {self.cfg}")
        if self.cfg['encoder_activation'] not in activations:
            self.logger.error(f"Specified activation {self.cfg['encoder_activation']} is not an allowed type.")
            raise AttributeError(f"Specified activation {self.cfg['encoder_activation']} is not an allowed type.")
        if self.cfg['encoder_normalization'] not in normalizations:
            self.logger.error(f"Specified normalization {self.cfg['encoder_normalization']} is not an allowed type.")
            raise AttributeError(f"Specified normalization {self.cfg['encoder_normalization']} is not an allowed type.")
        if self.cfg['decoder_activation'] not in activations:
            self.logger.error(f"Specified activation {self.cfg['decoder_activation']} is not an allowed type.")
            raise AttributeError(f"Specified activation {self.cfg['decoder_activation']} is not an allowed type.")
        if self.cfg['decoder_normalization'] not in normalizations:
            self.logger.error(f"Specified normalization {self.cfg['decoder_normalization']} is not an allowed type.")
            raise AttributeError(f"Specified normalization {self.cfg['decoder_normalization']} is not an allowed type.")
        if self.cfg['output_activation'] != 'linear':
            if self.cfg['output_activation'] not in activations:
                self.logger.error(f"Specified activation {self.cfg['output_activation']} is not an allowed type.")
                raise AttributeError(f"Specified activation {self.cfg['output_activation']} is not an allowed type.")
        # construct the model
        self.forward_views      = {}
        self.forward_view_map   = {}
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()
        
    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build chunc architecture using cfg: {self.cfg}")
        _encoder_dict = OrderedDict()
        _latent_dict = OrderedDict()
        _decoder_dict = OrderedDict()
        _output_dict = OrderedDict()

        self.input_dimension = self.cfg['input_dimension']
        input_dimension = self.input_dimension
        # iterate over the encoder
        for ii, dimension in enumerate(self.cfg['encoder_dimensions']):
            if self.cfg['encoder_normalization'] == 'bias':
                _encoder_dict[f'encoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=True
                )
            else:
                _encoder_dict[f'encoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=False
                )
                _encoder_dict[f'encoder_{ii}_batchnorm'] = nn.BatchNorm1d(
                    num_features=dimension
                )
            _encoder_dict[f'encoder_{ii}_activation'] = activations[self.cfg['encoder_activation']](**self.cfg['encoder_activation_params'])
            input_dimension=dimension
            
        # create the latent space
        _latent_dict['latent_layer'] = nn.Linear(
            in_features=dimension,
            out_features=self.cfg['latent_dimension'],
            bias=False
        )
        _latent_dict['latent_binary'] = nn.Linear(
            in_features=dimension,
            out_features=1,
            bias=False
        )
        _latent_dict['latent_binary_activation'] = activations[self.cfg['latent_binary_activation']](**self.cfg['latent_binary_activation_params'])
        
        input_dimension = self.cfg['latent_dimension'] + self.cfg['latent_binary']
        # iterate over the decoder
        for ii, dimension in enumerate(self.cfg['decoder_dimensions']):
            if self.cfg['decoder_normalization'] == 'bias':
                _decoder_dict[f'decoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=True
                )
            else:
                _decoder_dict[f'decoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=False
                )
                _decoder_dict[f'decoder_{ii}_batchnorm'] = nn.BatchNorm1d(
                    num_features=dimension
                )
            _decoder_dict[f'decoder_{ii}_activation'] = activations[self.cfg['decoder_activation']](**self.cfg['decoder_activation_params'])
            input_dimension=dimension
        # create the output
        _output_dict['output'] = nn.Linear(
            in_features=dimension,
            out_features=self.input_dimension,
            bias=False
        )
        if self.cfg['output_activation'] != 'linear':
            _output_dict['output_activation'] = activations[self.cfg['output_activation']](**self.cfg['output_activation_params'])
        # create the dictionaries
        self.encoder_dict = nn.ModuleDict(_encoder_dict)
        self.latent_dict = nn.ModuleDict(_latent_dict)
        self.decoder_dict = nn.ModuleDict(_decoder_dict)
        self.output_dict = nn.ModuleDict(_output_dict)
        # record the info
        self.logger.info(f"constructed chunc with dictionaries:\n{self.encoder_dict}\n{self.latent_dict}\n{self.decoder_dict}\n{self.output_dict}.")

    def forward(self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        x = x[0].to(self.device)
        # first the encoder
        for layer in self.encoder_dict.keys():
            x = self.encoder_dict[layer](x)
        latent = self.latent_dict['latent_layer'](x)
        binary = self.latent_dict['latent_binary'](x)
        binary = self.latent_dict['latent_binary_activation'](binary)
        x = torch.cat((latent, binary), dim=1)
        for layer in self.decoder_dict.keys():
            x = self.decoder_dict[layer](x)
        for layer in self.output_dict.keys():
            x = self.output_dict[layer](x)
        latent_output = torch.cat(
            (self.forward_views['latent_layer'],self.forward_views['latent_binary_activation']),
            dim=1
        )
        return x, latent_output

    def sample(self,
        x
    ):
        """
        Returns an output given a input from the latent space
        """
        for layer in self.decoder_dict.keys():
            x = self.decoder_dict[layer](x)
        for layer in self.output_dict.keys():
            x = self.output_dict[layer](x)
        return x

    def latent(self,
        x,
    ):
        """
        Get the latent representation of an input
        """
        x = x[0].to(self.device)
        # first the encoder
        for layer in self.encoder_dict.keys():
            x = self.encoder_dict[layer](x)
        latent = self.latent_dict['latent_layer'](x)
        binary = self.latent_dict['latent_binary'](x)
        binary = self.latent_dict['latent_binary_activation'](binary)
        x = torch.cat((latent, binary), dim=1)
        return x