"""
Implementation of the SWAE model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from chunc.models.common import activations, normalizations
from chunc.models import GenericModel

swae_cmssm_config = {
    # dimension of the input variables
    'input_dimension':      5,
    # encoder parameters
    'encoder_dimensions':   [10, 25, 50, 25, 10],
    'encoder_activation':   'leaky_relu',
    'encoder_activation_params':    {'negative_slope': 0.02},
    'encoder_normalization':'bias',
    # desired dimension of the latent space
    'latent_dimension':     5,
    'latent_constraints':   0,
    # decoder parameters
    'decoder_dimensions':   [10, 25, 50, 25, 10],
    'decoder_activation':   'leaky_relu',
    'decoder_activation_params':    {'negative_slope': 0.02},
    'decoder_normalization':'bias',
    # output activation
    'output_activation':    'linear',
    'output_activation_params':     {},
}

class SWAE(GenericModel):
    """
    
    """
    def __init__(self,
        name:   str='swae_cmssm',
        cfg:    dict=swae_cmssm_config,
    ):
        super(SWAE, self).__init__(name, cfg)
        self.cfg = cfg
        # check cfg
        self.logger.info(f"checking SWAE architecture using cfg: {self.cfg}")
        for item in swae_cmssm_config.keys():
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
        self.logger.info(f"Attempting to build SWAE architecture using cfg: {self.cfg}")
        _model_dict = OrderedDict()

        self.input_dimension = self.cfg['input_dimension']
        input_dimension = self.input_dimension
        # iterate over the encoder
        for ii, dimension in enumerate(self.cfg['encoder_dimensions']):
            if self.cfg['encoder_normalization'] == 'bias':
                _model_dict[f'encoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=True
                )
            else:
                _model_dict[f'encoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=False
                )
                _model_dict[f'encoder_{ii}_batchnorm'] = nn.BatchNorm1d(
                    num_features=dimension
                )
            _model_dict[f'encoder_{ii}_activation'] = activations[self.cfg['encoder_activation']](**self.cfg['encoder_activation_params'])
            input_dimension=dimension
        # create the latent space
        _model_dict['latent_layer'] = nn.Linear(
            in_features=dimension,
            out_features=self.cfg['latent_dimension'] + self.cfg['latent_constraints'],
            bias=False
        )
        input_dimension = self.cfg['latent_dimension'] + self.cfg['latent_constraints']
        # iterate over the decoder
        for ii, dimension in enumerate(self.cfg['decoder_dimensions']):
            if self.cfg['decoder_normalization'] == 'bias':
                _model_dict[f'decoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=True
                )
            else:
                _model_dict[f'decoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=False
                )
                _model_dict[f'decoder_{ii}_batchnorm'] = nn.BatchNorm1d(
                    num_features=dimension
                )
            _model_dict[f'decoder_{ii}_activation'] = activations[self.cfg['decoder_activation']](**self.cfg['decoder_activation_params'])
            input_dimension=dimension
        # create the output
        _model_dict['output'] = nn.Linear(
            in_features=dimension,
            out_features=self.input_dimension,
            bias=False
        )
        if self.cfg['output_activation'] != 'linear':
            _model_dict['output_activation'] = activations[self.cfg['output_activation']](**self.cfg['output_activation_params'])
        # create the dictionaries
        self.module_dict = nn.ModuleDict(_model_dict)
        # record the info
        self.logger.info(f"Constructed SWAE with dictionary: {self.module_dict}.")

    def forward(self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        x = x[0].to(self.device)
        for layer in self.module_dict.keys():
            x = self.module_dict[layer](x)
        return x, self.forward_views['latent_layer']

    def sample(self,
        x
    ):
        """
        Returns an output given a input from the latent space
        """
        reached_latent = False
        for layer in self.module_dict.keys():
            if reached_latent:
                x = self.module_dict[layer](x)
            else:
                if layer == 'latent_layer':
                    reached_latent = True
        return x

    def latent(self,
        x,
    ):
        """
        Get the latent representation of an input
        """
        for layer in self.module_dict.keys():
            if layer != 'latent_layer':
                x = self.module_dict[layer](x)
            else:
                return x