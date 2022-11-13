"""
Training loop for a CMSSM model   
"""
# CHUNC imports
from chunc.dataset.chunc import CHUNCDataset
from chunc.utils.loader import Loader
from chunc.losses import LossHandler
from chunc.optimizers import Optimizer
from chunc.metrics import MetricHandler
from chunc.trainer import Trainer
from chunc.utils.callbacks import CallbackHandler
from chunc.utils.distributions import generate_gaussian
from chunc.utils.utils import get_files, save_model
from chunc.models import CHUNCC


if __name__ == "__main__":

    # clean up directories first
    save_model()

    """
    Now we load our dataset as a torch dataset (chunccDataset),
    and then feed that into a dataloader.
    """
    features = [
            'gut_m0', 
            'gut_m12', 
            'gut_A0', 
            'gut_tanb', 
            'sign_mu'
    ]
    chuncc_dataset = CHUNCDataset(
        name="chuncc_dataset",
        input_file='datasets/cmssm_dataset_symmetric.npz',
        features = features,
        classes = ['valid']
    )
    chuncc_loader = Loader(
        chuncc_dataset, 
        batch_size=64,
        test_split=0.3,
        test_seed=100,
        validation_split=0.3,
        validation_seed=100,
        num_workers=4
    )
    """
    Construct the chuncc Model, specify the loss and the 
    optimizer and metrics.
    """
    chuncc_cmssm_config = {
        # dimension of the input variables
        'input_dimension':      5,
        # encoder parameters
        'encoder_dimensions':   [25, 50, 100, 50, 25],
        'encoder_activation':   'leaky_relu',
        'encoder_activation_params':    {'negative_slope': 0.02},
        'encoder_normalization':'bias',
        # desired dimension of the latent space
        'latent_dimension':     5,
        'latent_binary':        1,
        'latent_binary_activation': 'sigmoid',
        'latent_binary_activation_params':  {},
        # decoder parameters
        'decoder_dimensions':   [25, 50, 100, 50, 25],
        'decoder_activation':   'leaky_relu',
        'decoder_activation_params':    {'negative_slope': 0.02},
        'decoder_normalization':'bias',
        # output activation
        'output_activation':    'linear',
        'output_activation_params':     {},
    }
    chuncc_model = CHUNCC(
        name = 'chuncc_cmssm',
        cfg  = chuncc_cmssm_config
    ) 

    # create loss, optimizer and metrics
    chuncc_optimizer = Optimizer(
        model=chuncc_model,
        optimizer='Adam'
    )

    # create criterions
    chuncc_loss_config = {
        'L2OutputLoss':   {
            'alpha':    1.0,
            'reduction':'mean',
        },
        'LatentWassersteinLoss': {
            'alpha':    1.0,
            'latent_variables': [0,1,2,3,4],
            'distribution':     generate_gaussian(dimension=5),
            'num_projections':  1000,
        },
        'LatentBinaryLoss': {
            'alpha':    1.0,
            'binary_variable':  5,
            'reduction':    'mean',
        }
    }
    chuncc_loss = LossHandler(
        name="chuncc_loss",
        cfg=chuncc_loss_config,
    )
    
    # create metrics
    chuncc_metric_config = {
        'LatentBinaryAccuracy': {
            'cutoff':   0.5,
            'binary_variable':  5,
        },
        'LatentSaver':  {},
        'TargetSaver':  {},
        'InputSaver':   {},
        'OutputSaver':  {},
    }
    chuncc_metrics = MetricHandler(
        "chuncc_metric",
        cfg=chuncc_metric_config,
    )

    # create callbacks
    callback_config = {
        'loss':   {'criterion_list': chuncc_loss},
        'metric': {'metrics_list':   chuncc_metrics},
        'latent': {
            'criterion_list':   chuncc_loss,
            'metrics_list':     chuncc_metrics,
            'latent_variables': [0,1,2,3,4],
            'binary_variable':  5,
            'binary_bins':      10,
        },
        'output':   {
            'criterion_list':   chuncc_loss,
            'metrics_list':     chuncc_metrics,
            'input_variables':  features,
        }
    }
    chuncc_callbacks = CallbackHandler(
        "chuncc_callbacks",
        callback_config
    )

    # create trainer
    chuncc_trainer = Trainer(
        model=chuncc_model,
        criterion=chuncc_loss,
        optimizer=chuncc_optimizer,
        metrics=chuncc_metrics,
        callbacks=chuncc_callbacks,
        metric_type='test',
        gpu=True,
        gpu_device=0
    )
    
    chuncc_trainer.train(
        chuncc_loader,
        epochs=100,
        checkpoint=25
    )