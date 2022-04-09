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
from chunc.models import CHUNCC


if __name__ == "__main__":
    """
    Now we load our dataset as a torch dataset (SWAEDataset),
    and then feed that into a dataloader.
    """
    features = [
            'gut_m0', 
            'gut_m12', 
            'gut_A0', 
            'gut_tanb', 
            'sign_mu'
    ]
    swae_dataset = CHUNCDataset(
        name="swae_dataset",
        input_file='datasets/cmssm_dataset_symmetric.npz',
        features = features,
        classes = ['valid']
    )
    swae_loader = Loader(
        swae_dataset, 
        batch_size=64,
        test_split=0.3,
        test_seed=100,
        validation_split=0.3,
        validation_seed=100,
        num_workers=4
    )
    """
    Construct the SWAE Model, specify the loss and the 
    optimizer and metrics.
    """
    swae_cmssm_config = {
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
    swae_model = CHUNCC(
        name = 'swae_cmssm',
        cfg  = swae_cmssm_config
    ) 

    # create loss, optimizer and metrics
    swae_optimizer = Optimizer(
        model=swae_model,
        optimizer='Adam'
    )

    # create criterions
    swae_loss_config = {
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
    swae_loss = LossHandler(
        name="swae_loss",
        cfg=swae_loss_config,
    )
    
    # create metrics
    swae_metric_config = {
        'LatentBinaryAccuracy': {
            'cutoff':   0.5,
            'binary_variable':  5,
        },
        'LatentSaver':  {},
        'TargetSaver':  {},
        'InputSaver':   {},
        'OutputSaver':  {},
    }
    swae_metrics = MetricHandler(
        "swae_metric",
        cfg=swae_metric_config,
    )

    # create callbacks
    callback_config = {
        'loss':   {'criterion_list': swae_loss},
        'metric': {'metrics_list':   swae_metrics},
        'latent': {
            'criterion_list':   swae_loss,
            'metrics_list':     swae_metrics,
            'latent_variables': [0,1,2,3,4],
            'binary_variable':  5,
            'binary_bins':      10,
        },
        'output':   {
            'criterion_list':   swae_loss,
            'metrics_list':     swae_metrics,
            'input_variables':  features,
        }
    }
    swae_callbacks = CallbackHandler(
        "swae_callbacks",
        callback_config
    )

    # create trainer
    swae_trainer = Trainer(
        model=swae_model,
        criterion=swae_loss,
        optimizer=swae_optimizer,
        metrics=swae_metrics,
        callbacks=swae_callbacks,
        metric_type='test',
        gpu=True,
        gpu_device=0
    )
    
    swae_trainer.train(
        swae_loader,
        epochs=100,
        checkpoint=25
    )