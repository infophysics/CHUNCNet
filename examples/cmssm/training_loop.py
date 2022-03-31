"""
Training loop for a CMSSM model   
"""
# CHUNC imports
from chunc.dataset.chunc import CHUNCDataset
from chunc.utils.loader import Loader
from chunc.models import SWAE
from chunc.losses import LossHandler
from chunc.optimizers import Optimizer
from chunc.metrics import MetricHandler
from chunc.trainer import Trainer
from chunc.utils.callbacks import CallbackHandler
from chunc.utils.distributions import generate_sphere
from chunc.utils.distributions import generate_concentric_spheres
from chunc.utils.distributions import generate_gaussian
from chunc.models import SWAE


if __name__ == "__main__":
    """
    Now we load our dataset as a torch dataset (SWAEDataset),
    and then feed that into a dataloader.
    """
    swae_dataset = CHUNCDataset(
        name="swae_dataset",
        input_file='datasets/cmssm_dataset_symmetric.npz',
        features = [
            'gut_m0', 
            'gut_m12', 
            'gut_A0', 
            'gut_tanb', 
            'sign_mu'
        ],
        classes = ['valid']
    )
    swae_loader = Loader(
        swae_dataset, 
        batch_size=32,
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
    swae_model = SWAE(
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
        'WassersteinLoss': {
            'alpha':    1.0,
            'distribution_type':  'input',
            'encoded_type':       'output',
            'num_projections':    1000,
        }
    }
    swae_loss = LossHandler(
        name="swae_loss",
        cfg=swae_loss_config,
    )
    
    # # create metrics
    # swae_metric_config = {
    #     'binary_sphere':    {
    #         'cutoff':       0.5
    #     },
    #     'swae_saver':   {}
    # }
    # swae_metrics = MetricHandler(
    #     "swae_metric",
    #     cfg=swae_metric_config,
    # )

    # create callbacks
    callback_config = {
        'loss':   {'criterion_list': swae_loss},
        #'metric': {'metrics_list':   swae_metrics},
        #'swae_callback':  {'swae_saver':swae_metrics.metrics['swae_saver']}
        #'binary_classification': {}
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
        #metrics=swae_metrics,
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