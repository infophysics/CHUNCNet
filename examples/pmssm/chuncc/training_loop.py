"""
Training loop for a pmssm model   
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
    Now we load our dataset as a torch dataset (chunccDataset),
    and then feed that into a dataloader.
    """
    features = [
        'gut_m1', 'gut_m2', 
        'gut_m3', 'gut_mmu', 
        'gut_mA', 'gut_At', 
        'gut_Ab', 'gut_Atau', 
        'gut_mL1','gut_mL3', 
        'gut_me1','gut_mtau1', 
        'gut_mQ1','gut_mQ3', 
        'gut_mu1','gut_mu3', 
        'gut_md1','gut_md3', 
        'gut_tanb'
    ]
    chuncc_dataset = CHUNCDataset(
        name="chuncc_dataset",
        input_file='datasets/pmssm_higgs_dm_lsp_symmetric_no_gap.npz',
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
    chuncc_pmssm_config = {
        # dimension of the input variables
        'input_dimension':      19,
        # encoder parameters
        'encoder_dimensions':   [25, 50, 100, 50, 25],
        'encoder_activation':   'leaky_relu',
        'encoder_activation_params':    {'negative_slope': 0.02},
        'encoder_normalization':'bias',
        # desired dimension of the latent space
        'latent_dimension':     19,
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
        name = 'chuncc_pmssm',
        cfg  = chuncc_pmssm_config
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
            'latent_variables': [ii for ii in range(19)],
            'distribution':     generate_gaussian(dimension=19),
            'num_projections':  1000,
        },
        'LatentBinaryLoss': {
            'alpha':    1.0,
            'binary_variable':  19,
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
            'binary_variable':  19,
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
            'latent_variables': [ii for ii in range(19)],
            'binary_variable':  19,
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