"""
Various utility functions
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
import inspect
import os

"""
Get the names of the arrays in an .npz file
"""
def get_array_names(
    input_file: str
):
    # check that file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Specified input file: '{input_file}' does not exist!")
    loaded_file = np.load(input_file)
    return list(loaded_file.files)

"""
The following function takes in a .npz file, and a set
of arrays specified by a dictionary, and appends them
to the .npz file, provided there are no collisions.
"""
def append_npz(
    input_file: str,
    arrays:     dict,
    override:   bool=False,
):
    # check that file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Specified input file: '{input_file}' does not exist!")
    if not isinstance(arrays, dict):
        raise ValueError(f"Specified array must be a dictionary, not '{type(arrays)}'!")
    # otherwise load file and check contents
    loaded_file = np.load(input_file, allow_pickle=True)
    loaded_arrays = {
        key: loaded_file[key] for key in loaded_file.files
    }
    # check that there are no identical array names if override set to false
    if override == False:
        for item in loaded_arrays.keys():
            if item in arrays.keys():
                raise ValueError(f"Array '{item}' already exists in .npz file '{input_file}'!")
    # otherwise add the array and save
    loaded_arrays.update(arrays)
    np.savez(
        input_file,
        **loaded_arrays
    )

"""
Get a list of arguments and default values for a method.
"""
def get_method_arguments(method):
    # argpase grabs input values for the method
    try:
        argparse = inspect.getfullargspec(method)
        args = argparse.args
        args.remove('self')
        default_params = [None for item in args]
        if argparse.defaults != None:
            for ii, value in enumerate(argparse.defaults):
                default_params[-(ii+1)] = value
        argdict = {item: default_params[ii] for ii, item in enumerate(args)}
        return argdict
    except:
        return {}

"""
Method for getting shapes of data and various other
useful information.
"""
def get_shape_dictionary(
    dataset=None,
    dataset_loader=None,
    model=None,
):
    data_shapes = {}
    # list of desired dataset values
    dataset_values = [
        'feature_shape',
        'class_shape',
    ]
    for item in dataset_values:
        try:
            data_shapes[item] = getattr(dataset, item)
        except:
            data_shapes[item] = 'missing'
    # list of desired dataloader values
    dataset_loader_values = [
        'num_total_train',
        'num_test',
        'num_train',
        'num_validation',
        'num_train_batches',
        'num_validation_batches',
        'num_test_batches',
    ]
    for item in dataset_loader_values:
        try:
            data_shapes[item] = getattr(dataset_loader, item)
        except:
            data_shapes[item] = 'missing'
    # list of desired model values
    model_values = [
        'input_shape',
        'output_shape',
    ]
    for item in model_values:
        try:
            data_shapes[item] = getattr(model, item)
        except:
            data_shapes[item] = 'missing'
    return data_shapes

def boxcar(
    x,
    mean:   float=0.11,
    sigma:  float=0.3,
    mode:   str='regular',
):  
    """
    Returns a value between -1 and ...
    depending on whether the values (x - (mean +- sigma))
    are < 0, == 0, or > 0.  If 
        a) regular :  return 0 if x < or > mean+-sigma
        b) regular :  return 1 if x > low but <= high
    """
    unit = torch.tensor([1.0])
    high = torch.heaviside(x - torch.tensor([mean + sigma]), unit)
    low = torch.heaviside(x - torch.tensor([mean - sigma]), unit)
    if mode == 'regular':
        return low - high
    else:
        return unit + high - low

def get_base_classes(derived):
    """
    Determine the base classes of some potentially inherited object.
    """
    bases = []
    try:
        for base in derived.__class__.__bases__:
            bases.append(base.__name__)
    except:
        pass
    return bases

def generate_plot_grid(
    num_plots,
    **kwargs,
):
    nrows = int(np.floor(np.sqrt(num_plots)))
    ncols = int(np.ceil(num_plots/nrows))
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        **kwargs
    )
    nplots = nrows * ncols
    nextra = nplots - num_plots
    for ii in range(nextra):
        axs.flat[-(ii+1)].set_visible(False)
    return fig, axs

def concatenate_csv(
    files,
    output_file
):
    combined_csv = pd.concat([pd.read_csv(f) for f in files])
    combined_csv.to_csv(output_file, header=None, index=False,)
    
