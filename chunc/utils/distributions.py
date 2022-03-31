"""
Tools for generating distributions.
"""
import numpy as np
import torch
import os
from matplotlib import pyplot as plt


def generate_gaussian(
    number_of_samples:  int=100000,
    dimension:  int=5,
    mean:       float=0.0,
    sigma:      float=1.0,
):
    means = torch.full(
        size=(number_of_samples,dimension), 
        fill_value=mean
    )
    stds = torch.full(
        size=(number_of_samples,dimension), 
        fill_value=sigma
    )
    normal = torch.normal(
        mean=means,
        std=stds,
    )
    return normal

def generate_concentric_spheres(
    number_of_samples:  int=10000,
    dimension:      int=5,
    inner_radius:   float=0.3,
    outer_radius:   float=1.0,
    thickness:      float=0.3,
    save_plot:      bool=True,
):
    # construct inner uniform
    inner_uniform = inner_radius * torch.rand(
        size=(number_of_samples, dimension),
        dtype=float
    )

    # construct outer uniform
    outer_uniform = outer_radius + thickness * (2 * torch.rand(
        size=(number_of_samples, dimension),
        dtype=float
    ) - 1.)
    
    # construct inner sphere
    inner_sphere_vecs = 2 * torch.rand(
        size=(number_of_samples, dimension),
        dtype=float
    ) - 1.
    inner_sphere_norm = torch.norm(
        inner_sphere_vecs, p=2, dim=1).unsqueeze(1).expand_as(inner_sphere_vecs)
    inner_sphere_vecs /= inner_sphere_norm
    inner_sphere_vecs *= inner_uniform

    # construct outer sphere
    outer_sphere_vecs = 2 * torch.rand(
        size=(number_of_samples, dimension),
        dtype=float
    ) - 1.
    outer_sphere_norm = torch.norm(
        outer_sphere_vecs, p=2, dim=1).unsqueeze(1).expand_as(outer_sphere_vecs)
    outer_sphere_vecs /= outer_sphere_norm
    outer_sphere_vecs *= outer_uniform

    torchcon = torch.cat((inner_sphere_vecs, outer_sphere_vecs),0)
    if save_plot:
        if not os.path.isdir("plots/distribution/"):
            os.makedirs("plots/distribution/")
        fig, axs = plt.subplots()
        for ii in range(dimension):
            axs.hist(
                torchcon[:,ii].numpy(), 
                bins=100, 
                label=f'dimension_{ii}', 
                histtype='step', 
                density=True, stacked=True
            )
        axs.set_xlabel(f'x')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig("plots/distribution/concentric_x.png")

        fig, axs = plt.subplots()
        axs.hist(
            torch.norm(inner_sphere_vecs, p=2, dim=1).numpy(), 
            bins=100, 
            label=f'inner_sphere', 
            histtype='step', 
            density=True, stacked=True
        )
        axs.hist(
            torch.norm(outer_sphere_vecs, p=2, dim=1).numpy(),
            bins=100,
            label=f'outer_sphere',
            histtype='step',
            density=True, stacked=True
        )
        axs.set_xlabel(f'r')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig("plots/distribution/concentric_r.png")
        
    return torchcon

def generate_sphere(
    number_of_samples:  int=10000,
    dimension:  int=5,
    radius:     float=1.0,
    thickness:  float=0.1,
    save_plot:  bool=True,
):
    # construct uniform
    uniform = radius + thickness * (2 * torch.rand(
        size=(number_of_samples, dimension),
        dtype=float
    ) - 1.)

    # construct sphere
    sphere_vecs = 2 * torch.rand(
        size=(number_of_samples, dimension),
        dtype=float
    ) - 1.
    sphere_norm = torch.norm(
        sphere_vecs, p=2, dim=1).unsqueeze(1).expand_as(sphere_vecs)
    sphere_vecs /= sphere_norm
    sphere_vecs *= uniform

    if save_plot:
        if not os.path.isdir("plots/distribution/"):
            os.makedirs("plots/distribution/")
        fig, axs = plt.subplots()
        for ii in range(dimension):
            axs.hist(
                sphere_vecs[:,ii].numpy(), 
                bins=100, 
                label=f'dimension_{ii}', 
                histtype='step', 
                density=True, stacked=True
            )
        axs.set_xlabel(f'x')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"plots/distribution/sphere_x_{radius}_{thickness}.png")

        fig, axs = plt.subplots()
        axs.hist(
            torch.norm(sphere_vecs, p=2, dim=1).numpy(),
            bins=100,
            label=f'sphere',
            histtype='step',
            density=True, stacked=True
        )
        axs.set_xlabel(f'r')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"plots/distribution/sphere_r_{radius}_{thickness}.png")
        
    return sphere_vecs

def generate_uniform_annulus(
    number_of_samples:  int=10000, 
    dimension:      int=5,
    inner_radius:   float=1.0, 
    outer_radius:   float=1.5,
    scale:          float=1.5,
    samp=None
):
    # create a sample of num entries from a
    # concentric distirbution with missing sliver of
    # inner radius and outer radius
    if samp == None:
        samp = torch.tensor([[0.0 for ii in range(dimension)]])
    conlist = np.zeros((1,dimension), dtype=float)
    conlist = np.delete(conlist, 0, 0)
    torchcon = torch.from_numpy(conlist)
    while torchcon.size()[0] < number_of_samples:
        eps = (2*scale)*(torch.rand_like(samp) - 1/2)
        if (
            (torch.linalg.norm(eps) <= inner_radius) or 
            (torch.linalg.norm(eps) >= outer_radius and torch.linalg.norm(eps) <= scale)
        ):
            torchcon = torch.cat((torchcon,eps),0)
    return torchcon

