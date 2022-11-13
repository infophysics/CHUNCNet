"""
Class for running mapper
"""
import numpy as np
import kmapper as km
import pandas as pd
import os
import sklearn
from tqdm import tqdm

from chunc.dataset.dragons import DragonTransform
from chunc.dataset.parameters import *
from chunc.utils.logger import Logger

class MSSMMapper:
    """
    """
    def __init__(self,
        dataset,
        model
    ):
        # set up logger
        self.logger = Logger('mssm_dataset', file_mode='w')
        self.subspace = None
        self.search_columns = []
        self.dataset = dataset
        self.model = model
        
        # create constraint and dataset directories
        if not os.path.isdir("mapper/"):
            os.makedirs("mapper/")

        # create mapper object
        self.logger.info(f"creating Keppler Mapper.")
        self.mapper = km.KeplerMapper(verbose=0)
        self.mapper_nodes = None

    def run_mapper(self,
        num_covers: int=10,
    ):
        # input mapper
        inputs = self.dataset.normalize(self.dataset.event_features)
        outputs, latent = self.model(self.dataset.event_features.unsqueeze(0))
        inputs = inputs.cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        latent = latent.detach().cpu().numpy()

        dist_labels = np.linalg.norm(latent, 2, -1)
        valid_labels = self.dataset.event_classes.numpy().flatten()
        output_labels = np.linalg.norm((inputs - outputs), 2, -1)
        labels = np.vstack((valid_labels,dist_labels,output_labels)).T
        
        outputs = self.dataset.unnormalize(outputs).cpu().numpy()

        # input graph
        inputs_graph = self.parameter_mapper(
            inputs, 
            num_covers=num_covers,
            labels=valid_labels
        )
        # latent graph
        latent_graph = self.parameter_mapper(
            latent, 
            num_covers=num_covers,
            labels=valid_labels
        )
        # outputs graph
        outputs_graph = self.parameter_mapper(
            outputs, 
            num_covers=num_covers,
            labels=valid_labels
        )

        # input 
        self.visualize_mapper(
            inputs_graph, labels, 
            label_names=['valid/invalid', 'latent_distance', 'output_l2'], 
            output_file="inputs",
            title="CMSSM Inputs"
        )
        # latent
        self.visualize_mapper(
            latent_graph, labels, 
            label_names=['valid/invalid', 'latent_distance', 'output_l2'], 
            output_file="latent",
            title="CMSSM Latent"
        )
        # output
        self.visualize_mapper(
            outputs_graph, labels, 
            label_names=['valid/invalid', 'latent_distance', 'output_l2'], 
            output_file="outputs",
            title="CMSSM Output"
        )


    def parameter_mapper(self,
        data,
        num_covers: int=10,
        projections:list=[],
        labels=None
    ):
        """
        """
        # normalize according to standardization
        data = (data - np.mean(data))/np.std(data)
        # fit the data using mapper
        self.logger.info(f"Running fit transform on data.")
        if len(projections) != 0:
            dragon = DragonTransform(projections=projections)
            project_data = self.mapper.fit_transform(
                data, projection=DragonTransform(projections=projections)
            )
        else:
            project_data = self.mapper.fit_transform(
                data, 
                projection=sklearn.manifold.TSNE()
                #projection=umap.UMAP(n_components=2)
            )
        # create a cover with num_covers elements
        self.logger.info(f"Constructing cover with {num_covers} cubes per dimension.")
        cover = km.Cover(n_cubes=num_covers)
        # create dictionary called 'graph' with nodes, edges and etc.
        self.logger.info(f"Generating simplicial complex from mapper.")
        graph = self.mapper.map(
            project_data, 
            data, 
            clusterer=sklearn.cluster.DBSCAN(eps=0.1*np.sqrt(len(data[0])), min_samples=3),
            cover=cover
        )
        self.mapper_nodes = graph['nodes']
        self.f_i = [len(self.mapper_nodes[item])/len(data) for item in self.mapper_nodes.keys()]
        if len(labels) != 0:
            unique_labels = np.unique(labels)
            self.f_ji = [[0 for i in range(len(unique_labels)+1)] for j in range(len(self.f_i))]
            for ii, item in enumerate(self.mapper_nodes.keys()):
                points = self.mapper_nodes[item]
                for point in points:
                    self.f_ji[ii][int(labels[point])] += 1/len(points)
        return graph
    
    def visualize_mapper(self,
        simplicial_complex,
        labels,
        label_names:    list=['type'],
        output_file:    str='output',
        title:      str="CMSSM constrained/unconstrained search"
    ):
        html = self.mapper.visualize(
            simplicial_complex,
            path_html="mapper/"+output_file+".html",
            title=title,
            color_values=labels,
            color_function_name=label_names,
            node_color_function=['mean', 'std', 'median', 'max']
        )