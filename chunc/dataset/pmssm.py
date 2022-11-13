"""
Script for testing the mapper algorithm on CMSSM and PMSSM theories
    Nicholas Carrara [nmcarrara@ucdavis.edu]
"""
import numpy as np
import kmapper as km
import pandas as pd
import os
import socket
from datetime import datetime
import sklearn
from tqdm import tqdm
#from dragons import DragonTransform

from chunc.dataset.parameters import *
from chunc.dataset.mssm import MSSMDataset

class pMSSMDataset(MSSMDataset):
    """
    Basic cmssm/pmssm dataset creation and manipulation class.
    """
    def __init__(self,
        input_dir,
    ):
        super(pMSSMDataset, self).__init__(input_dir)
        self.subspace = 'pmssm'
        self.subspace_columns = pmssm_columns
        self.search_columns = self.subspace_columns + common_columns
    

    def generate_training_set(self,
        constrained_file:   str,
        unconstrained_file: str,
        symmetric_events:   bool=True,
        close_gap:          bool=False,
        labeling:           str='binary',
        save:               bool=True,
        output_file:        str='',
        meta_dict:          dict=None,
    ):
        self.logger.info(f"Attempting to construct training set from files: {constrained_file},{unconstrained_file}.")
        if not os.path.isfile(constrained_file):
            self.logger.error(f"Specified constrained file '{constrained_file}' not in path.")
            raise ValueError(f"Specified constrained file '{constrained_file}' not in path.")
        if not os.path.isfile(unconstrained_file):
            self.logger.error(f"Specified unconstrained file '{unconstrained_file}' not in path.")
            raise ValueError(f"Specified unconstrained file '{unconstrained_file}' not in path.")
        if labeling not in ['binary', 'higgs_relic']:
            self.logger.warning(f"Specified labeling '{labeling}' not allowed, using 'binary'")
            labeling = 'binary'
        self.logger.info(f"Constructing set with symmetric_events: {symmetric_events}.")
        self.gut_m1_gap = 50.0
        self.gut_m2_gap = 100.0
        self.gut_mmu_gap = 100.0
        # load files
        constrained_dataframe = pd.read_csv(
            constrained_file, 
            sep=',', 
            header=None, 
            names=pmssm_columns, 
            usecols=pmssm_columns
        )
        self.logger.info(f"Loaded constrained file: {constrained_file} with {len(constrained_dataframe)} events.")
        unconstrained_dataframe = pd.read_csv(
            unconstrained_file, 
            sep=',', 
            header=None, 
            names=pmssm_columns, 
            usecols=pmssm_columns
        )
        self.logger.info(f"Loaded unconstrained file: {unconstrained_file} with {len(unconstrained_dataframe)} events.")
        # create truth info
        if symmetric_events:
            num_events = min([len(constrained_dataframe),len(unconstrained_dataframe)])
            self.logger.info(f"Using {num_events} from each file.")
            constrained_events = constrained_dataframe.sample(num_events).values
            unconstrained_events = unconstrained_dataframe.sample(num_events).values
            event_sample_weights = np.ones(len(constrained_events)+len(unconstrained_events)).reshape(-1,1)
        else:
            self.logger.info(f"Using {len(constrained_dataframe)} constrained events and {len(unconstrained_dataframe)} unconstrained events.")
            if len(constrained_dataframe) < len(unconstrained_dataframe):
                constrained_weights = np.ones(len(constrained_dataframe)).reshape(-1,1)
                unconstrained_weight = len(constrained_weights)/len(unconstrained_dataframe)
                unconstrained_weights = np.full((len(unconstrained_dataframe),),unconstrained_weight).reshape(-1,1)
                event_sample_weights = np.concatenate((constrained_weights,unconstrained_weights))
            else:
                unconstrained_weights = np.ones(len(unconstrained_dataframe)).reshape(-1,1)
                constrained_weight = len(unconstrained_weights)/len(constrained_dataframe)
                constrained_weights = np.full((len(constrained_dataframe),),constrained_weight).reshape(-1,1)
                event_sample_weights = np.concatenate((constrained_weights,unconstrained_weights))
            constrained_events = constrained_dataframe.values
            unconstrained_events = unconstrained_dataframe.values
        constrained_labels = np.ones(len(constrained_events)).reshape(-1,1)
        unconstrained_labels = np.zeros(len(unconstrained_events)).reshape(-1,1)
        event_features = np.concatenate((constrained_events, unconstrained_events))
        event_classes = np.concatenate((constrained_labels, unconstrained_labels))
        
        meta_dict = {
            "who_created": "me",
            "when_created": datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
            "where_created": socket.gethostname(),
            "num_events": len(event_features),
            "features": {
                key:ii for ii, key in enumerate(pmssm_columns)
            },
            "close_gap": close_gap,
            "classes": {
                "valid": 0
            },
            "sample_weights": {
                "weights": 0
            },
        }
        
        if close_gap:
            for ii in range(len(event_features)):
                # close the gut_m1 gap
                if event_features[ii][0] < 0.0:
                    event_features[ii][0] += self.gut_m1_gap
                else:
                    event_features[ii][0] -= self.gut_m1_gap
                # close the gut m2 gap
                if event_features[ii][1] < 0.0:
                    event_features[ii][1] += self.gut_m2_gap
                else:
                    event_features[ii][1] -= self.gut_m2_gap
                # close the gut mmu gap
                if event_features[ii][3] < 0.0:
                    event_features[ii][3] += self.gut_mmu_gap
                else:
                    event_features[ii][3] -= self.gut_mmu_gap   
            pass

        if save:
            np.savez(
                "datasets/"+output_file,
                meta=meta_dict,
                event_features=event_features,
                event_classes=event_classes,
                event_sample_weights=event_sample_weights
            )
    
    def load_constrained_set(self,
        constrained_file:   str,
        labeling:           str='pmssm',
    ):
        self.logger.info(f"Attempting to construct training set from file: {constrained_file}.")
        if not os.path.isfile(constrained_file):
            self.logger.error(f"Specified constrained file '{constrained_file}' not in path.")
            raise ValueError(f"Specified constrained file '{constrained_file}' not in path.")
        if labeling not in ['pmssm']:
            self.logger.warning(f"Specified labeling '{labeling}' not allowed, using 'pmssm'")
            labeling = 'pmssm'
        # load files
        constrained_dataframe = pd.read_csv(
            constrained_file, 
            sep=',', 
            header=None, 
            names=self.subspace_columns, 
            usecols=self.subspace_columns
        )
        self.logger.info(f"Loaded constrained file: {constrained_file} with {len(constrained_dataframe)} events.")
        # create truth info
        events = constrained_dataframe.values
        if labeling == 'pmssm':
            constrained_labels_frame = pd.read_csv(
                constrained_file, 
                sep=',', 
                header=None, 
                names=['ann_channel'], 
                usecols=[5]
            )
            labels = constrained_labels_frame.values
        return events, labels


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
        output_file:    str='output'
    ):
        html = self.mapper.visualize(
            simplicial_complex,
            path_html="html/"+output_file+".html",
            title="CMSSM constrained/unconstrained search",
            color_values=labels,
            color_function_name=label_names,
            node_color_function=['mean', 'std', 'median', 'max']
        )