"""
Script for testing the mapper algorithm on CMSSM and PMSSM theories
    Nicholas Carrara [nmcarrara@ucdavis.edu]
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

class MSSMDataset:
    """
    Basic mssm dataset creation and manipulation class.
    """
    def __init__(self,
        input_dir,
    ):
        # set up logger
        self.logger = Logger('mssm_dataset', file_mode='w')
        self.logger.info(f"attempting to add input directory: {input_dir}.")
        if not os.path.isdir(input_dir):
            self.logger.error(f"specified input directory '{input_dir}' not in path.")
        self.logger.info(f"adding input directory: {input_dir}.")
        self.input_dir = input_dir
        self.subspace = None
        self.search_columns = []
        
        # create constraint and dataset directories
        if not os.path.isdir("constraints/"):
            os.makedirs("constraints/")
        if not os.path.isdir("datasets/"):
            os.makedirs("datasets/")

        # create mapper object
        self.logger.info(f"creating Keppler Mapper.")
        self.mapper = km.KeplerMapper(verbose=0)
        self.mapper_nodes = None

    def standard_model_constraints(self,
        X,
        params:     dict=base_params,
        apply_higgs:    bool=True,
        apply_dm:       bool=True,
        apply_lsp:      bool=True,
    ):
        """
        Constraint calculator for CMSSM or PMSSM models
        """
        # check neutralino lsp
        if apply_lsp:
            neutralino_lsp = (
                round(X['lspmass']) == round(X['weakm_mneut1'])
            )
            X = X[neutralino_lsp]
        # check if the relic density is saturated
        if apply_dm:
            relic_density_saturated = (
                (round(X['omegah2'],2) - params["dm_relic_density"]).abs() < params["dm_relic_density_sigma"]
            )
            X = X[relic_density_saturated]
        # see if the higgs mass is within experimental bounds
        if apply_higgs:
            higgs = (
                (round(X['weakm_mh'],2) - params["higgs_mass"]).abs() < params["higgs_mass_sigma"]
            )
            X = X[higgs]
        return X
    
    def diff(self,
        X,
        Y,
    ):
        """
        Computes |X-Y|/min(X,Y)
        """
        return abs(X-Y)/min(X,Y)

    def dm_channel_label(self,
        dm_mass,    # D+45 (49)
        stop_mass,  # D+56 (60)
        a0_mass,    # D+42 (46)
        sigma_ann,  # D+87 (91)
        stau_mass,  # D+61 (65)
        bino,       # D+72 (76)
    ):
        """
        Assigns one of five labels to the sample according to its DM
        annihilation channel.
            0) light chi            D+45
            1) stop coannihilation  
            2) A0-pole annihilation
            3) stau coannihilation
            4) well-tempered
            5) other
        """
        if (dm_mass < 70):
            return 0
        elif (self.diff(dm_mass, stop_mass) < 0.2):
            return 1
        elif (
            (self.diff(2*dm_mass, a0_mass) < 0.4) and
            (sigma_ann > 2e-27)
        ):
            return 2
        elif (self.diff(dm_mass, stau_mass) < 0.2):
            return 3
        elif (bino*bino < 0.9):
            return 4
        else:
            return 5

    def get_number_of_valid(self,
        output_file:    str='constrained_data',
        max_num_files:  int=10000,
        params:         dict=base_params,
        apply_higgs:    bool=True,
        apply_dm:       bool=True,
        apply_lsp:      bool=True,
    ):
        if (not apply_higgs and not apply_dm and not apply_lsp):
            self.logger.error(f"all constraints are set to False!")
        constraints = []
        if apply_higgs:
            constraints.append('higgs')
        if apply_dm:
            constraints.append('dm')
        if apply_lsp:
            constraints.append('lsp')
        constraint_dir = 'constraints/'
        for ii in range(len(constraints)):
            constraint_dir += constraints[ii]
            if ii < len(constraints)-1:
                constraint_dir += '_'
        if not os.path.isdir(constraint_dir):
            os.makedirs(constraint_dir)
        output = constraint_dir + "/" + output_file + '.csv'
        filenames = np.array(os.listdir(self.input_dir))
        individual_files = []
        file_loop = tqdm(
            enumerate(filenames, 0), 
            total=max([len(filenames),max_num_files]), 
            leave=False,
            colour='green'
        )
        for ii, filename in file_loop:
            if len(individual_files) > max_num_files:
                break
            try:
                file_dataframe = pd.read_csv(
                    self.input_dir + "/" + filename, 
                    sep=',', 
                    header=None, 
                    names=self.search_columns, 
                    usecols=self.search_columns
                )
                individual_files += [
                    self.standard_model_constraints(
                        file_dataframe,
                        params=params,
                        apply_higgs=apply_higgs,
                        apply_dm=apply_dm,
                        apply_lsp=apply_lsp
                    )
                ]
            except Exception as e:
                self.logger.warning(f"file {filename} threw an error: {e}")
        concatenated_files = pd.concat(individual_files)
        return len(concatenated_files)

    def generate_constrained_dataset(self,
        output_file:    str='constrained_data',
        input_file:     str='',
        max_num_files:  int=10000,
        params:         dict=base_params,
        apply_higgs:    bool=True,
        apply_dm:       bool=True,
        apply_lsp:      bool=True,
    ):
        """
        Generates a single dataset with constrained parameter values.
        """
        self.logger.info(f"generating constrained dataset for {self.subspace} subspace.")
        if (not apply_higgs and not apply_dm and not apply_lsp):
            self.logger.error(f"all constraints are set to False!")
        constraints = []
        if apply_higgs:
            constraints.append('higgs')
        if apply_dm:
            constraints.append('dm')
        if apply_lsp:
            constraints.append('lsp')
        constraint_dir = 'constraints/'
        for ii in range(len(constraints)):
            constraint_dir += constraints[ii]
            if ii < len(constraints)-1:
                constraint_dir += '_'
        if not os.path.isdir(constraint_dir):
            os.makedirs(constraint_dir)
        output = constraint_dir + "/" + output_file + '.csv'
        if input_file != '':
            filenames = [input_file]
        else:
            filenames = np.array(os.listdir(self.input_dir))
        individual_files = []
        file_loop = tqdm(
            enumerate(filenames, 0), 
            total=max([len(filenames),max_num_files]), 
            leave=False,
            colour='green'
        )
        for ii, filename in file_loop:
            if len(individual_files) > max_num_files:
                break
            try:
                file_dataframe = pd.read_csv(
                    self.input_dir + "/" + filename, 
                    sep=',', 
                    header=None, 
                    names=self.search_columns, 
                    usecols=self.search_columns
                )
                individual_files += [
                    self.standard_model_constraints(
                        file_dataframe,
                        params=params,
                        apply_higgs=apply_higgs,
                        apply_dm=apply_dm,
                        apply_lsp=apply_lsp
                    )
                ]
            except Exception as e:
                self.logger.warning(f"file {filename} threw an error: {e}")
        concatenated_files = pd.concat(individual_files)
        concatenated_files.to_csv(
            output, 
            sep=',', 
            header=None, 
            index=False, 
            columns=self.search_columns
        )
        self.logger.info(f"successfully generated {len(concatenated_files)} constrained parameter values from {len(filenames)} files.")


    def generate_unconstrained_dataset(self,
        output_file:    str='unconstrained_data',
        input_file:     str='',
        max_num_files:  int=10000,
        params:         dict=base_params,
        apply_higgs:    bool=True,
        apply_dm:       bool=True,
        apply_lsp:      bool=True,
    ):  
        """
        Generates a single dataset with !constrained parameter values
        """
        self.logger.info(f"generating unconstrained dataset for {self.subspace} subspace.")
        if (not apply_higgs and not apply_dm and not apply_lsp):
            self.logger.error(f"all constraints are set to False!")
        constraints = []
        if apply_higgs:
            constraints.append('higgs')
        if apply_dm:
            constraints.append('dm')
        if apply_lsp:
            constraints.append('lsp')
        constraint_dir = 'constraints/'
        for ii in range(len(constraints)):
            constraint_dir += constraints[ii]
            if ii < len(constraints)-1:
                constraint_dir += '_'
        if not os.path.isdir(constraint_dir):
            os.makedirs(constraint_dir)
        output = constraint_dir + "/" + output_file + '.csv'
        # fill parameter values
        if input_file != '':
            filenames = [input_file]
        else:
            filenames = np.array(os.listdir(self.input_dir))
        individual_files = []
        file_loop = tqdm(
            enumerate(filenames, 0), 
            total=max([len(filenames),max_num_files]), 
            leave=False,
            colour='green'
        )
        for ii, filename in file_loop:
            if len(individual_files) > max_num_files:
                break
            try:
                file_dataframe = pd.read_csv(
                    self.input_dir + "/" + filename, 
                    sep=',', 
                    header=None, 
                    names=self.search_columns, 
                    usecols=self.search_columns
                )
                constraint_indices = self.standard_model_constraints(
                    file_dataframe,
                    params=params,
                    apply_higgs=apply_higgs,
                    apply_dm=apply_dm,
                    apply_lsp=apply_lsp
                ).index
                mask = [
                    True if ii not in constraint_indices else False for ii in range(len(file_dataframe))
                ]
                individual_files += [file_dataframe.iloc[mask]]
            except Exception as e:
                self.logger.warning(f"file {filename} threw an error: {e}")
        concatenated_files = pd.concat(individual_files)
        concatenated_files.to_csv(
            output, 
            sep=',', 
            header=None, 
            index=False, 
            columns=self.search_columns
        )
        self.logger.info(f"successfully generated {len(concatenated_files)} !constrained parameter values from {len(filenames)} files.")

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