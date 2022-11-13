"""
Class for a generic dataset.
"""
import numpy as np
import os
import torch
from torch.utils.data import Dataset

# melange includes
from chunc.utils.logger import Logger

generic_dataset_meta = {
    "who_created":  "noone",
    "when_created": "never",
    "where_created":"nowhere",
    "num_events":   0,
    "features":     {},
    "classes":      {},
}

required_dataset_arrays = {
    'meta',
    'event_features',
    'event_classes',
}

class CHUNCDataset(Dataset):
    """
    CHUNC datatype class.  This class contains an n-dimensional
    array of values along with labels describing what the values
    represent.

    Datasets are stored as numpy arrays with two main branches,
    'meta' and 'events'.  Meta stores information about the dataset
    such as when is was created and who created it.  Events is
    broken down into individual feature columns.

    There may be additional arrays in the file, such as those
    containing event weights, class weights, or other items.

    The 'meta' block should contain at least the following:
        - who_created
        - when_created
        - where_created
        - num_events
        - features
        - classes
    """
    def __init__(self,
        name:   str,
        input_file: str,
        features:   list=None,
        classes:    list=None,
        sample_weights: str=None,
        class_weights:  str=None,
        normalized:  bool=True,
    ):
        # wether to fix broken meta or events
        self.replace_meta = False
        self.replace_events = False
        # basic information for this dataset
        self.name = name
        self.logger = Logger(self.name, file_mode='w')
        self.logger.info(f"constructing dataset.")
        self.features = features
        self.classes = classes
        self.sample_weights = sample_weights
        self.class_weights = class_weights
        if sample_weights != None:
            self.use_sample_weights = True
        else:
            self.use_sample_weights = False
        if class_weights != None:
            self.use_class_weights = True
        else:
            self.use_class_weights = False
        self.normalized = normalized
        self.logger.info(f"setting 'features': {self.features}.")
        self.logger.info(f"setting 'classes': {self.classes}.")
        self.logger.info(f"setting 'sample_weights': {self.sample_weights}.")
        self.logger.info(f"setting 'class_weights': {self.class_weights}.")
        self.logger.info(f"setting 'use_sample_weights': {self.use_sample_weights}.")
        self.logger.info(f"setting 'use_class_weights': {self.use_class_weights}.")
        self.logger.info(f"setting 'normalize': {self.normalized}.")

        # first check that file exists
        self.input_file = input_file
        if not os.path.isfile(self.input_file):
            self.logger.error(f"input file '{input_file}' does not exist.")
        
        # create the torch dataset
        self.data = np.load(input_file, allow_pickle=True)

        # check that 'meta', 'event_features' and 'event_classes' are in the file
        for item in required_dataset_arrays:
            if item not in self.data.keys():
                self.logger.error(f"item '{item}' not in dictionary of input file {input_file} ({self.data.keys()}).")

        # collect meta data
        self.meta = self.data['meta'].item()
        # check that meta data is present
        for item in generic_dataset_meta:
            if (item not in self.meta):
                self.logger.error(f"item {item} not in meta dictionary of input file {input_file}")
        
        # grab meta info
        self.num_events     = self.meta['num_events']
        self.meta_features  = self.meta['features']
        self.meta_classes   = self.meta['classes']
        if self.use_sample_weights:
            if 'sample_weights' not in self.meta.keys():
                self.logger.error(f"item 'sample_weights' not in dictionary of input file {input_file} ({self.data.keys()}).")
            self.meta_sample_weights = self.meta['sample_weights']
        if self.use_class_weights:
            if 'class_weights' not in self.meta.keys():
                self.logger.error(f"item 'class_weights' not in dictionary of input file {input_file} ({self.data.keys()}).")
            self.meta_class_weights = self.meta['class_weights']

        # collect event_features
        self.event_features = self.data['event_features']
        # checks for consistency with meta and events
        if len(self.event_features) != self.num_events:
            self.logger.warning(f"inconsistency between 'meta' num_events: {self.num_events} and length of 'events' {len(self.event_features)}")
        if (self.features == None):
            self.logger.warning(f"no input features given, using all features from input file {input_file} ({self.meta_features}).")
            self.features = self.meta_features    
        self.logger.info(f"attempting to load features {self.features}")
        for item in self.features:
            if (item not in self.meta_features.keys()):
                self.logger.error(f"feature {item} not in events dictionary of input file {input_file}")
        self.logger.info(f"successfully loaded features {self.features}")
        # check that features are an array of arrays
        if len(self.event_features.shape) <= 1:
            self.logger.warning(f"features array may have an incompatible shape ({self.event_features.shape})!")

        # collect classes
        self.event_classes = self.data['event_classes']
        # checks for consistency with meta and classes
        if len(self.event_classes) != self.num_events:
            self.logger.warning(f"inconsistency between 'meta' num_events: {self.num_events} and length of 'classes' {len(self.event_classes)}")
        if (self.classes == None):
            self.logger.warning(f"no input classes given, using all classes from input file {input_file}")
            self.classes = self.meta_classes
        self.logger.info(f"attempting to load classes {self.classes}")
        for item in self.classes:
            if (item not in self.meta_classes.keys()):
                self.logger.error(f"class {item} not in events dictionary of input file {input_file}")
        self.logger.info(f"successfully loaded classes {self.classes}")
        if len(self.event_classes.shape) <= 1:
            self.logger.warning(f"classes array may have an incompatible shape ({self.event_classes.shape})!")

        # construct feature indices
        self.feature_idx = [self.meta_features[item] for item in self.features]
        self.class_idx = [self.meta_classes[item] for item in self.classes]

        # determine if sample weights are available
        if not self.use_sample_weights:
            self.sample_weights = None
        else:
            if 'event_sample_weights' not in self.data.keys():
                self.logger.error(f"cannot use sample_weights since it is not defined in data!")
            self.event_sample_weights = self.data['event_sample_weights']
            # check that sample_weights is defined in meta
            self.meta_sample_weights = self.meta['sample_weights']
            self.logger.info(f"attempting to load sample weights {self.sample_weights}")
            if (self.sample_weights not in self.meta_sample_weights.keys()):
                self.logger.error(f"class {self.sample_weights} not in sample_weights dictionary of input file {input_file}")
            self.sample_weight_idx = self.meta_sample_weights[self.sample_weights]
            self.logger.info(f"successfully loaded sample weights {self.sample_weights}")

        # determine if class weights are available
        if not self.use_class_weights:
            self.class_weights = None
        else:
            if 'event_class_weights' not in self.data.keys():
                self.logger.error(f"cannot use class_weights since it is not defined in data!")
            self.event_class_weights = self.data['event_class_weights']
            # check that class_weights is defined in meta
            self.meta_class_weights = self.meta['class_weights']
            self.logger.info(f"attempting to load class weights {self.class_weights}")
            if (self.class_weights not in self.meta_class_weights.keys()):
                self.logger.error(f"class {self.class_weights} not in class_weights dictionary of input file {input_file}")
            self.class_weight_idx = self.meta_class_weights[self.class_weights]
            self.logger.info(f"successfully loaded class weights {self.class_weights}")

        # construct the normalization
        if self.normalize:
            if 'means' in self.meta.keys():
                self.means = self.meta['means']
                # check that all features are there
                for item in self.features:
                    if item not in self.means.keys():
                        self.replace_meta = True
                        self.logger.warning(f"feature {item} is not present in list of means, calculating mean value")
                        self.means[item] = np.mean(self.event_features[:,self.meta_features[item]])
                        self.meta['means'][item] = self.means[item]
            else:
                self.replace_meta = True
                self.logger.info(f"means information is not stored in meta, calculating means.")
                self.means = {}
                for item in self.features:
                    self.means[item] = np.mean(self.event_features[:,self.meta_features[item]])
                self.meta['means'] = self.means

            if 'stds' in self.meta.keys():
                self.stds = self.meta['stds']
                # check that all features are there
                for item in self.features:
                    if item not in self.stds.keys():
                        self.replace_meta = True
                        self.logger.warning(f"feature {item} is not present in list of stds, calculating std value")
                        self.stds[item] = np.std(self.event_features[:,self.meta_features[item]])
                        self.meta['stds'][item] = self.stds[item]
            else:
                self.replace_meta = True
                self.logger.info(f"stds information is not stored in meta, calculating stds.")
                self.stds = {}
                for item in self.features:
                    self.stds[item] = np.std(self.event_features[:,self.meta_features[item]])
                self.meta['stds'] = self.stds
            self.feature_means = [self.means[item] for item in self.features]
            self.feature_stds  = [self.stds[item] for item in self.features]

        
        # turn arrays into torch tensors
        self.event_features = torch.tensor(self.event_features, dtype=torch.float)
        if self.normalize:
            self.feature_means = torch.tensor(self.feature_means, dtype=torch.float)
            self.feature_stds  = torch.tensor(self.feature_stds, dtype=torch.float)
        self.event_classes = torch.tensor(self.event_classes, dtype=torch.float)
        if self.use_sample_weights:
            self.event_sample_weights = torch.tensor(self.event_sample_weights, dtype=torch.float)
        if self.use_class_weights:
            self.event_class_weights = torch.tensor(self.event_class_weights, dtype=torch.float)
        
        self.feature_shape = self.event_features[0].shape
        self.class_shape = self.event_classes[0].shape
        if self.use_sample_weights:
            self.event_sample_weights_shape = self.event_sample_weights[0].shape
        else:
            self.event_sample_weights_shape = None
        if self.use_class_weights:
            self.event_class_weights_shape = self.event_class_weights[0].shape
        else:
            self.event_class_weights_shape = None

        # TODO: Simplify editing the npz file

    def attach(self,
        device,
    ):
        self.event_features = self.event_features.to(device)
        if self.normalized:
            self.feature_means = self.feature_means.to(device)
            self.feature_stds = self.feature_stds.to(device)
        self.event_classes = self.event_classes.to(device)
        if self.use_sample_weights:
            self.event_sample_weights = self.event_sample_weights.to(device)
        if self.use_class_weights:
            self.event_class_weights = self.event_class_weights.to(device)

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        x = self.event_features[idx, self.feature_idx]
        if self.normalized:
            x = (x - self.feature_means)/self.feature_stds
        y = self.event_classes[idx, self.class_idx]
        if not self.use_sample_weights:
            return x, y 
        else:
            z = self.event_sample_weights[idx, self.sample_weight_idx]
            return x, y, z
    
    def feature(self,
        feature:        str,
    ):
        if feature not in self.features:
            self.logger.error(f"attempting to access feature {feature} which is not in 'features': {self.features}")
        return self.event_features[:, self.meta_features[feature]]
    
    def normalize(self,
        x
    ):
        return (x - self.feature_means)/self.feature_stds

    def unnormalize(self,
        x,
        detach=False
    ):
        x = self.feature_means + self.feature_stds * x
        if detach:
            return x.detach().numpy()
        else:
            return x