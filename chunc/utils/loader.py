"""
Generic data loader class for tpc_ml.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset, random_split

from chunc.utils.logger import Logger

class Loader:
    """
    """
    def __init__(self,
        dataset:    Dataset,
        batch_size: int,
        test_split:         float=0.2,
        test_seed:          int=-1,
        validation_split:   float=0.0,
        validation_seed:    int=-1,
        num_workers:        int=0,
    ):
        self.name = dataset.name + "_loader"
        self.logger = Logger(self.name, file_mode='w')
        self.logger.info(f"constructing dataset loader.")
        # check that dataset is either a torch Dataset or
        # inherited from GenericDataset
        if not issubclass(type(dataset), Dataset):
            if not issubclass(type(dataset), Dataset):
                self.logger.error(f"specified dataset '{dataset}' is neither an instance of {Dataset} nor {GenericDataset}!\nAny dataset must be inherited from either of those two classes.")
            else:
                self.logger.warning(f"specified dataset is inherited from {Dataset} rather than {GenericDataset}, which may lead to unexpected behavior.")
        # check for parameter compatability
        if batch_size <= 0:
            self.logger.error(f"specified batch size: {batch_size} not allowed, must be > 0!")
        if (validation_split < 0.0 or validation_split >= 1.0):
            self.logger.error(f"specified validation split: {validation_split} not allowed, must be 0.0 <= 'validation_split' < 1.0!")
        if (test_split < 0.0 or test_split >= 1.0):
            self.logger.error(f"specified test split: {test_split} not allowed, must be 0.0 <= 'test_split' < 1.0!")
        if (test_seed != -1 and test_seed < 0):
            self.logger.error(f"specified test seed: {test_seed} not allowed, must be == -1 or >= 0!")
        if not isinstance(test_seed, int):
            self.logger.error(f"specified test seed: {test_seed} is of type '{type(test_seed)}', must be of type 'int'!")
        if (validation_seed != -1 and validation_seed < 0):
            self.logger.error(f"specified validation seed: {validation_seed} not allowed, must be == -1 or >= 0!")
        if not isinstance(validation_seed, int):
            self.logger.error(f"specified validation seed: {validation_seed} is of type '{type(validation_seed)}', must be of type 'int'!")

        # assign parameters
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_split = test_split
        self.test_seed = test_seed
        self.validation_split = validation_split
        self.validation_seed = validation_seed
        self.num_workers = num_workers

        # record values
        self.logger.info(f"batch_size:  {self.batch_size}.")
        self.logger.info(f"test_split:  {self.test_split}.")
        self.logger.info(f"test_seed:   {self.test_seed}.")
        self.logger.info(f"validation_split: {self.validation_split}.")
        self.logger.info(f"validation_seed: {self.validation_seed}.")
        self.logger.info(f"num_workers: {self.num_workers}.")

        # determine if using sample weights
        if self.dataset.use_sample_weights == True:
            self.use_sample_weights = True
        else:
            self.use_sample_weights = False
        self.logger.info(f"use_sample_weights: {self.use_sample_weights}.")
        
        # determine if using class weights
        if self.dataset.use_class_weights == True:
            self.use_class_weights = True
        else:
            self.use_class_weights = False
        self.logger.info(f"use_class_weights: {self.use_class_weights}.")

        # determine number of all batches
        self.num_all_batches = int(len(self.dataset)/self.batch_size)
        if len(self.dataset) % self.batch_size != 0:
            self.num_all_batches += 1
        self.logger.info(f"number of total samples: {len(self.dataset)}.")
        self.logger.info(f"number of all batches: {self.num_all_batches}.")

        # determine number of training/testing samples
        self.num_total_train = int(len(self.dataset) * (1 - self.test_split))
        self.num_test  = int(len(self.dataset) - self.num_total_train)
        # determine number of batches for testing
        self.num_test_batches = int(self.num_test/self.batch_size)
        if self.num_test % self.batch_size != 0:
            self.num_test_batches += 1
        self.logger.info(f"number of total training samples: {self.num_total_train}")
        self.logger.info(f"number of test samples: {self.num_test}")
        self.logger.info(f"number of test batches: {self.num_test_batches}")

        # determine number of training/validation samples
        self.num_train  = int(self.num_total_train * (1 - self.validation_split))
        self.num_validation    = int(self.num_total_train - self.num_train)
        # determine number of batches for training/validation
        self.num_train_batches = int(self.num_train/self.batch_size)
        if self.num_train % self.batch_size != 0:
            self.num_train_batches += 1
        self.num_validation_batches   = int(self.num_validation/self.batch_size)
        if self.num_validation % self.batch_size != 0:
            self.num_validation_batches += 1
        self.logger.info(f"number of training samples: {self.num_train}")
        self.logger.info(f"number of training batches per epoch: {self.num_train_batches}")
        self.logger.info(f"number of validation samples: {self.num_validation}")
        self.logger.info(f"number of validation batches per epoch: {self.num_validation_batches}")

        # set up the training and testing sets
        if self.test_seed != -1:
            # create two split datasets
            self.total_train, self.test = random_split(
                dataset=self.dataset, 
                lengths=[self.num_total_train, self.num_test],
                generator=torch.Generator().manual_seed(self.test_seed)
            )
            self.total_train_indices = self.total_train.indices
            self.test_indices = self.test.indices
            self.logger.info(f"created train/test split with random seed: {self.test_seed}.")
        else:
            self.total_train_indices = range(self.num_total_train)
            self.test_indices = range(self.num_total_train, len(self.dataset))

            self.total_train = Subset(self.dataset, self.total_train_indices)
            self.test = Subset(self.dataset, self.test_indices)
            self.logger.info(f"created train/test split with first {self.num_total_train} samples for training and last {self.num_test} samples for testing.")
        # set up the training and validation sets
        if self.validation_seed != -1:
            # create train/validation split datasets
            self.train, self.validation = random_split(
                dataset=self.total_train, 
                lengths=[self.num_train, self.num_validation],
                generator=torch.Generator().manual_seed(self.validation_seed)
            )
            self.train_indices = self.train.indices
            self.validation_indices = self.validation.indices
            self.logger.info(f"created train/validation split with random seed: {self.validation_seed}.")
        else:
            self.train_indices = range(self.num_train)
            self.validation_indices = range(self.num_train, len(self.total_train))

            self.train = Subset(self.total_train, self.train_indices)
            self.validation = Subset(self.total_train, self.validation_indices)
            self.logger.info(f"created train/validation split with first {self.num_train} samples for training and last {self.num_validation} samples for validation.")
        self.all_indices = range(len(self.dataset))
        
        # set up dataloaders for each set
        self.train_loader = DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            pin_memory=True,
            num_workers=self.num_workers
        )
        self.validation_loader = DataLoader(
            self.validation, 
            batch_size=self.batch_size, 
            pin_memory=True,
            num_workers=self.num_workers
        )
        self.test_loader = DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            pin_memory=True,
            num_workers=self.num_workers
        )
        self.all_loader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            pin_memory=True,
            num_workers=self.num_workers
        )
        self.inference_loader = DataLoader(
            self.dataset, 
            batch_size=1, 
            pin_memory=True,
            num_workers=self.num_workers
        )