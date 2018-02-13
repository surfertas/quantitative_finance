# @author Tasuku Miura
# @brief Using Pytorch Dataset API to create data generators.

import os
import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn import preprocessing


class SequenceDataset(Dataset):
    """
    Custom dataset to convert features to inputs of sequences.
    """
    def __init__(self, csv_file, root_dir, time_steps, transform=None):
        self._steps = time_steps
        self._transform = transform
        self._frames = pd.read_csv(os.path.join(root_dir, csv_file))
        
        self._y = self._frames.pop("y")
        self._X = self._frames

    def __len__(self):
        return len(self._frames.index) - self._steps

    def __getitem__(self, idx):
        index = idx + self._steps
        # Normalize over sequence (TODO: Find a better way to handle)
        X = self._X.iloc[index-self._steps:index]
        # Get standardardize using the past "steps" samples.
        X_mean = X.apply(np.mean,axis=0)
        X_std = X.apply(np.std,axis=0)
        X_normalized = (X - X_mean)/(X_std+1e-10)
        sequence = torch.from_numpy(X_normalized.as_matrix())
        
        label = torch.from_numpy(np.array([self._y.iloc[index]]))
        return {'sequence': sequence, 'label': label}


class SequenceDatasetOneHot(Dataset):
    """
    Custom dataset to convert features to inputs of sequences.
    # y is a one hot vector indicating up, unched, down.
    # NOTE: Actually dont need to one hot, for PyTorch, as long as label is
    # class index!!!!!
    """
    def __init__(self, csv_file, root_dir, time_steps, transform=None):
        self._steps = time_steps
        self._transform = transform
        self._frames = pd.read_csv(os.path.join(root_dir, csv_file))
        
        self._y = self._get_one_hot()
        self._X = self._frames

    def _get_one_hot(self):
        y = self._frames.pop("y").values.reshape(-1,1)
        enc = preprocessing.OneHotEncoder()
        enc.fit(y)
        df = pd.DataFrame(enc.transform(y).toarray())
        assert(df.shape[0]==y.shape[0])
        return df

    def __len__(self):
        return len(self._frames.index) - self._steps

    def __getitem__(self, idx):
        index = idx + self._steps
        # Normalize over sequence (TODO: Find a better way to handle)
        X = self._X.iloc[index-self._steps:index]
        # Get standardardize using the past "steps" samples.
        X_mean = X.apply(np.mean,axis=0)
        X_std = X.apply(np.std,axis=0)
        X_normalized = (X - X_mean)/(X_std+1e-10)
        sequence = torch.from_numpy(X_normalized.as_matrix())
        
        label = torch.from_numpy(np.array([self._y.iloc[index]]))
        return {'sequence': sequence, 'label': label}

