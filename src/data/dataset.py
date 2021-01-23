import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import utils
from pathlib import Path


class ChestXrayDataset(Dataset):
    def __init__(self,root_dir,train_val_split,transform=None):
        """
        :param root_dir: get train or test image list using glob
            then split them to train and validation using train_val_split
            PS: FOLDERS BASED ON LABEL NAMES
        :param transform: transforms to be applied on dataset
        """
        pass

    def __len__(self):
        """
        :return: returns length of dataset
        """
        pass

    def __getitem__(self, idx):
        """
        :param idx: next image index
        :return: return next image in line with its label
        """
        pass
