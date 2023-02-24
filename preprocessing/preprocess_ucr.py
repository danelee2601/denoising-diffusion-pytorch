"""
`Dataset` (pytorch) class is defined.
"""
import os
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import get_root_dir


class DatasetImporter(object):
    """
    Import a dataset and store it in the instance.
    """
    def __init__(self, dirname, train_ratio: float = 0.7, data_scaling: bool = True, **kwargs):
        """
        :param data_scaling
        """
        # download_ucr_datasets()
        self.data_root = get_root_dir().joinpath("dataset")

        # fetch an entire dataset
        fnames = os.listdir(dirname)
        X = None
        for i, fname in enumerate(fnames):
            print(f'Data loading... {round(i / len(fnames) * 100)}%') if i % 100 == 0 else None
            x = pd.read_csv(os.path.join(dirname, fname), header=None).values  # (h w)

            # create n channels for n categories
            h, w = x.shape
            unique_categories = np.unique(x)
            x_new = np.zeros((len(unique_categories), h, w))  # (c h w)
            for j, c in enumerate(unique_categories):
                x_new[j] = np.array(x == c, dtype=np.float)

            if i == 0:
                b = len(fnames)
                X = np.zeros((b, len(unique_categories), h, w))  # (b c h w)
            X[i] = x_new

        # split X into X_train and X_test
        self.X_train, self.X_test = train_test_split(X, train_size=train_ratio, random_state=0)

        if data_scaling:
            min_val = np.min(self.X_train)
            max_val = np.max(self.X_train)
            self.X_train = (self.X_train - min_val) / (max_val - min_val)
            self.X_test = (self.X_test - min_val) / (max_val - min_val)

        print('self.X_train.shape:', self.X_train.shape)
        print('self.X_test.shape:', self.X_test.shape)



class GeoDataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer: DatasetImporter,
                 **kwargs):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        self.kind = kind

        if kind == "train":
            self.X = dataset_importer.X_train  # (b c h w)
        elif kind == "test":
            self.X = dataset_importer.X_test  # (b c h w)
        else:
            raise ValueError

        self._len = self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :]  # (c h w)
        return x

    def __len__(self):
        return self._len


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    os.chdir("../")

    # data pipeline
    dataset_importer = DatasetImporter("dataset/facies")
    dataset = GeoDataset("train", dataset_importer)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)

    # get a mini-batch of samples
    for batch in data_loader:
        x = batch
        break
    print('x.shape:', x.shape)
