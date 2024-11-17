from torch import Tensor
import torch
import numpy as np
from numpy import ndarray
import scanpy as sc
from PIL import Image
import scipy
from sklearn.decomposition import PCA
import pandas as pd
import anndata
from scipy.io import mmread
from torch.nn import functional as F
from .data_sourse import DigitsDataset


def read_mtx(filename, dtype='int32'):
    x = mmread(filename).astype(dtype)
    return x

def multi_one_hot(index_tensor, depth_list):
    one_hot_tensor = F.one_hot(index_tensor[:,0], num_classes=depth_list[0])
    for col in range(1, len(depth_list)):
        next_one_hot = F.one_hot(index_tensor[:,col], num_classes=depth_list[col])
        one_hot_tensor = torch.cat([one_hot_tensor, next_one_hot], 1)

    return one_hot_tensor

class SingleCellDataset(DigitsDataset):
    
    def __init__(self, data_name="SingleCell", raw_data:np.ndarray=None, label_batch:np.ndarray=None):
        self.data_name = data_name
        self.batch_class = [len(np.unique(label_batch[:, i])) for i in range(label_batch.shape[1])]

        data = raw_data.astype(np.float32)
        label_batch = label_batch

        sadata = anndata.AnnData(X=data)
        sc.pp.normalize_per_cell(sadata, counts_per_cell_after=1e4)
        sadata = sc.pp.log1p(sadata, copy=True)

        if data.shape[1] > 50 and data.shape[0] > 50:
            sc.tl.pca(sadata, n_comps=50)
            data = sadata.obsm['X_pca'].copy()
        else:
            data = sadata.X.copy()
        
        if self.batch_class == [1]:
            label_batch = np.concat([label_batch, label_batch], axis=1)
            self.batch_class = [len(np.unique(label_batch[:, i])) for i in range(label_batch.shape[1])]
            n_batch = self.batch_class
        else:
            n_batch = self.batch_class
        batchhot = multi_one_hot(torch.tensor(label_batch).long(), n_batch)

        self.data = torch.tensor(data)
        self.batchhot = batchhot.long()
        self.graphwithpca = False

class CommonDataset(DigitsDataset):
    def __init__(self, data: ndarray | Tensor):
        self.data_name = "Common"
        if len(data.shape) > 2:
            data = data.reshape((data.shape[0], -1))
        if isinstance(data, ndarray):
            data = Tensor(data)
        self.data = data
        self.graphwithpca = False