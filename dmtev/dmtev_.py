from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from lightning_fabric.utilities.seed import seed_everything
import numpy as np
import os
from os import PathLike
from pytorch_lightning import Trainer
import torch
import logging

from .LitPatNN_ import LitPatNN


class DMTEV(BaseEstimator):
    def __init__(self,
                 seed:int=1,
                 epochs:int=1500,
                 device_id:int|None=None,
                 checkpoint_path: PathLike|None="./",
                 **kwargs) -> None:
        super().__init__()
        seed_everything(seed)
        os.makedirs(checkpoint_path, exist_ok=True)

        self._validate_parameters(epochs=epochs, **kwargs)

        if device_id is not None and device_id >= 0 and torch.cuda.is_available():
            logging.debug("Using GPU")
            torch.set_float32_matmul_precision('high')
            self.trainer = Trainer(
                accelerator="gpu",
                devices=[device_id],
                max_epochs=epochs,
                logger=False,
                default_root_dir=checkpoint_path,
                checkpoint_callback=False,
                enable_checkpointing=False,
            )
            device = torch.device(f"cuda:{device_id}")
        else:
            logging.debug("Using CPU")
            self.trainer = Trainer(
                max_epochs=epochs,
                logger=False,
                default_root_dir=checkpoint_path,
            )
            device = torch.device("cpu")
        self.model = LitPatNN(device=device, epochs=epochs, **kwargs)

    def _validate_parameters(self, **kwargs):
        if kwargs['epochs'] < 10:
            raise ValueError("epochs must be greater than or equal to 10")
        if kwargs['num_fea_aim'] < -1:
            raise ValueError("num_fea_aim must be greater than or equal to -1")
        if 'metric' in kwargs.keys() and kwargs['metric'] not in ['euclidean', 'cossim']:
            raise ValueError("metric must be 'euclidean' or 'cossim'")

    def fit_transform(self, X:np.ndarray|torch.Tensor) -> np.ndarray:
        if not isinstance(X, np.ndarray) and not isinstance(X, torch.Tensor):
            raise ValueError("X must be a numpy array or a torch tensor")

        self.model.adapt(X)
        self.trainer.fit(self.model)
        return self.model.ins_emb

    # 暂不支持
    def fit(self, X, y=None):
        if not isinstance(X, np.ndarray) and not isinstance(X, torch.Tensor):
            raise ValueError("X must be a numpy array or a torch tensor")

        self.model.adapt(X)
        self.trainer.fit(self.model)
        

    # 暂不支持
    def transform(self, X):
        _, _, lat3 = self.model(X)
        return lat3.cpu().detach().numpy()