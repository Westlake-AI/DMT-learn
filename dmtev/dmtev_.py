from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

import os
from os import PathLike
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from lightning_fabric.utilities.seed import seed_everything
import torch

from .LitPatNN import LitPatNN


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

        self._validate_parameters()

        if device_id is not None and torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            self.trainer = Trainer(
                accelerator="gpu",
                devices=[device_id],
                max_epochs=epochs,
                logger=False,
                default_root_dir=checkpoint_path,
            )
            device = torch.device(f"cuda:{device_id}")
        else:
            self.trainer = Trainer(
                max_epochs=epochs,
                logger=False,
                default_root_dir=checkpoint_path,
            )
            device = torch.device("cpu")
        self.model = LitPatNN(device=device, epochs=epochs, **kwargs)

    def _validate_parameters(self):
        pass

    # 暂不支持
    def fit(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X=None, y=None):
        self.model.adapt(X)
        self.trainer.fit(self.model)
        return self.model.ins_emb

    # 暂不支持
    def transform(self, X):
        raise NotImplementedError