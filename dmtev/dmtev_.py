from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from lightning_fabric.utilities.seed import seed_everything
import numpy as np
import os
from os import PathLike
# from pytorch_lightning import Trainer
from lightning import Trainer
import torch
import logging
import umap

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
        _, _, lat3 = self.model(torch.tensor(X).float())
        return lat3.cpu().detach().numpy()

    def fit(self, X, y=None):
        if not isinstance(X, np.ndarray) and not isinstance(X, torch.Tensor):
            raise ValueError("X must be a numpy array or a torch tensor")

        self.model.adapt(X)
        self.trainer.fit(self.model)
        
    def transform(self, X):
        _, _, lat3 = self.model(X)
        return lat3.cpu().detach().numpy()
    
    def compare(self, X, plot=None):
        '''
        Compare the embeddings of DMT-EV, UMAP and TSNE
        
        '''
                
        if not isinstance(X, np.ndarray) and not isinstance(X, torch.Tensor):
            raise ValueError("X must be a numpy array or a torch tensor")

        logging.debug("Start DMT-EV")
        dmt_embedding = self.fit_transform(X)
        logging.debug("Start UMAP")
        umap_embedding = umap.UMAP().fit_transform(X)
        logging.debug("Start TSNE")
        tsne_embedding = TSNE().fit_transform(X)
        
        if plot:
            logging.debug("Plotting")
            if not isinstance(plot, str):
                raise ValueError("plot must be a path")
            if not os.path.exists(os.path.dirname(plot)):
                os.makedirs(os.path.dirname(plot))
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].scatter(dmt_embedding[:, 0], dmt_embedding[:, 1])
            ax[0].set_title("DMTEV")
            ax[1].scatter(umap_embedding[:, 0], umap_embedding[:, 1])
            ax[1].set_title("UMAP")
            ax[2].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1])
            ax[2].set_title("TSNE")
            plt.savefig(plot)
