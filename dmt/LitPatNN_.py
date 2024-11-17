import functools
from functools import reduce
import numpy as np
from numpy import ndarray
import os
from pytorch_lightning import LightningModule
import torch
from torch import nn, device
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from time import sleep
import logging

from .aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn
from .dataloader import data_base
from .dataloader.data_base import SingleCellDataset
from .loss import dmt_loss as dmt_loss_aug
from .model.model import NN_FCBNRL_MM
from .utils_ import gpu2np
from .manifolds.hyperbolic_project import ToEuclidean, ToPoincare, ToLorentz


class LitPatNN(LightningModule):
    def __init__(
        self,
        dataname:str="SingleCell",
        device:device=device('cpu'),
        # model param
        nu:float=1e-2,
        num_fea_aim:int=-1,
        K:int=5,
        Uniform_t:float=0.5,  # 0.3
        lr:float=5e-4,
        # trainer param
        batch_size:int=2000,
        epochs:int=1500, # 1500
        log_interval:int=100, # 300
        # set param
        metric:str="euclidean",
        manifold:str='Euclidean',
        detaalpha:float=1.001,
        l2alpha:float=10,
        num_fea_per_pat:int=100,  # 0.5
        Bernoulli_t:float=-1,
        Normal_t:float=-1,
        # train param
        NetworkStructure_1:list=[-1, 500, 300],
        NetworkStructure_2:list=[-1, 300, 100],
        augNearRate:float=1000,
        batchRate:float=100,
        ):

        super().__init__()

        # Set our init args as class attributes
        self.dataname = dataname
        self.l2alpha = l2alpha
        self.nu = nu
        self.num_fea_aim = num_fea_aim
        self.num_fea_per_pat = num_fea_per_pat
        self.K = K
        self.Uniform_t = Uniform_t
        self.Bernoulli_t = Bernoulli_t
        self.Normal_t = Normal_t
        self.NetworkStructure_1 = NetworkStructure_1
        self.NetworkStructure_2 = NetworkStructure_2
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval = min(log_interval, self.epochs)
        self.lr = lr
        self.my_device = device

        self.num_latent_dim = 2
        self.t = 0.1
        self.alpha = None
        self.stop = False
        self.detaalpha = detaalpha
        self.bestval = 0
        self.aim_cluster = None
        self.importance = None

        # choose manifold
        self.rie_pro_input = ToEuclidean()
        if manifold == 'Euclidean':
            self.metric_e = 'euclidean'
            self.rie_pro_latent = ToEuclidean()
        if manifold == 'PoincareBall':
            self.metric_e = 'poin_dist_mobiusm_v2'
            self.rie_pro_latent = ToPoincare()
        if manifold == 'Hyperboloid':
            self.num_latent_dim += 1
            self.metric_e = 'lor_dist_v2'
            self.rie_pro_latent = ToLorentz()

        self.uselabel = False  # only unsupervised for now
        self.mse = torch.nn.CrossEntropyLoss()
        self.loss_eye = torch.eye(batch_size).to(self.my_device)
        self.Loss = dmt_loss_aug.MyLoss(
            v_input=100,
            metric=metric,
            augNearRate=augNearRate,
            batchRate=batchRate,
            device=self.my_device,
        ).to(self.my_device)
        
        
    def setup(self, stage=None):
        logging.debug(f"stage: {stage}")
        # import pdb; pdb.set_trace()
        if stage == "fit" and (self.data_train is None or self.data_test is None):
            raise ValueError("Data not loaded")


    def adapt(self, data_path:str|ndarray=None, label_batch:ndarray=None):
        dataset_f = getattr(data_base, "CSVDataset") if isinstance(data_path, str) else SingleCellDataset
        if self.dataname == "SingleCell":
            self.data_train = dataset_f(
                data_name=self.dataname,
                raw_data=data_path,
                label_batch=label_batch,
            )
            # self.data_test = dataset_f(
            #     data_name=self.dataname,
            #     raw_data=data_path,
            #     label_batch=label_batch,
            # )
            self.data_test = self.data_train

        else:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data path {data_path} not found")
            self.data_train = dataset_f(
                data_name=self.dataname,
                train=True,
                datapath=data_path,
            )
            self.data_test = dataset_f(
                data_name=self.dataname,
                train=True,
                datapath=data_path,
            )

        if len(self.data_train.data) < self.batch_size:
            logging.debug(f"Batch size is larger than data size, eye size {self.batch_size} -> {self.data_train.data.shape[0]}")
            self.batch_size = self.data_train.data.shape[0]
            self.loss_eye = torch.eye(self.batch_size).to(self.my_device)

        if len(self.data_train.data.shape) == 2:
            self.data_train.cal_near_index(
                device=self.my_device,
                k=self.K,
                uselabel=bool(self.uselabel),
            )

        logging.debug("my_device: {}".format(self.my_device))
        self.data_train.to_device(self.my_device)
        self.data_test.to_device(self.my_device)

        self.dims = self.data_train.get_dim()
        
        # adopt the network structure to the data
        if self.num_fea_aim == -1:
            self.num_fea_aim = self.data_train.data.shape[1]
        self.num_fea_aim = min(
            self.num_fea_aim, reduce(lambda x, y: x*y, self.data_train.data.shape[1:]) 
        )
        logging.debug(f"num_fea_aim: {self.num_fea_aim}")

        if len(self.data_train.data.shape) > 2:
            self.transforms = transforms.AutoAugment(
                transforms.AutoAugmentPolicy.CIFAR10
            )

        self.fea_num = 1
        for i in range(len(self.data_train.data.shape) - 1):
            self.fea_num = self.fea_num * self.data_train.data.shape[i + 1]

        logging.debug(f"fea_num: {self.fea_num}")
        
        self.model_pat, self.model_b = self.InitNetworkMLP(
            self.NetworkStructure_1,
            self.NetworkStructure_2,
        )


    def forward_fea(self, x):
        lat = x
        lat1 = self.model_pat(lat)
        lat3 = lat1
        for i, m in enumerate(self.model_b):
            lat3 = m(lat3)
        return lat1, lat1, lat3
    
    def forward(self, x):
        return self.forward_fea(x)

    def predict(self, x):
        x = torch.tensor(x.to_numpy())
        return gpu2np(self.forward_simi(x))


    def forward_simi(self, x):
        x = torch.tensor(x).to(self.mask.device)
        out = self.forward_fea(x)[2]
        dis = torch.norm(out - torch.tensor(self.cf_aim).to(x.device), dim=1)
        return torch.exp(-1 * dis).reshape(-1)

    def training_step(self, batch, batch_idx):
        data, batchhot, index = batch
        data1 = data
        batchhot1 = batchhot
        # data1 = self.data_train.data[index]
        # batchhot1 = self.data_train.batchhot[index]
        data2, batchhot2 = self.augmentation_warper(index, data1, batchhot1)
        data = torch.cat([data1, data2])
        data = data.reshape(data.shape[0], -1)
        batchhot = torch.cat((batchhot, batchhot2))

        # forward
        pat, mid, lat = self(data)

        # projection
        lat = self.rie_pro_latent(lat)

        # loss
        loss_topo = self.Loss(
            input_data=mid.reshape(mid.shape[0], -1),
            latent_data=lat.reshape(lat.shape[0], -1),
            batchhot = batchhot,
            v_latent=self.nu,
            metric=self.metric_e,
        )

        return loss_topo

    def validation_step(self, batch, batch_idx):
        data, batchhot, index = batch
        # augmentation
        if (self.current_epoch + 1) % self.log_interval == 0:
            index = index.to(self.device)
            data = self.data_train.data[index]
            data = data.reshape(data.shape[0], -1)
            pat, mid, lat = self(data)

            return (
                gpu2np(data),
                gpu2np(pat),
                gpu2np(lat),
                gpu2np(index),
            )

    def Cal_Sparse_loss(self, PatM):
        loss_l2 = torch.abs(PatM).mean()
        return loss_l2


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-9
        )
        self.scheduler = StepLR(
            optimizer, step_size=self.epochs // 10, gamma=0.8
        )
        return [optimizer], [self.scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            drop_last=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def InitNetworkMLP(self, NetworkStructure_1, NetworkStructure_2):

        num_fea_per_pat = self.num_fea_per_pat
        struc_model_pat = (
            [functools.reduce(lambda x, y: x * y, self.dims)]
            + NetworkStructure_1[1:]
            + [num_fea_per_pat]
        )
        struc_model_b = NetworkStructure_2 + [self.num_latent_dim]
        struc_model_b[0] = num_fea_per_pat

        m_l = []
        for i in range(len(struc_model_pat) - 1):
            m_l.append(
                NN_FCBNRL_MM(
                    struc_model_pat[i],
                    struc_model_pat[i + 1],
                )
            )
        model_pat = nn.Sequential(*m_l)

        model_b = nn.ModuleList()
        for i in range(len(struc_model_b) - 1):
            if i != len(struc_model_b) - 2:
                model_b.append(NN_FCBNRL_MM(struc_model_b[i], struc_model_b[i + 1]))
            else:
                model_b.append(
                    NN_FCBNRL_MM(struc_model_b[i], struc_model_b[i + 1], use_RL=False)
                )

        # logging.debug(model_pat)
        # logging.debug(model_b)
        model_pat = model_pat.to(self.my_device)
        model_b = model_b.to(self.my_device)
        return model_pat, model_b

    def augmentation_warper(self, index, data1, batchhot1):
        if len(data1.shape) == 2:
            return self.augmentation(index, data1, batchhot1)
        else:
            return self.augmentation_img(index, data1)

    def augmentation_img(self, index, data):
        # aug = []
        # for i in range(data.shape[0]):
        #     aug.append(
        #         self.transforms(data.permute(0,3,1,2)).reshape(1,-1)
        #         )
        return self.transforms(data.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def augmentation(self, index, data1, batchhot1):
        data2_list = []
        batchhot2_list = []
        if self.Uniform_t > 0:
            data_new, batch_hot_new = aug_near_mix(
                index,
                self.data_train,
                k=self.K,
                random_t=self.Uniform_t,
                device=self.device,
            )
            data2_list.append(data_new)
            batchhot2_list.append(batch_hot_new)
        if self.Bernoulli_t > 0:
            data_new = aug_near_feautee_change(
                index,
                self.data_train,
                k=self.K,
                t=self.Bernoulli_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if self.Normal_t > 0:
            data_new = aug_randn(
                index,
                self.data_train,
                k=self.K,
                t=self.Normal_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if (
            max(
                [
                    self.Uniform_t,
                    self.Normal_t,
                    self.Bernoulli_t,
                ]
            )
            < 0
        ):
            data_new = data1
            batchhot_new = batchhot1
            data2_list.append(data_new)
            batchhot2_list.append(batchhot_new)

        if len(data2_list) == 1:
            data2 = data2_list[0]
            batchhot2 = batchhot2_list[0]
        elif len(data2_list) == 2:
            data2 = (data2_list[0] + data2_list[1]) / 2
            batchhot2 = (batchhot2_list[0] + batchhot2_list[1]) / 2
        elif len(data2_list) == 3:
            data2 = (data2_list[0] + data2_list[1] + data2_list[2]) / 3
            batchhot2 = (batchhot2_list[0] + batchhot2_list[1] + batchhot2_list[2]) / 3

        return data2, batchhot2
