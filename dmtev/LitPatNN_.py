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

# import Loss.dmt_loss_aug as dmt_loss_aug1
from .aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn
from .dataloader import data_base
from .dataloader.data_base import CommonDataset
from .model.model import NN_FCBNRL_MM
from .Loss import dmt_loss_aug2 as dmt_loss_aug
from .utils_ import gpu2np


class LitPatNN(LightningModule):
    def __init__(
        self,
        dataname:str="Common",
        device:device=device('cpu'),
        # model param
        metric:str="euclidean",
        detaalpha:float=1.001,
        l2alpha:float=10,
        nu:float=1e-2,
        num_fea_aim:int=64,
        num_fea_per_pat:int=80,  # 0.5
        K:int=5,
        Uniform_t:float=1,  # 0.3
        Bernoulli_t:float=-1,
        Normal_t:float=-1,
        # train param
        NetworkStructure_1:list=[-1, 200] + [200] * 5,
        NetworkStructure_2:list=[-1, 500, 80],
        augNearRate:float=1000,
        # trainer param
        log_interval:int=300,
        batch_size:int=1000,
        epochs:int=1500,
        lr:float=1e-3,
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

        self.t = 0.1
        self.alpha = None
        self.stop = False
        self.detaalpha = detaalpha
        self.bestval = 0
        self.aim_cluster = None
        self.importance = None

        
        self.uselabel = False  # only unsupervised for now
        self.mse = torch.nn.CrossEntropyLoss()
        self.loss_eye = torch.eye(batch_size).to(self.my_device)
        self.Loss = dmt_loss_aug.MyLoss(
            v_input=100,
            metric=metric,
            augNearRate=augNearRate,
            device=self.my_device,
        ).to(self.my_device)
        
        
    def setup(self, stage=None):
        logging.debug(f"stage: {stage}")
        # import pdb; pdb.set_trace()
        if stage == "fit" and (self.data_train is None or self.data_test is None):
            raise ValueError("Data not loaded")


    def adapt(self, data_path:str|ndarray=None):
        dataset_f = getattr(data_base, "CSVDataset") if isinstance(data_path, str) else CommonDataset
        if self.dataname == "Common":
            self.data_train = dataset_f(
                data=data_path,
            )
            self.data_test = dataset_f(
                data=data_path,
            )

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

        if len(self.data_train.data.shape) == 2:
            self.data_train.cal_near_index(
                device=self.my_device,
                k=self.K,
                uselabel=bool(self.uselabel),
            )

        logging.info("my_device: {}".format(self.my_device))
        self.data_train.to_device(self.my_device)
        self.data_test.to_device(self.my_device)

        self.dims = self.data_train.get_dim()
        
        # adopt the network structure to the data
        self.num_fea_aim = min(
            self.num_fea_aim, reduce(lambda x, y: x*y, self.data_train.data.shape[1:]) 
        )
        logging.info(f"num_fea_aim: {self.num_fea_aim}")

        if len(self.data_train.data.shape) > 2:
            self.transforms = transforms.AutoAugment(
                transforms.AutoAugmentPolicy.CIFAR10
            )

        self.fea_num = 1
        for i in range(len(self.data_train.data.shape) - 1):
            self.fea_num = self.fea_num * self.data_train.data.shape[i + 1]

        logging.debug(f"fea_num: {self.fea_num}")
        self.PM_root = nn.Linear(self.fea_num, 1)
        self.PM_root.weight.data = torch.ones_like(self.PM_root.weight.data) / 5
        
        self.model_pat, self.model_b = self.InitNetworkMLP(
            # self.model_pat, self.model_b = self.InitNetworkMLP_OLD(
            self.NetworkStructure_1,
            self.NetworkStructure_2,
        )


    def forward_fea(self, x):
        # import pdb; pdb.set_trace()
        # lat = torch.zeros(x.shape).to(x.device)
        self.mask = self.PM_root.weight.reshape(-1) > 0.1
        # for i in range(self.hparams.num_pat):
        # if self.alpha is not None:
        #     # logging.debug('x.shape', x.shape)
        #     # logging.debug('self.PM_root.weight', self.PM_root.weight.shape)
        #     lat = x * ((self.PM_root.weight.reshape(-1)) * self.mask)
        # else:
        #     lat = x * ((self.PM_root.weight.reshape(-1)) * self.mask).detach()
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
        index = batch.to(self.device)
        # augmentation
        data1 = self.data_train.data[index]
        data2 = self.augmentation_warper(index, data1)
        data = torch.cat([data1, data2])
        data = data.reshape(data.shape[0], -1)

        # forward
        pat, mid, lat = self(data)

        # loss
        loss_topo = self.Loss(
            input_data=mid.reshape(mid.shape[0], -1),
            latent_data=lat.reshape(lat.shape[0], -1),
            v_latent=self.nu,
            eye=self.loss_eye,
            metric="euclidean",
            # metric='cossim',
        )

        loss_l2 = 0
        if self.current_epoch >= self.log_interval and batch_idx == 0:
            if self.alpha is None:
                # logging.debug("--->")
                self.alpha = loss_topo.detach().item() / (
                    self.Cal_Sparse_loss(
                        self.PM_root.weight.reshape(-1),
                    ).detach()
                    * self.l2alpha
                )

            N_Feature = np.sum(gpu2np(self.mask) > 0)
            if N_Feature > self.num_fea_aim:
                loss_l2 = self.Cal_Sparse_loss(self.PM_root.weight.reshape(-1))
                self.alpha = self.alpha * self.detaalpha
                loss_topo += (loss_l2) * self.alpha
        return loss_topo

    def validation_step(self, batch, batch_idx):
        # augmentation
        if (self.current_epoch + 1) % self.log_interval == 0:
            index = batch.to(self.device)
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

    def validation_epoch_end(self, outputs):
        if not self.stop:
            logging.debug(f"es_monitor: {self.current_epoch}")
        else:
            logging.debug(f"es_monitor: 0")

        if (self.current_epoch + 1) % self.log_interval == 0:
            logging.debug(f"self.current_epoch: {self.current_epoch}")
            data = np.concatenate([data_item[0] for data_item in outputs])
            mid_old = np.concatenate([data_item[1] for data_item in outputs])
            ins_emb = np.concatenate([data_item[2] for data_item in outputs])
            index = np.concatenate([data_item[3] for data_item in outputs])

            self.data = data
            self.mid_old = mid_old
            self.ins_emb = ins_emb
            self.index = index

            N_link = np.sum(gpu2np(self.mask))
            feature_use_bool = gpu2np(self.mask) > 0
            N_Feature = np.sum(feature_use_bool)

            # import pdb; pdb.set_trace()
              
            if N_Feature <= self.num_fea_aim:
                self.stop = True
            else:
                self.stop = False

        else:
            logging.debug(f"SVC: 0")

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
        struc_model_b = NetworkStructure_2 + [2]
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

    def augmentation_warper(self, index, data1):
        if len(data1.shape) == 2:
            return self.augmentation(index, data1)
        else:
            return self.augmentation_img(index, data1)

    def augmentation_img(self, index, data):
        # aug = []
        # for i in range(data.shape[0]):
        #     aug.append(
        #         self.transforms(data.permute(0,3,1,2)).reshape(1,-1)
        #         )
        return self.transforms(data.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def augmentation(self, index, data1):
        data2_list = []
        if self.Uniform_t > 0:
            data_new = aug_near_mix(
                index,
                self.data_train,
                k=self.K,
                random_t=self.Uniform_t,
                device=self.device,
            )
            data2_list.append(data_new)
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
            data2_list.append(data_new)

        if len(data2_list) == 1:
            data2 = data2_list[0]
        elif len(data2_list) == 2:
            data2 = (data2_list[0] + data2_list[1]) / 2
        elif len(data2_list) == 3:
            data2 = (data2_list[0] + data2_list[1] + data2_list[2]) / 3

        return data2
