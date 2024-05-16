import functools
from functools import reduce
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import sys
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
import logging

# import Loss.dmt_loss_aug as dmt_loss_aug1
from dmtev.aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn
from dmtev.dataloader import data_base
from dmtev.model.model import NN_FCBNRL_MM
import dmtev.Loss.dmt_loss_aug2 as dmt_loss_aug

torch.set_num_threads(2)


def gpu2np(a: torch.Tensor):
    return a.cpu().detach().numpy()

class LitPatNN(LightningModule):
    def __init__(
        self,
        dataname,
        **kwargs,
    ):

        super().__init__()

        # Set our init args as class attributes
        self.my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataname = dataname
        # self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.t = 0.1
        self.alpha = None
        self.stop = False
        self.detaalpha = self.hparams.detaalpha
        self.bestval = 0
        self.aim_cluster = None
        self.importance = None
        self.setup()
        self.mse = torch.nn.CrossEntropyLoss()

        self.hparams.num_pat = min(self.data_train.data.shape[1], self.hparams.num_pat)

        self.model_pat, self.model_b = self.InitNetworkMLP(
            # self.model_pat, self.model_b = self.InitNetworkMLP_OLD(
            self.hparams.NetworkStructure_1,
            self.hparams.NetworkStructure_2,
        )
        self.hparams.num_fea_aim = min(
            self.hparams.num_fea_aim, reduce(lambda x, y: x*y, self.data_train.data.shape[1:]) 
        )

        self.Loss = dmt_loss_aug.MyLoss(
            v_input=100,
            metric=self.hparams.metric,
            augNearRate=self.hparams.augNearRate,
        )

        if len(self.data_train.data.shape) > 2:
            self.transforms = transforms.AutoAugment(
                transforms.AutoAugmentPolicy.CIFAR10
            )


        self.fea_num = 1
        for i in range(len(self.data_train.data.shape) - 1):
            self.fea_num = self.fea_num * self.data_train.data.shape[i + 1]

        logging.debug("fea_num", self.fea_num)
        self.PM_root = nn.Linear(self.fea_num, 1)
        self.PM_root.weight.data = torch.ones_like(self.PM_root.weight.data) / 5


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
            v_latent=self.hparams.nu,
            metric="euclidean",
            # metric='cossim',
        )


        if args.wandb:
            self.wandb_logs = {
                # "loss_mse": loss_mse,
                "loss_topo": loss_topo,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "epoch": self.current_epoch,
                # "T": self.t_list[self.current_epoch],
            }

        loss_l2 = 0
        if self.current_epoch >= self.hparams.log_interval and batch_idx == 0:
            if self.alpha is None:
                # logging.debug("--->")
                self.alpha = loss_topo.detach().item() / (
                    self.Cal_Sparse_loss(
                        self.PM_root.weight.reshape(-1),
                    ).detach()
                    * self.hparams.l2alpha
                )

            N_Feature = np.sum(gpu2np(self.mask) > 0)
            if N_Feature > self.hparams.num_fea_aim:
                loss_l2 = self.Cal_Sparse_loss(self.PM_root.weight.reshape(-1))
                self.alpha = self.alpha * self.detaalpha
                loss_topo += (loss_l2) * self.alpha
        return loss_topo

    def validation_step(self, batch, batch_idx):
        # augmentation
        if (self.current_epoch + 1) % self.hparams.log_interval == 0:
            index = batch.to(self.device)
            data = self.data_train.data[index]
            data = data.reshape(data.shape[0], -1)
            pat, mid, lat = self(data)

            return (
                gpu2np(data),
                gpu2np(pat),
                gpu2np(lat),
                np.array(self.data_train.label.cpu())[gpu2np(index)],
                gpu2np(index),
            )

    def Cal_Sparse_loss(self, PatM):
        loss_l2 = torch.abs(PatM).mean()
        return loss_l2

    def validation_epoch_end(self, outputs):
        if not self.stop:
            self.log("es_monitor", self.current_epoch)
        else:
            self.log("es_monitor", 0)

        if (self.current_epoch + 1) % self.hparams.log_interval == 0:
            logging.debug("self.current_epoch", self.current_epoch)
            data = np.concatenate([data_item[0] for data_item in outputs])
            mid_old = np.concatenate([data_item[1] for data_item in outputs])
            ins_emb = np.concatenate([data_item[2] for data_item in outputs])
            label = np.concatenate([data_item[3] for data_item in outputs])
            index = np.concatenate([data_item[4] for data_item in outputs])

            self.data = data
            self.mid_old = mid_old
            self.ins_emb = ins_emb
            self.label = label
            self.index = index

            N_link = np.sum(gpu2np(self.mask))
            feature_use_bool = gpu2np(self.mask) > 0
            N_Feature = np.sum(feature_use_bool)

            # import pdb; pdb.set_trace()
            
            have_the_label_info = (label.max() != label.min())
            
            if self.alpha is not None and N_Feature <= self.hparams.num_fea_aim and have_the_label_info:
                data_test = self.data_test.data
                label_test = self.data_test.label
                _, _, lat_test = self(data_test)
                

            if N_Feature <= self.hparams.num_fea_aim:
                self.stop = True
            else:
                self.stop = False


        else:
            self.log("SVC", 0)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-9
        )
        self.scheduler = StepLR(
            optimizer, step_size=self.hparams.epochs // 10, gamma=0.8
        )
        return [optimizer], [self.scheduler]

    def setup(self, stage=None):
        # import pdb; pdb.set_trace()
        dataset_f = getattr(data_base, self.dataname + "Dataset")
        self.data_train = dataset_f(
            data_name=self.hparams.data_name,
            train=True,
            datapath=self.hparams.data_path,
        )
        if len(self.data_train.data.shape) == 2:
            self.data_train.cal_near_index(
                device=self.my_device,
                k=self.hparams.K,
                uselabel=bool(self.hparams.uselabel),
            )
        self.data_train.to_device(self.my_device)

        self.data_test = dataset_f(
            data_name=self.hparams.data_name,
            train=True,
            datapath=self.hparams.data_path,
        )
        self.data_test.to_device(self.my_device)

        self.dims = self.data_train.get_dim()

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            drop_last=True,
            batch_size=min(self.hparams.batch_size, self.data_train.data.shape[0]),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=min(self.hparams.batch_size, self.data_train.data.shape[0]),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size)

    def InitNetworkMLP(self, NetworkStructure_1, NetworkStructure_2):

        num_fea_per_pat = self.hparams.num_fea_per_pat
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
        if self.hparams.Uniform_t > 0:
            data_new = aug_near_mix(
                index,
                self.data_train,
                k=self.hparams.K,
                random_t=self.hparams.Uniform_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if self.hparams.Bernoulli_t > 0:
            data_new = aug_near_feautee_change(
                index,
                self.data_train,
                k=self.hparams.K,
                t=self.hparams.Bernoulli_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if self.hparams.Normal_t > 0:
            data_new = aug_randn(
                index,
                self.data_train,
                k=self.hparams.K,
                t=self.hparams.Normal_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if (
            max(
                [
                    self.hparams.Uniform_t,
                    self.hparams.Normal_t,
                    self.hparams.Bernoulli_t,
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

    def pdist2(self, x: torch.Tensor, y: torch.Tensor):
        # calculate the pairwise distance
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12)
        return dist


    def up_mainfig_emb(
        self, data, ins_emb,
        label, index, mask,
        n_clusters=10, num_cf_example=2,
    ):
        color = np.array(label)

        fig = px.scatter(
            x=ins_emb[:, 0], y=ins_emb[:, 1], color=[str(c) for c in color]
        )
        
        fig.to_image(format="png", engine="kaleido")
        np.save("save_emb/" + self.hparams.data_name + "={}".format(self.current_epoch), ins_emb)
        
        return fig




def main(args):

    pl.utilities.seed.seed_everything(1)
    info = [str(s) for s in sys.argv[1:]]

    model = LitPatNN(dataname=args.data_name,**args.__dict__,)

    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=args.epochs,
    )
    logging.debug("start fit")
    trainer.fit(model)
    logging.debug("end fit")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="author: Zelin Zang; zangzelin@gmail.com")

    # data set param
    parser.add_argument(
        "--data_name",
        type=str,
        default="CSV",
        choices=[
            "CSV",
            "InsEmb_PBMC",
            "OTU",
            "Activity",
            "Gast10k1457",
            "PBMCD2638",
            "PBMC",
            "InsEmb_TPD_579_ALL_PRO",
            "InsEmb_TPD_579_ALL_PRO5C",
            "YONGJIE_UC",
            "Digits",
            "Mnist",
            "Mnist3000",
            "Mnist10000",
            "EMnist",
            "KMnist",
            "FMnist",
            "Coil20",
            "Coil100",
            "Smile",
            "ToyDiff",
            "SwissRoll",
            "EMnistBC",
            "EMnistBYCLASS",
            "Cifar10",
            "Colon",
            "Gast10k",
            "HCL60K50D",
            "HCL60K3037D",
            "HCL280K50D",
            "HCL280K3037D",
            "SAMUSIK",
            "MiceProtein",
            "BASEHOCK",
            "GLIOMA",
            "leukemia",
            "pixraw10P",
            "Prostatege",
            "WARPARIOP",
            "arcene",
            "MCA",
            "MCAD9119",
            "PeiHuman",
            "PeiHumanTop2",
            "E1",
        ],
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--data_path", type=str, default="data/niu")
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument(
        "--computer", type=str,
        default=os.popen("git config user.name").read()[:-1]
    )
    parser.add_argument(
        "--n_point",
        type=int,
        default=60000000,
    )
    # model param
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
    )
    parser.add_argument("--detaalpha", type=float, default=1.001)
    parser.add_argument("--l2alpha", type=float, default=10)
    parser.add_argument("--nu", type=float, default=1e-2)
    parser.add_argument("--num_link_aim", type=float, default=0.2)
    # parser.add_argument("--num_fea_aim", type=int, default=128)
    parser.add_argument("--num_fea_aim", type=int, default=50)
    parser.add_argument("--K_plot", type=int, default=40)

    parser.add_argument("--num_fea_per_pat", type=int, default=80)  # 0.5
    # parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--Uniform_t", type=float, default=1)  # 0.3
    parser.add_argument("--Bernoulli_t", type=float, default=-1)
    parser.add_argument("--Normal_t", type=float, default=-1)
    parser.add_argument("--uselabel", type=int, default=0)
    parser.add_argument("--showmainfig", type=int, default=1)

    # train param
    parser.add_argument(
        "--NetworkStructure_1", type=list, default=[-1, 200] + [200] * 5
    )
    parser.add_argument("--NetworkStructure_2", type=list, default=[-1, 500, 80])
    parser.add_argument("--num_pat", type=int, default=8)
    parser.add_argument("--num_latent_dim", type=int, default=2)
    parser.add_argument("--augNearRate", type=float, default=1000)
    parser.add_argument("--explevel", type=int, default=3)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
    )
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    
    # use wandb
    parser.add_argument("--wandb", action="store_true")

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    main(args)
