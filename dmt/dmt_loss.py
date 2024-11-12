# a pytorch based lisv2 code

import pdb
from multiprocessing import Pool

import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
from scipy import optimize
from torch import nn
from torch.autograd import Variable
from torch.functional import split
from torch.nn.modules import loss
from typing import Any
import scipy

class MyLoss(nn.Module):
    def __init__(
        self,
        v_input,
        v_latent,
        SimilarityFunc,
        # device: Any,
        augNearRate=100000,
        sigmaP=1.0,
        sigmaQ=1.0,
    ):
        super(MyLoss, self).__init__()

        # self.device = device
        self.v_input = v_input
        self.v_latent = v_latent
        self.gamma_input = self._CalGamma(v_input)
        self.ITEM_loss = self._TwowaydivergenceLoss
        self._Similarity = SimilarityFunc
        self.augNearRate = augNearRate
        self.sigmaP = sigmaP
        self.sigmaQ = sigmaQ
    
    def forward(self, input_data, input_data_aug, latent_data, latent_data_aug, rho=0, sigma=1):

        disInput = self._DistanceSquared(input_data, input_data)
        # metaDistance_ = metaDistance.clone().detach()
        # metaDistance_[torch.eye(metaDistance_.shape[0])==1.0] = metaDistance_.max()+1
        # nndistance, _ = torch.min(metaDistance_, dim=0)
        # nndistance = nndistance / self.augNearRate
        # downDistance = metaDistance + nndistance.reshape(-1, 1)
        # rightDistance = metaDistance + nndistance.reshape(1, -1)
        # rightdownDistance = metaDistance + nndistance.reshape(1, -1) + nndistance.reshape(-1, 1)

        # disInput = torch.cat(
        #     [
        #         torch.cat([metaDistance, downDistance]),
        #         torch.cat([rightDistance, rightdownDistance]),
        #     ],
        #     dim=1
        # )
        # latent = torch.cat([latent_data, latent_data_aug])
        distlatent = self._DistanceSquared(latent_data, latent_data)
        # print('self.sigmaP', self.sigmaP)
        loss = self._TwowaydivergenceLoss(
                    P=self._Similarity(
                        dist=disInput,
                        rho=0,
                        sigma_array=self.sigmaP,
                        gamma=self._CalGamma(self.v_input),
                        v=self.v_input
                    ),
                    Q=self._Similarity(
                        dist=distlatent,
                        rho=0,
                        sigma_array=self.sigmaQ,
                        gamma=self._CalGamma(self.v_latent),
                        v=self.v_latent,
                    )
                )

        # print(loss_ce.shape)
        return loss.mean()

    def _TwowaydivergenceLoss(self, P, Q):

        EPS = 1e-12
        # P_ = P[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        # Q_ = Q[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        losssum1 = (P * torch.log(Q + EPS))
        losssum2 = ((1-P) * torch.log(1-Q + EPS))
        losssum = -1*(losssum1 + losssum2)

        # if torch.isnan(losssum):
        #     input('stop and find nan')
        return losssum

    def _L2Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=2)/P.shape[0]
        return losssum
    
    def _L3Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=3)/P.shape[0]
        return losssum

    # def _Similarity(self, dist, rho, sigma_array, gamma, v=100):

    #     if torch.is_tensor(rho):
    #         dist_rho = (dist - rho) / sigma_array
    #     else:
    #         dist_rho = dist
        
    #     dist_rho[dist_rho < 0] = 0
    #     Pij = torch.pow(
    #         gamma * torch.pow(
    #             (1 + dist_rho / v),
    #             -1 * (v + 1) / 2
    #             ) * torch.sqrt(torch.tensor(2 * 3.14)),
    #             2
    #         )

    #     P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

    #     return P
    
    def _DistanceSquared(
        self,
        x,
        y
    ):
        # x = x.float().reshape(x.shape[0], -1)
        # y = y.float().reshape(y.shape[0], -1)

        # m, n = x.size(0), y.size(0)
        # xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        # dist = xx + yy
        # dist.addmm_(mat1=x, mat2=y.t(), beta=1, alpha=-2, )
        # dist = dist.clamp(min=1e-12)

        return torch.pow(torch.cdist(x, y, p=2),2)

    def _CalGamma(self, v):
        
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out