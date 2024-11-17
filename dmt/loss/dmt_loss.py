# a pytorch based lisv2 code

# from multiprocessing import Pool

import numpy as np
import torch
import torch.autograd
from torch import nn
from scipy.special import gamma
from ..manifolds import hyperboloid as hyperboloid
from ..manifolds import poincare as poincare


def UMAPSimilarity(dist, rho, sigma_array, gamma, v=100, h=1, pow=2):

    if torch.is_tensor(rho):
        dist_rho = (dist - rho) / sigma_array
        dist_rho[dist_rho < 0] = 0
    else:
        dist_rho = dist

    dist_rho[dist_rho < 0] = 0
    
    if v > 500:
        Pij = torch.pow(
            input=torch.exp(-1*dist_rho),
            exponent=pow
        )
    else:
        Pij = torch.pow(
            input=gamma * torch.pow(
                (1 + dist_rho / v),
                exponent= -1 * (v + 1) / 2
                ) * torch.sqrt(torch.tensor(2 * 3.14)),
            exponent=pow
            )

    Pij = Pij + Pij.t() - torch.mul(Pij, Pij.t())
    return Pij


class MyLoss(nn.Module):
    def __init__(
        self,
        v_input,
        SimilarityFunc=UMAPSimilarity,
        metric="braycurtis",
        eta=1,
        near_bound=0,
        far_bound=1,
        augNearRate=10000,
        batchRate=100,
        device=torch.device("cpu"),
    ):
        super(MyLoss, self).__init__()

        self.v_input = v_input
        self.gamma_input = self._CalGamma(v_input)
        self.ITEM_loss = self._TwowaydivergenceLoss
        self._Similarity = SimilarityFunc
        self.metric = metric
        self.eta = eta
        self.near_bound = near_bound
        self.far_bound = far_bound
        self.augNearRate = augNearRate
        self.batchRate = batchRate
        self.device = device

    def forward(
        self, 
        input_data, 
        latent_data, 
        batchhot, 
        v_latent,
        metric='euclidean',
    ):

        dis_batchhot = self._DistanceSquared(batchhot, metric='euclidean', c=1)

        data = input_data[:input_data.shape[0]//2]
        dis_P = self._DistanceSquared(data, metric='euclidean', c=1)
        dis_P_ = dis_P.clone().detach()
        dis_P_[torch.eye(dis_P_.shape[0])==1.0] = dis_P_.max()+1
        nndistance, _ = torch.min(dis_P_, dim=0)
        nndistance = nndistance / self.augNearRate

        downDistance = dis_P + nndistance.reshape(-1, 1)
        rightDistance = dis_P + nndistance.reshape(1, -1)
        rightdownDistance = dis_P + nndistance.reshape(1, -1) + nndistance.reshape(-1, 1)
        disInput = torch.cat(
            [
                torch.cat([dis_P, downDistance]),
                torch.cat([rightDistance, rightdownDistance]),
            ],
            dim=1
        )
        P = self._Similarity(
                dist=disInput,
                rho=0,
                sigma_array=1,
                gamma=self.gamma_input,
                v=self.v_input,
        )
        
        dis_Q = self._DistanceSquared(latent_data, metric=metric, c=1)
        dis_Q = dis_Q + self.batchRate * dis_batchhot
        Q = self._Similarity(
                dist=dis_Q,
                rho=0,
                sigma_array=1,
                gamma=self._CalGamma(v_latent),
                v=v_latent,
        )
        
        loss_ce = self.ITEM_loss(P_=P, Q_=Q)
        return loss_ce
    
    def ForwardInfo(
        self,
        input_data,
        latent_data,
        rho,
        sigma,
        v_latent,
    ):

        dis_P = self._DistanceSquared(input_data)
        P = self._Similarity(
            dist=dis_P,
            rho=rho,
            sigma_array=sigma,
            gamma=self.gamma_input,
            v=self.v_input,
        )

        dis_Q = self._DistanceSquared(latent_data)
        Q = self._Similarity(
            dist=dis_Q, rho=0, sigma_array=1, gamma=self._CalGamma(v_latent), v=v_latent
        )

        loss_ce = self.ITEM_loss(
            P_=P,
            Q_=Q,
        )
        return (
            loss_ce.detach().cpu().numpy(),
            dis_P.detach().cpu().numpy(),
            dis_Q.detach().cpu().numpy(),
            P.detach().cpu().numpy(),
            Q.detach().cpu().numpy(),
        )

    def _TwowaydivergenceLoss(self, P_, Q_, select=None):

        EPS = 1e-12
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)

        return losssum.mean()

    def _L2Loss(self, P, Q):

        losssum = torch.norm(P - Q, p=2) / P.shape[0]
        return losssum

    def _L3Loss(self, P, Q):

        losssum = torch.norm(P - Q, p=3) / P.shape[0]
        return losssum

    def _DistanceSquared(self, x, y=None, metric="euclidean", c=1):
        if metric == "euclidean":
            if y is not None:
                m, n = x.size(0), y.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-22)
            else:
                m, n = x.size(0), x.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = xx.t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-22)
                dist[torch.eye(dist.shape[0]) == 1] = 1e-22
        
        if metric == "cossim":
            input_a, input_b = x, x
            normalized_input_a = torch.nn.functional.normalize(input_a)  
            normalized_input_b = torch.nn.functional.normalize(input_b)
            dist = torch.mm(normalized_input_a, normalized_input_b.T)
            dist *= -1 # 1-dist without copy
            dist += 1
            dist[torch.eye(dist.shape[0]) == 1] = 1e-22

        if metric == 'poin_dist_mobiusm_v2':
            PoincareBall = poincare.PoincareBall()
            dist = PoincareBall.sqdist_xu_mobius_v2(x, x, c=1)
            dist = dist.clamp(min=1e-22)
            dist[torch.eye(dist.shape[0]) == 1] = 1e-22

        if metric == 'lor_dist_v2':
            Hyperboloid = hyperboloid.Hyperboloid()
            dist = Hyperboloid.sqdist_xu_v2(x, x, c=1)
            dist = dist.clamp(min=1e-22)
            dist[torch.eye(dist.shape[0]) == 1] = 1e-22

        return dist

    def _CalGamma(self, v):

        a = gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * gamma(v / 2)
        out = float(a / b)  # type: np.float64

        return out
