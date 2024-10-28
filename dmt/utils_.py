import torch

def gpu2np(a: torch.Tensor):
    return a.cpu().detach().numpy()