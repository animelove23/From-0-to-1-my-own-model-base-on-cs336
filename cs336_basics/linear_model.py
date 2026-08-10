from einops import rearrange,einsum
import torch
from torch import nn
import math


class LinearModel(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype= None):
        super(LinearModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features,device=device,dtype=dtype))
        std=math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight,mean=0, std = std,a=-3*std, b=3*std)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return einsum(x,self.weight, "... in_features,out_features in_features -> ... out_features")


