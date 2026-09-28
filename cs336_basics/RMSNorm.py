import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.g= nn.Parameter(torch.empty(d_model,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.g,mean=0.0,std=eps,a=-3*eps,b=eps)# 这里有疑问
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_type = x.dtype
        x=x.to(dtype=torch.float32)
        result = (x/torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))*self.g
        result = result.to(dtype=d_type)
        return result