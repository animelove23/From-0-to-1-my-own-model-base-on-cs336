import torch
from einops import einsum
from torch import nn

class SwiGLU(nn.Module):
    def __init__(self, in_features,dff = None, device = None, dtype = None):
        super(SwiGLU, self).__init__()
        self.in_features = in_features
        self.w1_weight = nn.Parameter(torch.empty(dff or round(8*in_features/3/64)*64,in_features ,device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty(dff or round(8*in_features/3/64)*64,in_features ,device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(in_features,dff or round(8*in_features/3/64)*64,device=device, dtype=dtype))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        gate = einsum(x,self.w1_weight,"... dff, out_features dff -> ... out_features")
        swi_output = gate*torch.sigmoid(gate)
        temp_output = einsum(x,self.w3_weight,"... dff, out_features dff -> ... out_features")
        glu_output = swi_output * temp_output
        return einsum(glu_output,self.w2_weight,"... out_features, dff out_features -> ... dff")