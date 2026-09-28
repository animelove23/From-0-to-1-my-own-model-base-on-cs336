import torch
from torch import nn

def softmax(x:torch.Tensor,i):
    max_ele = x.max(dim=i,keepdim=True).values
    num= x-max_ele
    exp_num = torch.exp(num)
    soft = exp_num/exp_num.sum(dim= i, keepdim=True)
    return soft

