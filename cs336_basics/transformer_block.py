import torch
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.FFN import SwiGLU
from cs336_basics.embadding import Embedding
from cs336_basics.Pre_norm import RMSNorm
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self,dff,d_model,n_heads,open_pos= True,theta = None,max_seq_len = None,token_positions = None):
        super(TransformerBlock, self).__init__()
        self.dff = dff
        self.n_heads = n_heads
        self.open_pos = open_pos
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions
        self.rmsnorm = RMSNorm(d_model)
        self.causal_att = MultiHeadSelfAttention(d_model,n_heads,open_pos,theta,max_seq_len,token_positions)
        self.ffn = SwiGLU(d_model,dff)
    def forward(self,x):
        residual = x
        norm = self.rmsnorm(x)
        attention = self.causal_att(norm)
        result_1 = residual+attention
        residual_1 = result_1
        norm = self.rmsnorm(residual_1)
        ffn_1 = self.ffn(norm)
        result_2 = residual+ffn_1
        return result_2