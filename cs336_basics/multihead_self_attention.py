import torch
from torch import nn
from cs336_basics.RoPE import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, open_pos,theta =None,max_seq_len = None,token_positions = None):
        super(MultiHeadSelfAttention, self).__init__()
        self.d = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions
        self.open_pos = open_pos
        self.w_q = nn.Parameter(torch.empty(self.d*n_heads, d_model))
        self.w_k = nn.Parameter(torch.empty(self.d*n_heads, d_model))
        self.w_v = nn.Parameter(torch.empty(self.d*n_heads, d_model))
        self.w_o = nn.Parameter(torch.empty(d_model, self.d*n_heads))
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        nn.init.xavier_uniform_(self.w_o)
    def forward(self,x):
        q = x@self.w_q.T
        k = x@self.w_k.T
        v = x@self.w_v.T
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.d)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.d)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.d)
        mask = torch.triu(torch.ones(q.shape[1], q.shape[1], dtype = torch.bool), diagonal=1)
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)
        if self.open_pos:
            theta = self.theta or 10000
            max_seq_len = self.max_seq_len or q.shape[2]
            rope = RoPE(theta,self.d, max_seq_len)
            q=rope(q, self.token_positions)
            k=rope(k, self.token_positions)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.transpose(1,2)
        scaled_attention = scaled_attention.reshape(scaled_attention.shape[0], scaled_attention.shape[1],self.n_heads*(self.d))
        scaled_attention= scaled_attention@self.w_o.T
        return scaled_attention





