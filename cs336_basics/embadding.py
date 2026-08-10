import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding_matrix,mean=0.0,std=1,a=-3,b=3)
    def forward(self,token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
