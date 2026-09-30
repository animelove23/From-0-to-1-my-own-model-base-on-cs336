from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.FFN import SwiGLU
from cs336_basics.embadding import Embedding
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.linear_model import LinearModel
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.RoPE import RoPE
from torch import nn

class Transformer(nn.Module):
    def __init__(self,dff,d_model,n_heads,vocab_size,context_length,num_layers,open_pos= True,theta = None):
        super(Transformer, self).__init__()
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dff,
                d_model,
                n_heads,
                open_pos,
                theta,
                context_length,
            )
            for _ in range(num_layers)
        ])
        self.embedding = Embedding(vocab_size,d_model)
        self.linear = LinearModel(d_model, vocab_size)
        self.RMSNorm = RMSNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.num_layers = num_layers
    def forward(self,x):
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.RMSNorm(x)
        # [batch, seq, d_model]
        logits = self.linear(x)
        # [batch, seq, vocab_size]
        return logits