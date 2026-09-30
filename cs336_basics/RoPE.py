from torch import nn
import torch

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RoPE, self).__init__()
        k= d_k//2
        self.num_dim=k
        k= torch.arange(1,k+1)
        frequency = 1/ (theta**((2*k-2)/d_k))
        position = torch.arange(max_seq_len)
        frequency= frequency.unsqueeze_(0)
        position= position.unsqueeze_(1)
        angle = frequency * position
        self.register_buffer("cosine_table", torch.cos(angle), persistent=False)
        self.register_buffer("sine_table", torch.sin(angle), persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(
            x.shape[2],
            device=x.device
        )
        cosine_angle = self.cosine_table[token_positions]
        sine_angle = self.sine_table[token_positions]
        cosine_angle = cosine_angle.to(x.dtype)
        sine_angle = sine_angle.to(x.dtype)
        original_shape = x.shape
        x= x.reshape(*original_shape[:-1],self.num_dim,2)
        first_ele = x[...,0]
        second_ele = x[...,1]
        first_rotation= first_ele*cosine_angle- second_ele*sine_angle
        second_rotation= first_ele*sine_angle+ second_ele*cosine_angle
        result = torch.stack([first_rotation, second_rotation], dim=-1)
        return result.reshape(original_shape)