from cs336_basics.softmax import softmax
import torch
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    q_k= q@k.transpose(-1,-2)/math.sqrt(k.shape[-1])
    if mask is not None:
        score = q_k.masked_fill(mask, -float('inf'))
    else:
        score = q_k
    attention = softmax(score, -1)@v
    return attention

