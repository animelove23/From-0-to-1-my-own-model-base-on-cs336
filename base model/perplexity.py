import torch
from cs336_basics.cross_entropy import cross_entropy

def perplexity(logits, targets):
    perplexity = torch.exp(logits)
    return perplexity