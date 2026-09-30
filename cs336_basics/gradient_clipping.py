import torch

def gradient_clipping(weight, max_l2):
    e = 10**(-6)
    l2_norm = 0
    for i in weight:
        if i.grad is not None:
            l2_norm += i.grad.pow(2).sum()
    l2_norm = l2_norm.sqrt()
    if l2_norm > max_l2:
        scale = max_l2 / (l2_norm + e)

        for i in weight:
            if i.grad is not None:
                i.grad.mul_(scale)

    