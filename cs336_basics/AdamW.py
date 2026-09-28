from collections.abc import Callable
from typing import Optional

import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params,lambd, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(AdamW, self).__init__(params, defaults)
        self.lambd = lambd
        self.eps = eps
        self.betas = betas
    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:

            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state['m'] = self.betas[0] * state['m'] + (1 - self.betas[0]) * grad
                state['v'] = self.betas[1] * state['v'] + (1 - self.betas[1]) * grad ** 2
                t = state.get("step", 0) + 1
                p.data-= lr*self.lambd*p.data # 权重衰退
                at =lr*math.sqrt(1-self.betas[1]**t)/(1-self.betas[0]**t)
                p.data -= at*(state['m']/(torch.sqrt(state['v'])+self.eps))
                state['step'] = t

        return loss


