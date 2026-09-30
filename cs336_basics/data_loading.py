import torch
import numpy as np
def data_loading(x, batch_size,context_length,device=None):
    label = []
    target = []
    start = np.random.randint(0,len(x)-context_length, size = batch_size)
    for i in range(batch_size):
        label.append(
            x[start[i]:start[i] + context_length]
        )

        target.append(
            x[start[i] + 1:start[i] + context_length + 1]
        )
    label = torch.LongTensor(label).to(device)
    target = torch.LongTensor(target).to(device)
    return (label,target)