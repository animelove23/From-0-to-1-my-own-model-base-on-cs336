import torch
import random
from cs336_basics.softmax import softmax
def decoder(x, logits, stop_token_id, max_len = 256, temperature = 1.0, p = 0.1): # x 是输入的还没被decode的句子（token id）
    last_pos = logits[0, -1, :]
    if x.shape[1] > max_len:
        return x, False # 达到最大长度，不能输出下一个token
    prob = softmax(last_pos/temperature, -1)
    prob, index = torch.sort(prob, dim = 0, descending = True)
    total = 0
    k=0
    for pr in prob:
        total += pr.item()
        k += 1
        if total > p:
            break
    prob = prob[:k]
    index = index[:k]
    next_token = torch.multinomial(prob, num_samples=1)
    next_token_id = index[next_token]
    x = torch.cat([x, next_token_id.view(1, 1)],dim=1)
    if(next_token_id == stop_token_id):
        return x, False # 达到停止词，不能输出下一个token
    return x,True # 我们返回选出的token和原来句子的拼接，把decode 任务留到对话脚本中进行


