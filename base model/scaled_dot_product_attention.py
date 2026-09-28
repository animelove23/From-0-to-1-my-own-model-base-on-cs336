from cs336_basics.softmax import softmax
import torch
import math


def scaled_dot_product_attention(q, k, v, mask=None):
    q_k= q@k.transpose(-1,-2)/math.sqrt(k.shape[-1])
    if mask is not None:
        score = q_k.masked_fill(mask, -float('inf'))
    else:
        score = q_k
    attention = softmax(score, -1)@v # 注意力分数被mask， 因此每个token位置只能获取他之前已经生成的token的信息来重新组合，
    # seq seq @ seq dim，attention score 决定从每个 token 的 Value 向量中抽取多少信息，然后加权求和，形成当前位置的新表示。
    return attention

