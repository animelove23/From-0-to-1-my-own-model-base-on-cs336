import torch


def cross_entropy(logits, targets):
    max_token = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_token
    log_sum_exp = torch.log(
        torch.exp(shifted_logits).sum(dim=-1)
    )
    batch_indices = torch.arange(
        logits.shape[0],
        device=logits.device
    )
    target_logits = shifted_logits[
        batch_indices,
        targets
    ]
    loss = log_sum_exp - target_logits

    return loss.mean()