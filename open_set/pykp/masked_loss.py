import torch
import math

EPS = 1e-8

def masked_cross_entropy(class_dist, target, trg_mask, trg_lens=None,):
    """
    :param class_dist: [batch_size, trg_seq_len, num_classes]
    :param target: [batch_size, trg_seq_len]
    :param trg_mask: [batch_size, trg_seq_len]
    :param trg_lens: a list with len of batch_size
    :return:
    """
    num_classes = class_dist.size(2)
    class_dist_flat = class_dist.view(-1, num_classes)  # [batch_size*trg_seq_len, num_classes]
    log_dist_flat = torch.log(class_dist_flat + EPS)
    target_flat = target.view(-1, 1)  # [batch*trg_seq_len, 1]
    losses_flat = -torch.gather(log_dist_flat, dim=1, index=target_flat)  # [batch * trg_seq_len, 1]
    losses = losses_flat.view(*target.size())  # [batch, trg_seq_len]
    if trg_mask is not None:
        losses = losses * trg_mask
    loss = losses.sum(dim=1)  # [batch_size]
    loss = loss.sum()

    # Debug
    if math.isnan(loss.item()):
        print("class distribution")
        print(class_dist)
        print("log dist flat")
        print(log_dist_flat)

    return loss
