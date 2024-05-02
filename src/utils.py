import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def calculate_entropy(logits: Tensor) -> Tensor:
    r"""
    Calculate the entropy of the logits.
    """
    return (F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)


def calculate_batch_accuracy(output: Tensor, target: Tensor) -> float:
    r"""
    Calculate the accuracy of the batch.
    """
    return (output.argmax(dim=1) == target).float().mean().item()


class MarginLoss(nn.Module):
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s

        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)
