import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def calculate_entropy_logits(logits: Tensor, dim: int) -> Tensor:
    r"""
    Calculate the entropy from logits.
    """
    return -(F.softmax(logits, dim=dim) * F.log_softmax(logits, dim=dim)).sum(dim=dim)

def calculate_entropy_probs(probs: Tensor) -> Tensor:#, dim: int) -> Tensor:
    r"""
    Calculate the entropy from probabilities.
    """
    # return F.binary_cross_entropy(probs, probs, reduction='sum')#.sum(dim=dim)
    return (-probs * torch.log(probs.clamp(1e-8, 1))).sum()

def calculate_batch_accuracy(output: Tensor, target: Tensor) -> float:
    r"""
    Calculate the accuracy of the batch.
    """
    return (output.argmax(dim=1) == target).float().mean().item()


class MarginLoss(nn.Module):
    def __init__(self, m=0.2, weight=None, temperature=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.temperature = temperature
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        # print("Index shape: ", index.shape)
        # print("Target shape: ", target.shape)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.temperature

        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)
