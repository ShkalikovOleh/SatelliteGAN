import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma, weights=None):
        self.weights = weights
        self.gamma = gamma

    def __call__(self, pred, target):
        p = F.softmax(pred, dim=1).gather(1, target.unsqueeze(dim=1))
        mult = (1 - p)**self.gamma
        ce = F.cross_entropy(pred, target,
                             weight=self.weights,
                             reduction='none')
        return (ce * mult).mean()
