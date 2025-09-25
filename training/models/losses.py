import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, reduction='mean'):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        weights = self.get_cb_weights(samples_per_class, beta)
        self.register_buffer("weights", weights)

    def get_cb_weights(self, samples_per_cls, beta):
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)
        return torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weights.to(logits.device), reduction=self.reduction)
        return ce
