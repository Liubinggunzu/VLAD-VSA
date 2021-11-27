from torch.autograd import Function
import torch.nn as nn
import torch

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse
class MSE_nosum(nn.Module):
    def __init__(self):
        super(MSE_nosum, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        # n = torch.numel(diffs.data)
        # mse = torch.sum(diffs.pow(2)) / n

        return diffs.pow(2)

class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse