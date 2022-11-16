import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys


class FF(torch.nn.Module):
    def __init__(self, n_input=1_300_000, n_output=8):
        self.lin_one = torch.nn.Linear(1_300_000, 1024)
        self.lin_two = torch.nn.Linear(1024, 256)
        self.lin_three = torch.nn.Linear(256, 8)
        self.non_lin = torch.nn.ReLU()

    def forward(self, x):
        x = self.lin_one(x)
        x = self.lin_two(x)
        x = self.lin_three(x)
        return x