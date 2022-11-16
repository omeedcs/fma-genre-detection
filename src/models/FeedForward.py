import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

# feed forward neural network for audio data
class FF(torch.nn.Module):
    def __init__(self, n_input, genre_output = 8):
        super(FF, self).__init__()
        # not sure about the 1_300_000, made it n_input to depend off of train_neural.
        self.lin_one = nn.Linear(n_input, 1024)
        
        self.lin_two = nn.Linear(1024, 256)
        # genre output should be 8, using small dataset.
        self.lin_three = nn.Linear(256, genre_output)
        # we have dropout to prevent overfitting, the .2 is a low prob of dropping node.
        self.drop_one = nn.Dropout(0.2)
        # we use ReLu as our activation
        self.non_linear_one = torch.nn.ReLU()

        # this is the forward pass, we depend on the non-linear activation function
    def forward(self, x):
        x = self.lin_one(x)
        x = self.non_linear_one(x)
        x = self.drop_one(x)
        x = self.lin_two(x)
        x = self.non_linear_one(x)
        x = self.drop_one(x)
        x = self.lin_three(x)
        return x