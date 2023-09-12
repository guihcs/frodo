import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, in_size, out_size):
        super(Agent, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

        )

        self.vl = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        self.pl = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, a):
        h = self.seq(torch.cat([x, a], dim=-1))
        return self.vl(h), self.pl(h)

