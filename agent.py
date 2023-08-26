import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, in_size):
        super(Agent, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.seq(x)

