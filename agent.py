import torch.nn as nn

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(45, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.seq(x)

