import torch
import torch.nn as nn
import math

def pos_encode(max_len, d_model):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(0)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Agent(nn.Module):
    def __init__(self, bs, in_size, out_size, nhead=4, dim_f=128, num_layers=1, max_len=5):
        super(Agent, self).__init__()

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=in_size, nhead=nhead, dim_feedforward=dim_f, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.pe = pos_encode(max_len, in_size)

        self.seq = nn.Sequential(
            nn.Linear(bs, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, in_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.hidden = nn.Sequential(
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
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, mem):
        x = self.seq(x)
        mem = self.seq(mem)
        mem += self.pe[:mem.shape[1], :].unsqueeze(0).to(mem.device)
        mask = torch.ones((x.shape[1], mem.shape[1]))
        mask = torch.triu(mask, diagonal=1).bool().to(x.device)
        h = self.decoder(x, mem, memory_mask=mask).sum(dim=1)
        h = self.hidden(h)
        return self.vl(h), self.pl(h)


def bae(b, a):
    return torch.cat([b.to_tensor(), to_one_hot(a, b.get_total_actions())]).unsqueeze(0)


def to_one_hot(x, n):
    oh = torch.zeros(n)
    oh[x] = 1
    return oh

# agent = Agent(45, 60, 15, max_len=5, nhead=4, num_layers=1)
#
#
# v, p = agent(torch.rand((1, 1, 45)), torch.rand((1, 5, 60)))
#
# print(v.shape, p.shape)