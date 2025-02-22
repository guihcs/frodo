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


class FP(nn.Module):
    def __init__(self, n_dim, a_space, m_space):
        super(FP, self).__init__()


        self.fl = nn.Sequential(
            nn.Linear(n_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.fh = nn.Sequential(
            nn.Linear(m_space, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.dec_l = nn.TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        self.dec = nn.TransformerDecoder(self.dec_l, num_layers=4)

        self.pe = pos_encode(15, 128)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, a_space),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, h):
        nx = self.fl(x)
        nh = self.fh(h) + self.pe[:h.shape[1], :].unsqueeze(0).to(h.device)
        # mask = torch.ones((nx.shape[1], nh.shape[1]))
        # mask = torch.triu(mask, diagonal=1).bool().to(x.device)
        hidden = self.dec(nx.unsqueeze(1), nh).mean(dim=1)
        return self.fc(hidden)