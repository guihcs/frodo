#%%
from heapq import heappush, heappop

from board2 import Board2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from controller import ActionController
from tqdm.auto import tqdm
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt
import math
import time
#%%
class Node:
    def __init__(self, board, player):
        self.board = board
        self.controller = ActionController(board)
        self.children = []
        self.p = player
        self.qsa = 0
        self.psa = 0
        self.nsa = 0
        self.a = None

    def search(self, player):
        winner = self.get_winner()
        if winner is not None:
            self.nsa += 1
            return winner

        if len(self.children) == 0:

            actions = []
            for a in self.controller.get_available_moves():
                board = self.board.copy()
                p = -self.p
                n = Node(board, p)
                n.a = a
                n.controller.execute_action(a)
                board.step()
                actions.append(a)
                self.children.append(n)

            boards = torch.cat([x.board.grid.clone().unsqueeze(0) for x in self.children], dim=0)

            with torch.no_grad():
                pl, vl = player(boards)

            for i, (n, a) in enumerate(zip(self.children, actions)):

                if n.get_winner() is not None:
                    n.qsa = n.get_winner()
                else:
                    n.qsa = vl[i].item()
                n.psa = pl.exp()[i, a].item()
                n.board.swap_enemy()

            self.qsa = torch.mean(vl).item()
            self.nsa = len(self.children)

            return self.qsa

        best = max(self.children, key=lambda x: x.qsa * x.p + 1.5 * x.psa * math.sqrt(sum([x.nsa for x in self.children]) / (x.nsa + 1)))

        res = best.search(player)

        self.qsa = (self.qsa * self.nsa + res) / (self.nsa + 1)
        self.nsa += 1

        return res

    def simulate(self, player):
        winner = self.get_winner()
        if winner is not None:
            return winner

        with torch.no_grad():
            pl, vl = player(self.board.grid.clone().unsqueeze(0))
        return vl.item()


    def get_winner(self):

        if self.controller.is_win():
            return 1
        elif self.controller.is_lose():
            return -1
        elif self.controller.is_block():
            return -1
        return None

    def get_improved_policy(self):
        s = sum([x.nsa for x in self.children])
        return [x.nsa / s for x in self.children]
#%%
def pos_encode(max_len, d_model, dtype=torch.float32):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.pos = pos_encode(9, 16)

        self.q = nn.Linear(16, 16)
        self.k = nn.Linear(16, 16)
        self.v = nn.Linear(16, 16)
        self.n_heads = 16
        self.n_dim = 16

        self.dropout = nn.Dropout(0.1)
        self.lc = nn.Linear(16, 16)


    def forward(self, x):
        wq = self._head_reshape(self.q(x))
        wk = self._head_reshape(self.k(x))
        wv = self._head_reshape(self.v(x))

        a = wq @ wk.transpose(2, 3) / math.sqrt(self.n_dim)
        a = torch.softmax(a, dim=-1)
        a = self.dropout(a)

        o = a @ wv

        return self.lc(self._head_reshape_back(o))

    def _head_reshape(self, x):
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.n_dim // self.n_heads).transpose(1, 2)

    def _head_reshape_back(self, x):
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.n_dim)

class Enc(nn.Module):
    def __init__(self):
        super(Enc, self).__init__()
        self.attn = Attention()
        self.ln1 = nn.LayerNorm(16)

        self.seq = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )

        self.ln2 = nn.LayerNorm(16)

    def forward(self, x):
        h = self.attn(x)
        h1 = self.ln1(h + x)
        h2 = self.seq(h1)
        return self.ln2(h1 + h2)

class FrodoPolicy(nn.Module):
    def __init__(self, action_space):
        super(FrodoPolicy, self).__init__()

        self.flatten = nn.Flatten(start_dim=1)
        # self.pos = pos_encode(9, 16)
        #
        # self.enc = nn.ModuleList([Enc() for _ in range(4)])
        #
        # self.pl = nn.Sequential(
        #     nn.Linear(16, action_space),
        #     nn.LogSoftmax(dim=-1)
        # )
        #
        # self.vl = nn.Linear(16, 1)

        self.enc = nn.Linear(144, 32)
        self.pl = nn.Sequential(
            nn.Linear(32, action_space),
            nn.LogSoftmax(dim=-1)
        )

        self.vl = nn.Linear(32, 1)



    def forward(self, x):
        fx = self.flatten(x)
        h = self.enc(fx)
        # px = self.pos.unsqueeze(0) + self.flatten(x)
        #
        # for enc in self.enc:
        #     px = enc(px)
        #
        # cv = px.mean(dim=1)
        #
        # return self.pl(cv), self.vl(cv)
        return self.pl(h), self.vl(h)




# board = Board2()
# controller = ActionController(board)
#
# player = FrodoPolicy(controller.get_action_space())
# print(player(board.grid.clone().unsqueeze(0)).exp().shape)
#%%

board = Board2()
print(board)
#%%
controller = ActionController(board)
player = FrodoPolicy(controller.get_action_space())
enemy = FrodoPolicy(controller.get_action_space())

root = Node(board, -1)
for i in range(1000):
    root.search(player)
#%%
