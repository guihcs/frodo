from board3 import Board3, empty_cells
from controller3 import ActionController

import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from nnl import to_emb

import matplotlib.pyplot as plt

from tqdm.auto import tqdm


def step(b, a, time=1600):
    bc = b.copy()
    nc = ActionController(bc)

    nc.execute_action(a)
    bc.step(time)

    reward = 0
    end = False
    if nc.is_win():
        reward = 1
        end = True
    elif nc.is_lose():
        reward = -1
        end = True
    elif nc.is_block():
        reward = -1
        end = True

    return bc, reward, end





def gen_init_board():
    b = Board3(walk_time=500)

    # b.players_positions = [(0, 2), (1, 0), (3, 3)]

    return b


m = {}


e = 1.0
for i in tqdm(range(2000)):

    e = max(0.1, e - 0.001 * i)

    b = gen_init_board()

    h = []

    for _ in range(50):

        if b not in m:
            m[b] = [0.5 for _ in range(ActionController.get_action_space())]

        if random.random() < e:
            a = random.randint(0, ActionController.get_action_space() - 1)
        else:
            a = max(range(ActionController.get_action_space()), key=lambda x: m[b][x])

        bc, r, end = step(b, a, 1000)

        h.append((b.copy(), a, r))

        # if end:
        #     y = r
        # else:
        #     if bc not in m:
        #         m[bc] = [0.5 for _ in range(ActionController.get_action_space())]
        #     y = r + 0.85 * max(m[bc])
        #
        # m[b][a] += 0.1 * (y - m[b][a])

        if end:
            break

        b = bc
    accr = 0
    for b, a, r in h[::-1]:
        accr = r + 0.85 * accr
        m[b][a] += 0.1 * (accr - m[b][a])

print(len(m))
