import math
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from board import Board
from agent import Agent, to_one_hot, bae
from heapq import heappush, heappop
from tree import Tree
# set white background
plt.rcParams['figure.facecolor'] = 'white'
import time


def get_best_action(board, memory, agent, timestep=200, max_depth=5, max_iter=100):
    root = Tree(board, memory=memory)

    res, iterations = root.search(agent, max_depth=max_depth, max_iter=max_iter, timestep=timestep)

    n = res
    i = 0
    while n.parent is not None and n.parent.parent is not None:
        n = n.parent
        i += 1

    return n.action, iterations


def evaluate(agent, n=100, max_steps=20, board_stack=2, timestep=200, max_depth=5, max_iter=100):
    v = 0
    fh = []
    iterations = 0
    for _ in range(n):
        board = Board()
        board.board = [[0, 1, 0],
                       [0, 2, 0],
                       [1, 1, 3]]
        bs = bae(board, 0).repeat(board_stack, 1)
        ah = []
        for _ in range(max_steps):
            a, it = get_best_action(board, bs, agent, timestep=timestep, max_depth=max_depth, max_iter=max_iter)
            iterations += it
            be = bae(board, a)
            bs = torch.cat([bs[1:], be])
            with torch.no_grad():
                value, pl = agent(bs.unsqueeze(0))
            board.do_action(a)
            if pl.exp().argmax() == a:
                ah.append(1)
            else:
                ah.append(0)
            if board.is_win():
                fh.extend(ah)
                v += 1
                break
            elif board.board[1][1] == 4 and board.board[0][1] != 3:
                break

            board.step(timestep)

    return v / n, sum(fh) / (len(fh) if len(fh) > 0 else 1), iterations


board_stack = 8
time_step = 100
max_iter = 200
max_depth = 20
max_steps = 60
agent = Agent(60, Board().get_total_actions(), max_len=board_stack, nhead=4, num_layers=1)
agent.load_state_dict(torch.load('agent.pt'))

value, act, its = evaluate(agent, max_steps=max_steps, board_stack=board_stack, timestep=time_step, max_iter=max_iter,
                           max_depth=max_depth)