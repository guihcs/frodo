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
from game_utils import get_best_action, evaluate
from heapq import heappush, heappop
from tree import Tree
# set white background
plt.rcParams['figure.facecolor'] = 'white'
import time

# set white background
plt.rcParams['figure.facecolor'] = 'white'



board_stack = 8
time_step = 500
max_iter = 5000
max_depth = 15
max_steps = 40
agent = Agent(45, 60, Board().get_total_actions(), max_len=board_stack, nhead=4, num_layers=1)
agent.load_state_dict(torch.load('agent.pt'))

mem_size = 300
epochs = 2000
data_points = 10

optimizer = optim.AdamW(agent.parameters(), lr=0.00001, weight_decay=0.01)
crit = nn.MSELoss()

boards_memory = []
memories_memory = []
actions_memory = []
rewards_memory = []

best_value = -math.inf
best_it = math.inf
lh = []
lq = []
gq = []
wq = []
iterations = []
actq = []

for e in tqdm(range(epochs)):

    if e % (epochs / data_points) == 0:
        value, act, its = evaluate(agent, n=100, use_tree=True, max_steps=max_steps, board_stack=board_stack,
                                   timestep=time_step, max_iter=max_iter, max_depth=max_depth)
        if value >= best_value and its < best_it:
            best_value = value
            torch.save(agent.state_dict(), 'models/g200.pt')
        lh.append(value)
        actq.append(act)
        iterations.append(its)

    board = Board()
    # board.board = [[0, 3, 0],
    #                [0, 1, 0],
    #                [1, 2, 1]]
    boards = []
    actions = []
    memories = []
    rewards = []
    bs = bae(board, 0).repeat(board_stack, 1)

    for _ in range(max_steps):
        if random.random() < 0.1:
            a = random.randint(0, board.get_total_actions() - 1)
        else:

            a, _ = get_best_action(board, bs, agent, timestep=time_step, max_depth=max_depth, max_iter=max_iter)

        bc = board.copy()
        boards.append(bc.to_tensor())
        memories.append(bs)
        actions.append(a)
        board.do_action(a)

        be = bae(board, a)
        bs = torch.cat([bs[1:], be])

        if board.is_win():
            rewards.append(15)
            break

        elif board.board[1][1] == 4 and board.board[0][1] != 3:
            rewards.append(-10)
            break
        else:
            rewards.append(-1)

        board.step(time_step)

    acc_reward = 0
    agent.train()
    for b, m, a, r in reversed(list(zip(boards, memories, actions, rewards))):
        acc_reward += r
        boards_memory.append(b.unsqueeze(0).unsqueeze(0))
        memories_memory.append(m.unsqueeze(0))
        actions_memory.append(torch.Tensor([a]).long().unsqueeze(0))
        rewards_memory.append(torch.Tensor([acc_reward]).unsqueeze(0))

    if len(boards_memory) > mem_size:
        boards_memory = boards_memory[len(boards_memory) - mem_size:]
        memories_memory = memories_memory[len(memories_memory) - mem_size:]
        actions_memory = actions_memory[len(actions_memory) - mem_size:]
        rewards_memory = rewards_memory[len(rewards_memory) - mem_size:]

    el = []
    gl = []
    ws = []

    for _ in range(2):
        for b, m, a, r in DataLoader(
                TensorDataset(torch.cat(boards_memory), torch.cat(memories_memory), torch.cat(actions_memory),
                              torch.cat(rewards_memory)), batch_size=32,
                shuffle=True):
            optimizer.zero_grad()
            pred, pol = agent(b, m)
            l1 = crit(pred, r)
            cat = Categorical(pol)
            l2 = torch.mean(-cat.log_prob(a.squeeze(1)).unsqueeze(1) * r)
            loss = l1 + l2
            loss.backward()
            for p in agent.parameters():
                if p.grad is not None:
                    gl.append(p.grad.data.norm(2).item())
                ws.append(p.data.norm(2).item())
            optimizer.step()
            el.append(loss.item())

    if e % (epochs / data_points) == 0:
        lq.append(sum(el) / len(el))
        gq.append(sum(gl) / len(gl))
        wq.append(sum(ws) / len(ws))

fig, ax = plt.subplots(1, 6, figsize=(20, 5))

ax[0].plot(lh, label='win rate')
ax[1].plot(actq, label='action rate')
ax[2].plot(iterations, label='iterations')
ax[3].plot(lq, label='loss', color='red')
ax[4].plot(gq, label='grad norm', color='green')
ax[5].plot(wq, label='weight norm', color='orange')

# add title for each subplot
for i in range(6):
    ax[i].set_title(ax[i].get_xlabel(), fontsize=14)
    ax[i].set_xlabel('epochs', fontsize=12)
    ax[i].legend(fontsize=12)

plt.show()
