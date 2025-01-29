from typing import Union
import math
from fastapi import FastAPI
from agent import Agent, bae
from board import Board
import torch
from game_utils import get_best_action
import torch.nn as nn
from tree import Tree
import time
app = FastAPI()


board_stack = 10
time_step = 200
max_depth = 20
max_iter = 5000



agent = Agent(Board().to_tensor().shape[0], 60, Board().get_total_actions(), max_len=board_stack, nhead=4, num_layers=1)
agent.load_state_dict(torch.load('models/g200.pt'))
agent.eval()
# agent.cuda(0)

bs = None

board = Board()

last_time = time.time()

@app.post("/")
def get_action(data: dict):
    global bs
    global board
    global last_time
    # print(time.time() - last_time)
    last_time = time.time()


    board_data = torch.LongTensor(data['board']).reshape((3, 4)).tolist()
    board.board = board_data
    # print(board.board)
    # print(board.pushs)

    if board.is_win() or board.is_loss():
        return {'action': -1}

    if bs is None:
        bs = board.to_tensor().unsqueeze(0).repeat(board_stack, 1)


    action, _ = get_best_action(board, bs, agent, timestep=time_step, max_depth=max_depth, max_iter=max_iter)
    be = board.to_tensor().unsqueeze(0)
    bs = torch.cat([bs[1:], be])

    board.do_action(action)
    board.step(time_step)
    # print(action)
    # print(board.board)
    # print(board.pushs)


    return {'action': action}

