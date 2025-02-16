from typing import Union
import math
from fastapi import FastAPI
from agent import Agent, bae
from board3 import Board3
from controller3 import ActionController
from mcts import NT
import torch
import torch.nn as nn
import itertools
import time
from mcts import search
app = FastAPI()


board_stack = 10
time_step = 200
max_depth = 20
max_iter = 5000



# agent = Agent(Board().to_tensor().shape[0], 60, Board().get_total_actions(), max_len=board_stack, nhead=4, num_layers=1)
# agent.load_state_dict(torch.load('models/g200.pt'))
# agent.eval()
# agent.cuda(0)

bs = None

# board = Board()

last_time = time.time()

@app.post("/")
def get_action(data: dict):
    board_data = data['board']
    # timers = list(itertools.batched(data['mw_timers'], 4))
    # push_timers = data['pushTimers']
    # push_location = data['pushLocations']
    history = [(list(itertools.batched(x, 4)), y) for x, y in data['history']]
    bc = list(itertools.batched(board_data, 4))
    board = ActionController.board_from_data(bc, history, time_step=1000)
    board.walk_time = 500
    controller = ActionController(board)

    if controller.is_win() or controller.is_lose() or controller.is_block():
        return {'action': -1}

    path = search(board, 8000)
    if path is None:
        print('no path')
        return {'action': -1}

    return {'action': path[0][1]}
