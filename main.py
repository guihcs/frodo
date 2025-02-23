from typing import Union
import math
from fastapi import FastAPI
from agent import FP
from board3 import Board3
from controller3 import ActionController
from mcts import NT
import torch
import torch.nn as nn
import itertools
import time
from mcts import search
from nnl import to_emb, emb_mem
app = FastAPI()


board_stack = 10
time_step = 200
max_depth = 20
max_iter = 5000


b = Board3(walk_time=200)
x = to_emb(b)


hl = []
for b, a in [(b, 0), (b, 0)]:
    board_embedding = to_emb(b)
    action_embedding = nn.functional.one_hot(torch.LongTensor([a]), ActionController(b).get_action_space())
    hl.append(torch.cat([board_embedding, action_embedding], dim=1))
hl = torch.cat(hl, dim=0).unsqueeze(0)

fp = FP(x.shape[1], ActionController(b).get_action_space(), hl.shape[-1])
fp.load_state_dict(torch.load('models/fp.pth', map_location=torch.device('cpu')), strict=False)
fp.eval()
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
    b = ActionController.board_from_data(board_data)
    b.walk_time = 200
    history = [(ActionController.board_from_data(x), y) for x, y in data['history']]
    be = to_emb(b)
    me = emb_mem(history)
    with torch.no_grad():
        o, _ = fp(be, me)
        action = torch.argmax(o).item()
    # bc = list(itertools.batched(board_data, 4))
    # board = ActionController.board_from_data(bc, history, time_step=1000)
    # board.walk_time = 500
    controller = ActionController(b)
    # controller.execute_action(action)
    # b.step(500, walk_time=200)
    #
    if controller.is_win() or controller.is_lose() or controller.is_block():
        return {'action': -1}
    #
    # path = search(board, 8000)
    # if path is None:
    #     print('no path')
    #     return {'action': -1}
    #
    # return {'action': path[0][1]}
    return {'action': action}
