from typing import Union
import math
from fastapi import FastAPI
from agent import Agent
from board import Board
import torch

app = FastAPI()


agent = Agent()
agent.load_state_dict(torch.load('agent.pt'))
agent.eval()

def get_best_action(board, agent):
    pp = board.get_position(2)
    tp = board.get_position(3)

    best_action = None
    best_value = -math.inf

    for a in range(board.get_total_actions()):
        board_copy = board.copy()
        board_copy.do_action(pp, tp, a)

        with torch.no_grad():
            agent.eval()
            value = agent(board_copy.to_tensor()).item()

        if value > best_value:
            best_value = value
            best_action = a

    return best_action

@app.post("/")
def get_action(data: dict):
    board_data = torch.LongTensor(data['board']).reshape((3, 3)).tolist()
    board = Board()
    board.board = board_data
    action = get_best_action(board, agent)
    if board.is_win():
        return {'action': -1}
    return {'action': action}

