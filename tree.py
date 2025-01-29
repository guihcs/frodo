from board import Board
from heapq import heappush, heappop
import math
import time
import torch
from agent import Agent, bae


class Tree:

    def __init__(self, board, action=0, depth=0, parent=None, cost=1e5, memory=None):
        self.board = board
        self.parent = parent
        self.action = action
        self.depth = depth
        self.cost = cost
        self.memory = memory

    def search(self, agent, max_depth=5, timestep=1000, max_iter=5000, visited=None):
        agent.eval()
        q = []
        heappush(q, (self.cost, time.time(), self))

        if visited is None:
            visited = {}
        frontier = {self.board}
        iterations = 0
        for _ in range(max_iter):
            if len(q) == 0:
                print(self.board)
                raise Exception("No solution found")

            node = heappop(q)[-1]
            frontier.remove(node.board)
            iterations += 1

            if node.depth == max_depth:
                continue

            if node.board.is_win():
                return node, iterations

            visited[node.board] = node

            with torch.no_grad():
                emb = node.board.to_tensor()
                value, pol = agent(emb.unsqueeze(0).unsqueeze(0), node.memory.unsqueeze(0))
                ep = value * pol.exp()

                expected_value = ep.squeeze(0).squeeze(0).tolist()

            for action, value in zip(range(node.board.get_total_actions()), expected_value):

                if not node.board.is_action_possible(action):
                    continue

                new_board = node.board.copy()
                new_board.do_action(action)
                new_board.step(timestep)

                if action == 0:
                    start_board = new_board.copy()
                    for _ in range(20):
                        if new_board.board != start_board.board:
                            break
                        new_board.step(timestep)

                emb = new_board.to_tensor().unsqueeze(0)
                nemb = torch.cat([node.memory[1:, :], emb], dim=0)

                t = Tree(new_board, action, node.depth + 1, node, node.cost + 1, memory=nemb)
                if new_board.is_win():
                    value = 15

                if new_board in visited and t.cost < visited[new_board].cost:
                    visited[new_board].cost = t.cost
                    visited[new_board].parent = t.parent
                    visited[new_board].action = t.action
                    visited[new_board].action = t.memory

                elif new_board not in visited and new_board not in frontier:
                    frontier.add(new_board)
                    heappush(q, (t.cost - value, time.time(), t))

        return heappop(q)[-1], iterations