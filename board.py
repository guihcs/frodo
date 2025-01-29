from termcolor import colored
import copy
import torch
import torch.nn as nn
import random
from heapq import heappush, heappop
import time
import math


def tr(v):
    f, w = math.modf(v)
    if f >= 0.5:
        return math.ceil(v)
    else:
        return math.floor(v)


def get_empty_neighbours(board, position):
    neighbours = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0: continue
            if board.is_empty((position[0] + i, position[1] + j)):
                neighbours.append((position[0] + i, position[1] + j))
    return neighbours


def sqr_distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


class Path:

    def __init__(self, board, position=None, parent=None, cost=0):
        self.board = board
        self.position = position
        self.parent = parent
        self.cost = cost
        pass

    def search(self, max_iter=1000):

        self.position = self.board.get_position(2)

        end = self.board.get_position(3)

        q = []

        heappush(q, (self.cost, time.time(), self))

        for _ in range(max_iter):
            if len(q) == 0:
                return None
            current = heappop(q)[-1]

            if sqr_distance(current.position, end) <= 2:
                return current

            for neighbour in get_empty_neighbours(current.board, current.position):
                new_board = current.board.copy()
                new_board.move(current.position, neighbour)
                path = Path(new_board, neighbour, current, current.cost + sqr_distance(current.position, neighbour))
                heappush(q, (path.cost + sqr_distance(current.position, end), time.time(), path))

        return None


def get_path(board):
    path = Path(board)

    solution = path.search()
    if solution is None:
        return None
    path = []

    while solution.parent is not None:
        path.append(solution.position)
        solution = solution.parent
    path.reverse()
    return path


init_board = [[1, 0, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 1, 1],
              ]

yv, xv = torch.where(torch.Tensor(init_board) == 1)
init_rp = list(zip(yv.tolist(), xv.tolist()))

mw_mask = [[0, 0, 0, 0],
           [1, 0, 1, 0],
           [1, 1, 1, 1],
           ]

yv, xv = torch.where(torch.Tensor(mw_mask) == 1)
mw_p = list(zip(yv.tolist(), xv.tolist()))

init_actions = {}

init_actions[0] = lambda self: None

move_matrix = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

al = len(init_actions)
for i, move in enumerate(move_matrix):
    init_actions[i + al] = lambda self, mv=move: self.walk(mv)

al = len(init_actions)
for i, rp in enumerate(init_rp):
    init_actions[i + al] = lambda self, r=rp: self.push(r)

# init_actions[14] = lambda self, r=rp: self.throw_mw((1, 1))


al = len(init_actions)
for i, rp in enumerate(mw_p):
    init_actions[i + al] = lambda self, r=rp: self.throw_mw(r)


class Board:

    def __init__(self, board=None):
        self.board = board

        if self.board is None:
            self.board = copy.deepcopy(init_board)

            tp = random.sample(init_rp, k=2)
            self.board[tp[1][0]][tp[1][1]] = 3
            self.board[tp[0][0]][tp[0][1]] = 2
        self.actions = init_actions
        self.pushs = {}
        self.wp = None

    def is_action_possible(self, action):
        pp = self.get_position(2)

        if 1 <= action < 9:
            return self.can_move((pp[0] + move_matrix[action - 1][0], pp[1] + move_matrix[action - 1][1]))
        elif action == self.get_total_actions() - 1:
            return self.is_empty((1, 1))
        return True

    def is_empty(self, np):
        if np[0] < 0 or np[0] > len(self.board) - 1 or np[1] < 0 or np[1] > len(self.board[0]) - 1:
            return False
        if self.board[np[0]][np[1]] != 1:
            return False
        return True

    def can_move(self, np):
        return self.is_empty(np)

    def can_throw_mw(self, np):

        if not self.is_empty(np):
            return False

        pp = self.get_position(2)

        dx = np[1] - pp[1]
        dy = np[0] - pp[0]

        if abs(dx) > abs(dy):
            if pp[1] > np[1]:
                pp, np = np, pp

            a = (np[0] - pp[0]) / (np[1] - pp[1])
            y = pp[0]
            for x in range(pp[1], np[1] + 1):
                if self.board[tr(y)][x] == 4:
                    return False
                y += a

        else:
            if pp[0] > np[0]:
                pp, np = np, pp

            a = (np[1] - pp[1]) / (np[0] - pp[0])
            x = pp[1]

            for y in range(pp[0], np[0] + 1):
                if self.board[y][tr(x)] == 4:
                    return False
                x += a

        return True

    def move(self, oldp, np):
        if not self.can_move(np):
            return
        self.board[np[0]][np[1]] = self.board[oldp[0]][oldp[1]]
        self.board[oldp[0]][oldp[1]] = 1

    def push(self, rl):

        pp = self.get_position(2)
        tp = self.get_position(3)

        if (pp[0] - tp[0]) ** 2 + (pp[1] - tp[1]) ** 2 > 2:
            self.wp = get_path(self)
            if self.wp is not None:
                if 2 not in self.pushs:
                    self.pushs[2] = [None, [3, rl, 1350]]
                else:
                    self.pushs[2][1] = [3, rl, 1350]

        else:
            if 2 not in self.pushs:
                self.pushs[2] = [[3, rl, 1420], None]
            else:
                self.pushs[2][0] = [3, rl, 1420]

            if self.pushs[2][1] is not None and self.pushs[2][0][2] > self.pushs[2][1][2]:
                tmp = self.pushs[2][0][2]
                self.pushs[2][0][2] = self.pushs[2][1][2]
                self.pushs[2][1][2] = tmp

    def walk(self, dr):
        pp = self.get_position(2)
        np = (pp[0] + dr[0], pp[1] + dr[1])

        if not self.can_move(np):
            return

        self.pushs.pop(2, None)
        self.move(pp, np)

    def do_action(self, a):
        self.actions[a](self)

    def get_total_actions(self):
        return len(self.actions)

    def throw_mw(self, rl):
        if self.board[rl[0]][rl[1]] != 1:
            return
        self.board[rl[0]][rl[1]] = 4

    def get_position(self, p):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == p:
                    return i, j

    def copy(self):
        new_board = Board()
        new_board.board = copy.deepcopy(self.board)
        new_board.pushs = copy.deepcopy(self.pushs)
        return new_board

    def is_win(self):
        return self.board[0][0] == 3 and self.board[1][0] == 4 or self.board[0][2] == 3 and self.board[1][2] == 4

    def is_loss(self):
        return self.board[0][0] == 2 and self.board[1][0] == 4 or self.board[0][2] == 2 and self.board[1][2] == 4

    def to_tensor(self):
        return torch.flatten(nn.functional.one_hot(torch.LongTensor(self.board), 5).float())

    def step(self, millis=200, wt=50):
        for k, v in list(self.pushs.items()):
            for i, v2 in list(enumerate(v)):
                if v2 is None:
                    continue

                if i == 1 and self.wp is not None:
                    rmn = millis
                    while rmn > 0 and len(self.wp) > 0:
                        rmn -= wt
                        mv = self.wp.pop(0)
                        pp = self.get_position(2)
                        self.move(pp, mv)
                    if len(self.wp) == 0:
                        self.wp = None

                    else:
                        continue
                    millis = rmn
                v2[2] -= millis
                if v2[2] <= 0:
                    tp = self.get_position(v2[0])
                    rl = v2[1]
                    if self.board[rl[0]][rl[1]] != 1:
                        return

                    if (tp[0] - rl[0]) ** 2 + (tp[1] - rl[1]) ** 2 > 2:
                        return
                    self.move(tp, rl)
                    self.pushs[k][i] = None

                    if self.pushs[k][0] is None and self.pushs[k][1] is None:
                        self.pushs.pop(k, None)

            if len(v) == 0:
                self.pushs.pop(k, None)

    def __repr__(self):
        txt = ''
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    txt += colored('%', 'yellow') + ' '
                elif self.board[i][j] == 1:
                    txt += colored('.', 'white') + ' '
                elif self.board[i][j] == 2:
                    txt += colored('P', 'green') + ' '
                elif self.board[i][j] == 3:
                    txt += colored('E', 'red') + ' '
                elif self.board[i][j] == 4:
                    txt += colored('#', 'blue') + ' '
            txt += '\n'

        return txt

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return self.board == other.board and self.pushs == other.pushs

    def __hash__(self):
        return hash(str(self.board) + str(self.pushs))


if __name__ == '__main__':
    board = Board()
    board.board = [[1, 0, 3, 0],
                   [1, 0, 1, 0],
                   [2, 1, 1, 1],
                   ]

    print(board.get_total_actions())
    board.do_action(18)
    print(board.is_win())
    print(board)
