from termcolor import colored
import copy
import torch
import torch.nn as nn
import random


class Board:

    def __init__(self, board=None):
        if board is None:
            self.board = [[ 0, 1, 0],
                      [ 0, 1, 0],
                      [ 1, 1, 1],
                      ]

        yv, xv = torch.where(torch.Tensor(self.board) == 1)
        self.rp = list(zip(yv.tolist(), xv.tolist()))
        tp = random.sample(self.rp, k=2)
        self.board[tp[0][0]][tp[0][1]] = 2
        self.board[tp[1][0]][tp[1][1]] = 3
        self.actions = {}

        self.actions[0] = lambda x, y: None

        move_matrix = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

        al = len(self.actions)
        for i, move in enumerate(move_matrix):
            self.actions[i + al] = lambda x, y, move=move: self.move(x, (x[0] + move[0], x[1] + move[1]))

        al = len(self.actions)
        for i, rp in enumerate(self.rp):
            self.actions[i + al] = lambda x, y, rp=rp: self.push(x, y, rp)

        self.actions[len(self.actions)] = lambda x, y: self.throw_mw(x, (1, 1))
        # self.actions[len(self.actions)] = lambda x, y: self.throw_mw(x, (1, 2))

    def move(self, oldp, np):
        if np[0] < 0 or np[0] > len(self.board) - 1 or np[1] < 0 or np[1] > len(self.board[0]) - 1:
            return
        if self.board[np[0]][np[1]] != 1:
            return
        self.board[np[0]][np[1]] = self.board[oldp[0]][oldp[1]]
        self.board[oldp[0]][oldp[1]] = 1

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

    def push(self, pp, tp, rl):
        if self.board[rl[0]][rl[1]] != 1:
            return
        if (pp[0] - tp[0]) ** 2 + (pp[1] - tp[1]) ** 2 > 2:
            return
        if (tp[0] - rl[0]) ** 2 + (tp[1] - rl[1]) ** 2 > 2:
            return
        self.move(tp, rl)

    def do_action(self, pp, tp, a):
        self.actions[a](pp, tp)

    def get_total_actions(self):
        return len(self.actions)

    def throw_mw(self, pp, rl):
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
        return new_board

    def is_win(self):
        return self.board[0][1] == 3 and self.board[1][1] == 4

    def to_tensor(self):
        return torch.flatten(nn.functional.one_hot(torch.LongTensor(self.board), 5).float())


if __name__ == '__main__':
    board = Board()
    print(board.get_total_actions())



