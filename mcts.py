from board3 import Board3, sqr_distance, is_adjacent
from controller3 import ActionController
import math
from heapq import heappush, heappop


def search(board, max_iterations=100, time_step=1000):
    q = [(0, 0, board, [])]
    visited = set()

    visited.add(board)
    for _ in range(max_iterations):

        if len(q) == 0:
            return None

        _, c, b, hist = heappop(q)
        ct = ActionController(b)

        if ct.is_win():
            path = []
            for b, a in hist:
                path.append((b, a))
            return path

        elif ct.is_lose() or ct.is_block():
            continue

        for a in ct.get_available_moves():
            b2 = b.copy()
            ct2 = ActionController(b2)
            ct2.execute_action(a)
            b2.step(time_step)

            if b2 in visited:
                continue

            visited.add(b2)
            qc = c + 1
            if a == 0:
                qc += 1

            if a >= ct2.get_mw_range()[0]:
                qc += 1

            pc = min(sqr_distance(*b2.get_enemy_position(), 0, 0), sqr_distance(*b2.get_enemy_position(), 0, 2))
            heappush(q, (qc + pc, qc, b2, hist + [(b, a)]))

    return None


class NT:

    def __init__(self, board, player):
        self.board = board
        self.controller = ActionController(board)
        self.children = None
        self.player = player
        self.a = None
        self.p = 0
        self.v = 0
        self.n = 0
        pass


    def get_winner(self):
        if self.controller.is_win():
            return 1
        if self.controller.is_lose():
            return -1
        if self.controller.is_block():
            return -1

        return None

    def search(self, fp):

        winner = self.get_winner()
        if winner is not None:
            self.n += 1
            self.v = -winner
            return self.v


        if self.children is None:
            self.expand(fp)

            return -self.v

        cs = sum([x.n for x in self.children])
        sv = max(self.children, key=lambda x: x.v + 1.5 * x.p * math.sqrt(cs / (x.n + 1)))
        res = sv.search(fp)
        self.v = (self.v * self.n + res) / (self.n + 1)
        self.n += 1

        return -res

    def expand(self, fp):
        self.children = []
        vs = []
        for p in self.controller.get_available_moves():
            board_copy = self.board.copy()
            nt = NT(board_copy, -self.player)
            nt.controller.execute_action(p)
            board_copy.step(500, walk_time=200)
            nt.a = p
            nt.v = 0
            nt.p = eval_board(board_copy)
            nt.n = 1
            vs.append(nt.p)
            board_copy.swap_enemy()
            self.children.append(nt)

        self.v = sum(vs) / len(vs)
        self.n = len(vs)




