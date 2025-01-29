from board2 import Board2


class ActionController:

    def __init__(self, board: Board2):
        self.board = board
        self.push_cells = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
        self.mw_cells = [(1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
        self.move_directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    def is_win(self):
        return self.board.grid[-4, 0, 0] == 1 and self.board.grid[-9, 1, 0] > 0 or self.board.grid[-4, 0, 2] == 1 and self.board.grid[-9, 1, 2] > 0

    def is_lose(self):
        return self.board.grid[-3, 0, 0] == 1 and self.board.grid[-9, 1, 0] > 0 or self.board.grid[-3, 0, 2] == 1 and self.board.grid[-9, 1, 2] > 0

    def is_block(self):
        return self.board.grid[-4, 0, 0] == 0 and self.board.grid[-9, 1, 0] > 0 or self.board.grid[-4, 0, 2] == 0 and self.board.grid[-9, 1, 2] > 0

    def move(self, dy, dx):
        y, x = self.board.get_player_position()
        self.board.player_walk(y + dy, x + dx)

    def push_enemy(self, dy, dx):
        self.board.push_enemy(dy, dx)

    def push_todd(self, dy, dx):
        self.board.push_todd(dy, dx)

    def throw_mw(self, dy, dx):
        self.board.player_throw_mw(dy, dx)

    def execute_action(self, n):
        if n == 0:
            return
        if n < len(self.move_directions):
            dy, dx = self.move_directions[n - 1]
            self.move(dy, dx)
        elif n < len(self.move_directions) + len(self.push_cells):
            dy, dx = self.push_cells[n - len(self.move_directions) - 1]
            self.push_enemy(dy, dx)
        elif n < len(self.move_directions) + len(self.push_cells) * 2:
            dy, dx = self.push_cells[n - len(self.move_directions) - len(self.push_cells) - 1]
            self.push_todd(dy, dx)
        else:
            dy, dx = self.mw_cells[n - len(self.move_directions) - len(self.push_cells) * 2 - 1]
            self.throw_mw(dy, dx)

    def explain_action(self, n):
        if n == 0:
            return "Do nothing"
        if n < len(self.move_directions):
            dy, dx = self.move_directions[n - 1]
            return f"Move {dy} {dx}"
        elif n < len(self.move_directions) + len(self.push_cells):
            dy, dx = self.push_cells[n - len(self.move_directions) - 1]
            return f"Push enemy {dy} {dx}"
        elif n < len(self.move_directions) + len(self.push_cells) * 2:
            dy, dx = self.push_cells[n - len(self.move_directions) - len(self.push_cells) - 1]
            return f"Push Todd {dy} {dx}"
        else:
            dy, dx = self.mw_cells[n - len(self.move_directions) - len(self.push_cells) * 2 - 1]
            return f"Throw MW {dy} {dx}"

    def get_action_space(self):
        return len(self.move_directions) + len(self.push_cells) * 2 + len(self.mw_cells)