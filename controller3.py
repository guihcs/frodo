from board3 import Board3

PUSH_CELLS = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
TODD_PUSH_CELLS = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
MW_CELLS = [(1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
MOVE_DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


class ActionController:

    def __init__(self, board: Board3):
        self.board = board

    @staticmethod
    def board_from_data(data, history, time_step=200):
        b = Board3()
        b.mw = set()
        c = ActionController(b)

        if len(history) > 0:
            fb = history[0][0]

            for i in range(4):
                for j in range(4):
                    if fb[i][j] == 1:
                        b.set_player(i, j)
                    elif fb[i][j] == 2:
                        b.set_enemy(i, j)
                    elif fb[i][j] == 3:
                        b.set_todd(i, j)
                    elif fb[i][j] == 4:
                        b.mw.add((i, j))

            for _, a in history:
                c.execute_action(a)
                b.step(time_step)

            b.mw = set()

        for i in range(4):
            for j in range(4):
                if data[i][j] == 1:
                    b.set_player(i, j)
                elif data[i][j] == 2:
                    b.set_enemy(i, j)
                elif data[i][j] == 3:
                    b.set_todd(i, j)
                elif data[i][j] == 4:
                    b.mw.add((i, j))

        return b

    def is_win(self):
        c1 = self.board.get_enemy_position() == (0, 0) and (1, 0) in self.board.mw
        c2 = self.board.get_enemy_position() == (0, 2) and (1, 2) in self.board.mw
        return c1 or c2

    def is_lose(self):
        c1 = self.board.get_player_position() == (0, 0) and (1, 0) in self.board.mw
        c2 = self.board.get_player_position() == (0, 2) and (1, 2) in self.board.mw
        return c1 or c2

    def is_block(self):
        c1 = self.board.get_player_position() != (0, 0) and self.board.get_enemy_position() != (0, 0) and (
        1, 0) in self.board.mw
        c2 = self.board.get_player_position() != (0, 2) and self.board.get_enemy_position() != (0, 2) and (
        1, 2) in self.board.mw
        return c1 or c2

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
        if n < 1 + len(MOVE_DIRECTIONS):
            dy, dx = MOVE_DIRECTIONS[n - 1]
            self.move(dy, dx)
        elif n < 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS):
            dy, dx = PUSH_CELLS[n - len(MOVE_DIRECTIONS) - 1]
            self.push_enemy(dy, dx)
        elif n < 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS) + len(TODD_PUSH_CELLS):
            dy, dx = TODD_PUSH_CELLS[n - len(MOVE_DIRECTIONS) - len(PUSH_CELLS) - 1]
            self.push_todd(dy, dx)
        else:
            dy, dx = MW_CELLS[n - len(MOVE_DIRECTIONS) - len(PUSH_CELLS) - len(TODD_PUSH_CELLS) - 1]
            self.throw_mw(dy, dx)

    @staticmethod
    def explain_action(n):
        if n == 0:
            return "Skip"
        if n < 1 + len(MOVE_DIRECTIONS):
            dy, dx = MOVE_DIRECTIONS[n - 1]
            return f"Move {dy} {dx}"
        elif n < 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS):
            dy, dx = PUSH_CELLS[n - len(MOVE_DIRECTIONS) - 1]
            return f"Push E {dy} {dx}"
        elif n < 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS) + len(TODD_PUSH_CELLS):
            dy, dx = TODD_PUSH_CELLS[n - len(MOVE_DIRECTIONS) - len(PUSH_CELLS) - 1]
            return f"Push T {dy} {dx}"
        else:
            dy, dx = MW_CELLS[n - len(MOVE_DIRECTIONS) - len(PUSH_CELLS) - len(TODD_PUSH_CELLS) - 1]
            return f"MW {dy} {dx}"

    @staticmethod
    def get_action_space():
        return len(MOVE_DIRECTIONS) + len(PUSH_CELLS) + len(TODD_PUSH_CELLS) + len(MW_CELLS) + 1

    def get_available_moves(self):
        moves = []
        py, px = self.board.get_player_position()
        for n in range(self.get_action_space()):
            if n == 0:
                moves.append(0)
            elif n < len(MOVE_DIRECTIONS) + 1:
                dy, dx = MOVE_DIRECTIONS[n - 1]

                if self.board.can_move(py + dy, px + dx):
                    moves.append(n)
            elif n < 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS):
                moves.append(n)
            elif n < 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS) + len(TODD_PUSH_CELLS):
                moves.append(n)
            else:
                dy, dx = MW_CELLS[
                    n - len(MOVE_DIRECTIONS) - len(PUSH_CELLS) - len(TODD_PUSH_CELLS) - 1]
                if self.board.can_throw_mw(py, px, dy, dx):
                    moves.append(n)

        return moves

    def execute_action_from_name(self, name):
        actions = {self.explain_action(i): i for i in range(self.get_action_space())}
        self.execute_action(actions[name])

    @staticmethod
    def get_mw_range():
        return 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS) + len(TODD_PUSH_CELLS), ActionController.get_action_space()

    @staticmethod
    def get_push_range():
        return 1 + len(MOVE_DIRECTIONS), 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS)

    @staticmethod
    def get_todd_push_range():
        return 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS), 1 + len(MOVE_DIRECTIONS) + len(PUSH_CELLS) + len(TODD_PUSH_CELLS)
