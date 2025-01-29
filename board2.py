import torch
from termcolor import colored
import random
from heapq import heappush, heappop
import math

tod_cells = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
empty_cells = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]

FLOOR_LAYER = 8
TODD_LAYER = 7
PLAYER_LAYER = 6
ENEMY_LAYER = 5
SHORT_ENEMY_PUSH_LAYER = 4
LONG_ENEMY_PUSH_LAYER = 3
SHORT_TODD_PUSH_LAYER = 2
LONG_TODD_PUSH_LAYER = 1
MW_LAYER = 0

SHORT_ENEMY_PUSH_LAYER_ENEMY = 3
LONG_ENEMY_PUSH_LAYER_ENEMY = 2
SHORT_TODD_PUSH_LAYER_ENEMY = 1
LONG_TODD_PUSH_LAYER_ENEMY = 0


def tr(v):
    f, w = math.modf(v)
    if f >= 0.5:
        return math.ceil(v)
    else:
        return math.floor(v)


def is_adjacent(y1, x1, y2, x2):
    return abs(y1 - y2) < 2 and abs(x1 - x2) < 2


def sqr_distance(y1, y2, x1, x2):
    return (y1 - y2) ** 2 + (x1 - x2) ** 2


def step_push(grid, layer, time_step=200, push_time=1000):
    if len(grid[layer, :, :].nonzero()) <= 0:
        return
    py, px = grid[layer, :, :].nonzero().squeeze(0)
    if grid[layer, py, px] <= (time_step / push_time):
        grid[layer, py, px] = -1
    else:
        grid[layer, py, px] -= time_step / push_time


class Board2:

    def __init__(self):
        self.grid = torch.zeros(9, 4, 4)
        self.grid[FLOOR_LAYER, 0, 1] = 1
        self.grid[FLOOR_LAYER, 1, 1] = 1
        self.grid[FLOOR_LAYER, 0, 3] = 1
        self.grid[FLOOR_LAYER, 1, 3] = 1

        ty, tx = random.choice(tod_cells)
        self.grid[TODD_LAYER, ty, tx] = 1
        self.current_todd = 3000

        py, px = random.choice(list(set(empty_cells) - {(ty, tx)}))
        self.grid[PLAYER_LAYER, py, px] = 1

        ey, ex = random.choice(list(set(empty_cells) - {(ty, tx), (py, px)}))
        self.grid[ENEMY_LAYER, ey, ex] = 1

        self.current_player_mw = 0
        self.current_enemy_mw = 0

        self.enemy_push_grid = torch.zeros(4, 4, 4)

    def get_todd_position(self):
        return self.grid[TODD_LAYER, :, :].nonzero().squeeze(0)

    def set_player(self, y, x):
        self.grid[PLAYER_LAYER, :, :] = 0
        self.grid[PLAYER_LAYER, y, x] = 1

    def set_enemy(self, y, x):
        self.grid[ENEMY_LAYER, :, :] = 0
        self.grid[ENEMY_LAYER, y, x] = 1

    def set_todd(self, y, x):
        self.grid[TODD_LAYER, :, :] = 0
        self.grid[TODD_LAYER, y, x] = 1

    def clear_short_push(self):
        self.grid[SHORT_ENEMY_PUSH_LAYER, :, :] = 0
        self.grid[SHORT_TODD_PUSH_LAYER, :, :] = 0

    def clear_long_push(self):
        self.grid[LONG_ENEMY_PUSH_LAYER, :, :] = 0
        self.grid[LONG_TODD_PUSH_LAYER, :, :] = 0

    def reset_push(self):
        self.clear_short_push()
        self.clear_long_push()

    def move_tod(self, y, x):
        if not self.can_move_todd(y, x):
            return
        todd_pos = self.get_todd_position()
        self.grid[TODD_LAYER, todd_pos[0], todd_pos[1]] = 0
        self.grid[TODD_LAYER, y, x] = 1

    def move_player(self, y, x):
        if not self.can_move(y, x):
            return
        player_pos = self.get_player_position()
        self.grid[PLAYER_LAYER, player_pos[0], player_pos[1]] = 0
        self.grid[PLAYER_LAYER, y, x] = 1

    def move_enemy(self, y, x):
        if not self.can_move(y, x):
            return
        enemy_pos = self.get_enemy_position()
        self.grid[ENEMY_LAYER, enemy_pos[0], enemy_pos[1]] = 0
        self.grid[ENEMY_LAYER, y, x] = 1

    def player_walk(self, y, x):
        self.reset_push()
        self.move_player(y, x)

    def is_cell_empty(self, y, x):
        return not self.grid[FLOOR_LAYER, y, x].any() and not self.grid[TODD_LAYER, y, x].any() and not self.grid[PLAYER_LAYER, y, x].any() and not \
        self.grid[ENEMY_LAYER, y, x].any() and not self.grid[0, y, x].any()

    def get_empty_neighbours(self, y, x):
        neighbours = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0 or y + i < 0 or y + i > 3 or x + j < 0 or x + j > 3: continue
                if self.is_cell_empty(y + i, x + j):
                    neighbours.append((y + i, x + j))
        return neighbours

    def can_move_todd(self, y, x):
        if y < 1 or y >= 4 or x < 1 or x >= 4:
            return False
        if not self.is_cell_empty(y, x):
            return False

        return True

    def can_move(self, y, x):
        if y < 0 or y >= 4 or x < 0 or x >= 4:
            return False
        if not self.is_cell_empty(y, x):
            return False

        return True

    def get_todd_available_moves(self):
        y, x = self.get_todd_position()
        moves = []

        if self.can_move_todd(y - 1, x):
            moves.append((y - 1, x))
        if self.can_move_todd(y + 1, x):
            moves.append((y + 1, x))
        if self.can_move_todd(y, x - 1):
            moves.append((y, x - 1))
        if self.can_move_todd(y, x + 1):
            moves.append((y, x + 1))

        return moves

    def get_player_position(self):
        return self.grid[PLAYER_LAYER, :, :].nonzero().squeeze(0)

    def get_enemy_position(self):
        return self.grid[ENEMY_LAYER, :, :].nonzero().squeeze(0)

    def push_enemy(self, y, x):
        py, px = self.get_player_position()
        ey, ex = self.get_enemy_position()
        if sqr_distance(py, ey, px, ex) > 2:
            self.reset_push()
            self.grid[LONG_ENEMY_PUSH_LAYER, y, x] = 1
        else:
            self.clear_short_push()
            self.grid[SHORT_ENEMY_PUSH_LAYER, y, x] = 0
            self.grid[SHORT_ENEMY_PUSH_LAYER, y, x] = 1

    def push_todd(self, y, x):
        py, px = self.get_player_position()
        ty, tx = self.get_todd_position()
        if sqr_distance(py, ty, px, tx) > 2:
            self.reset_push()
            self.grid[LONG_TODD_PUSH_LAYER, :, :] = 0
            self.grid[LONG_TODD_PUSH_LAYER, y, x] = 1
        else:
            self.clear_short_push()
            self.grid[SHORT_TODD_PUSH_LAYER, y, x] = 0
            self.grid[SHORT_TODD_PUSH_LAYER, y, x] = 1

    def can_throw_mw(self, py, px, my, mx):
        if not self.is_cell_empty(my, mx):
            return False

        dx = mx - px
        dy = my - py

        if abs(dx) > abs(dy):
            if px > mx:
                py, px, my, mx = my, mx, py, px

            a = (my - py) / (mx - px)
            y = py
            for x in range(px, mx + 1):
                if self.grid[MW_LAYER, tr(y), x] > 0:
                    return False
                y += a

        else:
            if py > my:
                py, px, my, mx = my, mx, py, px

            a = (mx - px) / (my - py)
            x = px
            for y in range(py, my + 1):
                if self.grid[MW_LAYER, y, tr(x)] > 0:
                    return False
                x += a

        return True

    def player_throw_mw(self, y, x):
        self.grid[LONG_TODD_PUSH_LAYER:ENEMY_LAYER, :, :] = 0
        if self.current_player_mw > 0:
            return
        if self.grid[MW_LAYER, y, x] > 0:
            return
        if not self.is_cell_empty(y, x):
            self.current_player_mw = 1000
            return
        py, px = self.get_player_position()
        if not self.can_throw_mw(py.item(), px.item(), y, x):
            return
        self.grid[MW_LAYER, y, x] = 1
        self.current_player_mw = 2000

    def step_mw(self, time_step=200, mw_time=20000):

        self.grid[MW_LAYER, :, :] -= time_step / mw_time
        self.grid[MW_LAYER, self.grid[MW_LAYER, :, :] < 0] = 0

    def shortest_path(self, y1, x1, y2, x2, max_steps=100):
        start = (y1, x1)
        end = (y2, x2)
        queue = []
        heappush(queue, (0, (0, *start, [start])))

        for _ in range(max_steps):
            if not queue:
                return []
            c, (oc, y, x, p) = heappop(queue)
            if sqr_distance(y, end[0], x, end[1]) < 2:
                return p[1:]

            for ny, nx in self.get_empty_neighbours(y, x):
                nc = 1 if sqr_distance(y, ny, x, nx) < 2 else 6
                cost = oc + nc + sqr_distance(ny, end[0], nx, end[1])
                heappush(queue, (cost, (c + nc, ny, nx, p + [(ny, nx)])))

        return []

    def copy(self):
        new_board = Board2()
        new_board.grid = self.grid.clone()
        new_board.current_todd = self.current_todd
        new_board.current_player_mw = self.current_player_mw
        new_board.current_enemy_mw = self.current_enemy_mw
        return new_board

    def swap_enemy(self):
        ey, ex = self.get_enemy_position()
        py, px = self.get_player_position()
        self.set_enemy(py, px)
        self.set_player(ey, ex)

        self.current_player_mw, self.current_enemy_mw = self.current_enemy_mw, self.current_player_mw

        grid = self.grid.clone()
        og = self.enemy_push_grid.clone()

        self.enemy_push_grid[SHORT_ENEMY_PUSH_LAYER_ENEMY, :, :] = grid[SHORT_ENEMY_PUSH_LAYER, :, :]
        self.enemy_push_grid[LONG_ENEMY_PUSH_LAYER_ENEMY, :, :] = grid[LONG_ENEMY_PUSH_LAYER, :, :]
        self.enemy_push_grid[SHORT_TODD_PUSH_LAYER_ENEMY, :, :] = grid[SHORT_TODD_PUSH_LAYER, :, :]
        self.enemy_push_grid[LONG_TODD_PUSH_LAYER_ENEMY, :, :] = grid[LONG_TODD_PUSH_LAYER, :, :]

        self.grid[SHORT_ENEMY_PUSH_LAYER, :, :] = og[SHORT_ENEMY_PUSH_LAYER_ENEMY, :, :]
        self.grid[LONG_ENEMY_PUSH_LAYER, :, :] = og[LONG_ENEMY_PUSH_LAYER_ENEMY, :, :]
        self.grid[SHORT_TODD_PUSH_LAYER, :, :] = og[SHORT_TODD_PUSH_LAYER_ENEMY, :, :]
        self.grid[LONG_TODD_PUSH_LAYER, :, :] = og[LONG_TODD_PUSH_LAYER_ENEMY, :, :]

    def step(self, time_step=200, walk_time=200):
        self.current_todd -= time_step
        step_push(self.grid, SHORT_ENEMY_PUSH_LAYER, time_step)
        step_push(self.grid, SHORT_TODD_PUSH_LAYER, time_step)
        step_push(self.grid, LONG_ENEMY_PUSH_LAYER, time_step, push_time=1600)
        step_push(self.grid, LONG_TODD_PUSH_LAYER, time_step, push_time=1600)

        step_push(self.enemy_push_grid, SHORT_ENEMY_PUSH_LAYER_ENEMY, time_step)
        step_push(self.enemy_push_grid, SHORT_TODD_PUSH_LAYER_ENEMY, time_step)
        step_push(self.enemy_push_grid, LONG_ENEMY_PUSH_LAYER_ENEMY, time_step, push_time=1600)
        step_push(self.enemy_push_grid, LONG_TODD_PUSH_LAYER_ENEMY, time_step, push_time=1600)

        self.step_mw(time_step)

        if self.current_player_mw > 0:
            self.current_player_mw -= time_step
            if self.current_player_mw < 0:
                self.current_player_mw = 0

        if (self.grid[SHORT_ENEMY_PUSH_LAYER, :, :] < 0).any():

            ey, ex = self.get_enemy_position()
            py, px = self.grid[SHORT_ENEMY_PUSH_LAYER, :, :].nonzero().squeeze(0)
            if is_adjacent(ey, ex, py, px) and self.can_move(py, px):
                self.move_enemy(py, px)

            self.grid[SHORT_ENEMY_PUSH_LAYER, :, :] = 0

        if (self.grid[LONG_ENEMY_PUSH_LAYER, :, :] != 0).any():

            ey, ex = self.get_enemy_position()
            py, px = self.get_player_position()

            if not is_adjacent(ey, ex, py, px):
                paths = self.shortest_path(py.item(), px.item(), ey.item(), ex.item())
                remaining_time = time_step
                while paths and remaining_time > 0:
                    y, x = paths.pop(0)
                    self.move_player(y, x)
                    remaining_time -= walk_time

            if (self.grid[LONG_ENEMY_PUSH_LAYER, :, :] < 0).any():
                ey, ex = self.get_enemy_position()
                py, px = self.grid[LONG_ENEMY_PUSH_LAYER, :, :].nonzero().squeeze(0)
                if is_adjacent(ey, ex, py, px) and self.can_move(py, px):
                    self.move_enemy(py, px)

                self.grid[LONG_ENEMY_PUSH_LAYER, :, :] = 0

        if (self.grid[SHORT_TODD_PUSH_LAYER, :, :] < 0).any():

            ey, ex = self.get_todd_position()
            py, px = self.grid[SHORT_TODD_PUSH_LAYER, :, :].nonzero().squeeze(0)
            if is_adjacent(ey, ex, py, px) and self.can_move(py, px):
                self.move_tod(py, px)

            self.grid[SHORT_TODD_PUSH_LAYER, :, :] = 0

        if (self.grid[LONG_TODD_PUSH_LAYER, :, :] != 0).any():
            ey, ex = self.get_todd_position()
            py, px = self.get_player_position()

            if not is_adjacent(ey, ex, py, px):
                paths = self.shortest_path(py.item(), px.item(), ey.item(), ex.item())
                remaining_time = time_step
                while paths and remaining_time > 0:
                    y, x = paths.pop(0)
                    self.move_player(y, x)
                    remaining_time -= 200

            if (self.grid[LONG_TODD_PUSH_LAYER, :, :] < 0).any():
                ey, ex = self.get_todd_position()
                py, px = self.grid[LONG_TODD_PUSH_LAYER, :, :].nonzero().squeeze(0)
                if is_adjacent(ey, ex, py, px) and self.can_move(py, px):
                    self.move_tod(py, px)

                self.grid[LONG_TODD_PUSH_LAYER, :, :] = 0

        if self.current_enemy_mw > 0:
            self.current_enemy_mw -= time_step
            if self.current_enemy_mw < 0:
                self.current_enemy_mw = 0

        if (self.enemy_push_grid[LONG_TODD_PUSH_LAYER_ENEMY, :, :] != 0).any():
            ey, ex = self.get_todd_position()
            py, px = self.get_enemy_position()

            if not is_adjacent(ey, ex, py, px):
                paths = self.shortest_path(py.item(), px.item(), ey.item(), ex.item())
                remaining_time = time_step
                while paths and remaining_time > 0:
                    y, x = paths.pop(0)
                    self.move_enemy(y, x)
                    remaining_time -= 200

            if (self.enemy_push_grid[LONG_TODD_PUSH_LAYER_ENEMY, :, :] < 0).any():
                ey, ex = self.get_todd_position()
                py, px = self.enemy_push_grid[LONG_TODD_PUSH_LAYER_ENEMY, :, :].nonzero().squeeze(0)
                if is_adjacent(ey, ex, py, px) and self.can_move(py, px):
                    self.move_tod(py, px)

                self.enemy_push_grid[LONG_TODD_PUSH_LAYER_ENEMY, :, :] = 0

        if (self.enemy_push_grid[SHORT_TODD_PUSH_LAYER_ENEMY, : :, :] < 0).any():

            ey, ex = self.get_todd_position()
            py, px = self.enemy_push_grid[SHORT_TODD_PUSH_LAYER_ENEMY, : :, :].nonzero().squeeze(0)
            if is_adjacent(ey, ex, py, px) and self.can_move(py, px):
                self.move_tod(py, px)

            self.enemy_push_grid[SHORT_TODD_PUSH_LAYER_ENEMY, : :, :] = 0

        if (self.enemy_push_grid[LONG_ENEMY_PUSH_LAYER_ENEMY, : :, :] != 0).any():

            ey, ex = self.get_player_position()
            py, px = self.get_enemy_position()

            if not is_adjacent(ey, ex, py, px):
                paths = self.shortest_path(py.item(), px.item(), ey.item(), ex.item())
                remaining_time = time_step
                while paths and remaining_time > 0:
                    y, x = paths.pop(0)
                    self.move_enemy(y, x)
                    remaining_time -= walk_time

            if (self.enemy_push_grid[LONG_ENEMY_PUSH_LAYER_ENEMY, : :, :] < 0).any():
                ey, ex = self.get_player_position()
                py, px = self.enemy_push_grid[LONG_ENEMY_PUSH_LAYER_ENEMY, : :, :].nonzero().squeeze(0)
                if is_adjacent(ey, ex, py, px) and self.can_move(py, px):
                    self.move_player(py, px)

                self.enemy_push_grid[LONG_ENEMY_PUSH_LAYER_ENEMY, : :, :] = 0

        if (self.enemy_push_grid[SHORT_ENEMY_PUSH_LAYER_ENEMY, : :, :] < 0).any():

            ey, ex = self.get_player_position()
            py, px = self.enemy_push_grid[SHORT_ENEMY_PUSH_LAYER_ENEMY, : :, :].nonzero().squeeze(0)
            if is_adjacent(ey, ex, py, px) and self.can_move(py, px):
                self.move_player(py, px)

            self.enemy_push_grid[SHORT_ENEMY_PUSH_LAYER_ENEMY, : :, :] = 0

        if self.current_todd <= 0:

            moves = self.get_todd_available_moves()
            if moves:
                self.current_todd = 3000
                am = random.choice(moves)
                self.move_tod(*am)

    def __str__(self):
        rm = [['' for _ in range(4)] for _ in range(4)]
        for j in range(4):
            for k in range(4):
                for i in range(self.grid.shape[0]):
                    if i == self.grid.shape[0] - 9:
                        if self.grid[i, j, k] > 0:
                            rm[j][k] = colored('#', 'blue')
                            break
                    if i == self.grid.shape[0] - 8:
                        if self.grid[i, j, k] > 0:
                            rm[j][k] = colored('o', 'light_green')
                            break
                    if i == self.grid.shape[0] - 7:
                        if self.grid[i, j, k] > 0:
                            rm[j][k] = colored('o', 'green')
                            break
                    if i == self.grid.shape[0] - 6:
                        if self.grid[i, j, k] > 0:
                            rm[j][k] = colored('o', 'light_magenta')
                            break
                    if i == self.grid.shape[0] - 5:
                        if self.grid[i, j, k] > 0:
                            rm[j][k] = colored('o', 'magenta')
                            break
                    if i == self.grid.shape[0] - 4:
                        if self.grid[i, j, k] == 1:
                            rm[j][k] = colored('E', 'red')
                            break
                    if i == self.grid.shape[0] - 3:
                        if self.grid[i, j, k] == 1:
                            rm[j][k] = colored('P', 'green')
                            break
                    if i == self.grid.shape[0] - 2:
                        if self.grid[i, j, k] == 1:
                            rm[j][k] = colored('T', 'white')
                            break
                    if i == self.grid.shape[0] - 1:
                        if self.grid[i, j, k]:
                            rm[j][k] = colored('%', 'yellow')
                        else:
                            rm[j][k] = colored('.', 'white')

        return '\n'.join(['  '.join(r) for r in rm])
