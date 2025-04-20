import copy
from termcolor import colored
import random
from heapq import heappush, heappop, heapify
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

BLOCK_CELLS = {(0, 1), (1, 1), (0, 3), (1, 3)}

TODD_WANDER_EVENT = 0
REMOVE_MW_EVENT = 1
SHORT_PUSH_EVENT = 2
SHORT_PUSH_LONG_EVENT = 3
CHASE_PUSH_EVENT = 4


def tr(v):
    f, w = math.modf(v)
    if f >= 0.5:
        return math.ceil(v)
    else:
        return math.floor(v)


def is_adjacent(y1, x1, y2, x2):
    return abs(y1 - y2) < 2 and abs(x1 - x2) < 2


def sqr_distance(y1, x1, y2, x2):
    return (y1 - y2) ** 2 + (x1 - x2) ** 2


class EventQueue:

    def __init__(self):
        self.events = []

    def push(self, event):
        heappush(self.events, event)

    def peek(self):
        return self.events[0]

    def pop(self):
        return heappop(self.events)

    def remove(self, event):
        self.events.remove(event)
        heapify(self.events)

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events)


class Board3:

    def __init__(self, blank=False, walk_frodo=True, walk_time=200):
        self.todd = 2

        if not blank:
            ty, tx = random.choice(tod_cells)
            py, px = random.choice(list(set(empty_cells) - {(ty, tx)}))
            ey, ex = random.choice(list(set(empty_cells) - {(ty, tx), (py, px)}))

            self.players_positions = [(py, px), (ey, ex), (ty, tx)]
            self.current_player = 0
            self.current_enemy = 1
            self.cooldowns = [0, 0]
            self.mw: set = set()
            self.current_time = 0
            self.ev = EventQueue()
            if walk_frodo:
                ev = (3000, TODD_WANDER_EVENT)
                self.ev.push(ev)

            self.walk_time = walk_time

    def push_enemy(self, y, x):
        ey, ex = self.get_enemy_position()
        self.push(ey, ex, y, x, self.current_enemy)

    def push_todd(self, y, x):
        ty, tx = self.get_todd_position()
        self.push(ty, tx, y, x, self.todd)

    def push(self, ey, ex, y, x, enemy):
        py, px = self.get_player_position()

        if sqr_distance(py, px, ey, ex) > 2:
            self.reset_push()
            path = self.shortest_path(py, px, ey, ex)
            if not path:
                return

            ev = (
                self.current_time + self.walk_time, CHASE_PUSH_EVENT, self.current_time, self.current_player, path,
                enemy,
                y,
                x)
            self.ev.push(ev)
        else:

            for e in self.ev:
                if e[1] == SHORT_PUSH_LONG_EVENT and e[0] - self.current_time < 1000:
                    return
            self.clear_short_push({SHORT_PUSH_EVENT})
            ev = (self.current_time + 1000, SHORT_PUSH_EVENT, enemy, y, x)
            self.ev.push(ev)

    def player_throw_mw(self, y, x):
        self.reset_push()
        if self.current_time < self.get_player_cooldown():
            return
        if (y, x) in self.mw:
            self.set_player_cooldown(self.current_time + 1000)
            return
        if not self.is_cell_empty(y, x):
            self.set_player_cooldown(self.current_time + 1000)
            return
        py, px = self.get_player_position()
        if not self.can_throw_mw(py, px, y, x):
            return
        self.mw.add((y, x))
        ev = (self.current_time + 20000, REMOVE_MW_EVENT, y, x)
        self.ev.push(ev)
        self.set_player_cooldown(self.current_time + 2000)

    def add_mw(self, y, x):
        if (y, x) in self.mw:
            return
        if not self.is_cell_empty(y, x):
            return

        self.mw.add((y, x))
        ev = (self.current_time + 20000, REMOVE_MW_EVENT, y, x)
        self.ev.push(ev)

    def get_player_cooldown(self):
        return self.cooldowns[self.current_player]

    def set_player_cooldown(self, value):
        self.cooldowns[self.current_player] = value

    def is_cell_empty(self, y: int, x: int):
        cell = (y, x)

        if self.players_positions[0] == cell or \
                self.players_positions[1] == cell or \
                self.players_positions[2] == cell or \
                cell in self.mw or \
                cell in BLOCK_CELLS:
            return False

        return True

    def get_todd_position(self):
        return self.players_positions[self.todd]

    def set_player(self, y, x):
        self.players_positions[self.current_player] = (y, x)

    def set_enemy(self, y, x):
        self.players_positions[self.current_enemy] = (y, x)

    def set_todd(self, y, x):
        self.players_positions[self.todd] = (y, x)

    def clear_short_push(self, events=None):
        if events is None:
            events = {SHORT_PUSH_EVENT, SHORT_PUSH_LONG_EVENT}
        for e in self.ev:
            if e[1] in events:
                self.ev.remove(e)
                break

    def clear_long_push(self):
        for e in self.ev:
            if e[1] == CHASE_PUSH_EVENT:
                self.ev.remove(e)
                break

    def reset_push(self):
        self.clear_short_push()
        self.clear_long_push()

    def move_todd(self, y, x):
        if not self.can_move_todd(y, x):
            return
        ty, tx = self.get_todd_position()
        if not is_adjacent(ty, tx, y, x):
            return
        self.set_todd(y, x)

    def move_player(self, y, x):
        if not self.can_move(y, x):
            return
        py, px = self.get_player_position()
        if not is_adjacent(py, px, y, x):
            return
        self.set_player(y, x)

    def move_enemy(self, y, x):
        if not self.can_move(y, x):
            return
        ey, ex = self.get_enemy_position()
        if not is_adjacent(ey, ex, y, x):
            return
        self.set_enemy(y, x)

    def move_p(self, p, y, x):
        if p == 2:
            mc = self.can_move_todd
        else:
            mc = self.can_move

        if not mc(y, x):
            return
        py, px = self.players_positions[p]
        if not is_adjacent(py, px, y, x):
            return
        self.set_p(p, y, x)

    def set_p(self, p, y, x):
        self.players_positions[p] = (y, x)

    def player_walk(self, y, x):
        self.reset_push()
        self.move_player(y, x)

    def get_empty_neighbours(self, y, x):
        neighbours = []
        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            if y + i < 0 or y + i > 3 or x + j < 0 or x + j > 3: continue
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

    def get_pos_available_moves(self, y, x):
        moves = []

        if self.can_move(y - 1, x):
            moves.append((y - 1, x))
        if self.can_move(y + 1, x):
            moves.append((y + 1, x))
        if self.can_move(y, x - 1):
            moves.append((y, x - 1))
        if self.can_move(y, x + 1):
            moves.append((y, x + 1))

        if self.can_move(y - 1, x - 1):
            moves.append((y - 1, x - 1))
        if self.can_move(y - 1, x + 1):
            moves.append((y - 1, x + 1))
        if self.can_move(y + 1, x - 1):
            moves.append((y + 1, x - 1))
        if self.can_move(y + 1, x + 1):
            moves.append((y + 1, x + 1))

        return moves

    def get_player_position(self):
        return self.players_positions[self.current_player]

    def get_enemy_position(self):
        return self.players_positions[self.current_enemy]

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
                if (tr(y), x) in self.mw:
                    return False
                y += a

        else:
            if py > my:
                py, px, my, mx = my, mx, py, px

            a = (mx - px) / (my - py)
            x = px
            for y in range(py, my + 1):
                if (y, tr(x)) in self.mw:
                    return False
                x += a

        return True

    def shortest_path(self, y1: int, x1: int, y2: int, x2: int, max_steps=50) -> list[tuple[int, int]]:
        start, end = (y1, x1), (y2, x2)
        queue = []
        heappush(queue, (0, 0, y1, x1))
        came_from = {}
        visited: set[tuple[int, int]] = set()

        for _ in range(max_steps):
            if not queue:
                return []

            _, oc, y, x = heappop(queue)
            if (y, x) in visited:
                continue
            visited.add((y, x))

            if is_adjacent(y, x, end[0], end[1]):
                path = []
                while (y, x) != start:
                    path.append((y, x))
                    y, x = came_from[(y, x)]
                return path[::-1]

            for ny, nx in self.get_empty_neighbours(y, x):
                if (ny, nx) in visited:
                    continue
                nc = 1 if sqr_distance(y, x, ny, nx) < 2 else 6
                cost = oc + nc + sqr_distance(ny, nx, end[0], end[1])
                heappush(queue, (cost, oc + nc, ny, nx))
                came_from[(ny, nx)] = (y, x)

        return []

    def copy(self):
        new_board = Board3(blank=True)
        new_board.players_positions = [self.players_positions[0], self.players_positions[1], self.players_positions[2]]
        new_board.current_player = self.current_player
        new_board.current_enemy = self.current_enemy

        new_board.cooldowns = [self.cooldowns[0], self.cooldowns[1]]

        new_board.mw = copy.copy(self.mw)

        new_board.current_time = self.current_time
        new_board.ev = copy.deepcopy(self.ev)

        new_board.walk_time = self.walk_time

        return new_board

    def swap_enemy(self):
        self.current_player, self.current_enemy = self.current_enemy, self.current_player

    def step(self, time_step=200):
        self.current_time += time_step
        while len(self.ev) > 0 and self.ev.peek()[0] <= self.current_time:
            _, event, *args = self.ev.pop()
            if event == REMOVE_MW_EVENT:
                self.mw.remove((args[0], args[1]))
            elif event == TODD_WANDER_EVENT:
                moves = self.get_todd_available_moves()
                if moves:
                    ev = (self.current_time + 3000, TODD_WANDER_EVENT)
                    self.ev.push(ev)
                    am = random.choice(moves)
                    self.move_todd(*am)
            elif event == SHORT_PUSH_EVENT or event == SHORT_PUSH_LONG_EVENT:
                ey, ex = self.players_positions[args[0]]
                py, px = args[1], args[2]
                if is_adjacent(ey, ex, py, px) and self.can_move(py, px):
                    self.move_p(*args)

            elif event == CHASE_PUSH_EVENT:
                start_time, current_player, path, current_enemy, py, px = args

                if not path:
                    continue

                if not self.can_move(*path[0]):
                    path = self.shortest_path(*self.players_positions[current_player],
                                              *self.players_positions[current_enemy])

                if not path:
                    continue

                ny, nx = path.pop(0)

                self.move_p(current_player, ny, nx)

                if is_adjacent(*self.players_positions[current_player], *self.players_positions[current_enemy]):
                    dif = 1600 - (self.current_time - start_time)
                    ev = (
                        self.current_time + dif, SHORT_PUSH_LONG_EVENT, current_enemy, py, px)
                    self.ev.push(ev)
                else:
                    ev = (
                        self.current_time + self.walk_time, CHASE_PUSH_EVENT, start_time, current_player, path,
                        current_enemy, py, px)
                    self.ev.push(ev)

    def __str__(self):
        rm = [[colored('.', 'white') for _ in range(4)] for _ in range(4)]

        for y, x in BLOCK_CELLS:
            rm[y][x] = colored('%', 'yellow')

        py, px = self.get_player_position()
        rm[py][px] = colored('P', 'green')
        ey, ex = self.get_enemy_position()
        rm[ey][ex] = colored('E', 'red')
        ty, tx = self.get_todd_position()
        rm[ty][tx] = colored('T', 'white')

        for (y, x) in self.mw:
            rm[y][x] = colored('#', 'blue')

        short_push = self.get_short_push_events()
        if short_push is not None:
            y, x = short_push[3], short_push[4]
            if short_push[2] == 2:
                rm[y][x] = colored('o', 'light_green')
            else:
                rm[y][x] = colored('o', 'light_magenta')

        long_push = self.get_long_push_events()
        if long_push is not None:
            if long_push[1] == SHORT_PUSH_LONG_EVENT:
                y, x = long_push[3], long_push[4]
                if long_push[2] == 2:
                    rm[y][x] = colored('O', 'light_green')
                else:
                    rm[y][x] = colored('O', 'light_magenta')
            elif long_push[1] == CHASE_PUSH_EVENT:
                y, x = long_push[6], long_push[7]
                if long_push[5] == 2:
                    rm[y][x] = colored('O', 'light_green')
                else:
                    rm[y][x] = colored('O', 'light_magenta')

        return '\n'.join(['  '.join(r) for r in rm])

    def get_short_push_events(self):
        for e in self.ev:
            if e[1] == SHORT_PUSH_EVENT:
                return e
        return None

    def get_long_push_events(self):
        for e in self.ev:
            if e[1] == SHORT_PUSH_LONG_EVENT or e[1] == CHASE_PUSH_EVENT:
                return e
        return None

    def board_repr(self):
        rm = [[0 for _ in range(4)] for _ in range(4)]

        for y, x in BLOCK_CELLS:
            rm[y][x] = -1

        py, px = self.get_player_position()
        rm[py][px] = 1
        ey, ex = self.get_enemy_position()
        rm[ey][ex] = 2
        ty, tx = self.get_todd_position()
        rm[ty][tx] = 3

        for k, v in self.mw:
            y, x = k
            if v > 0:
                rm[y][x] = 4

        return rm

    def __lt__(self, other):
        return 0

    def get_relative_cooldown(self, p):
        if self.cooldowns[p] > self.current_time:
            return self.cooldowns[p] - self.current_time
        return self.cooldowns[p]


    def get_relative_events(self):
        return [(e[0] - self.current_time, *e[1:]) for e in self.ev]

    def __eq__(self, other):
        if not isinstance(other, Board3):
            return False

        conditions = [
            self.players_positions == other.players_positions,
            self.current_player == other.current_player,
            self.current_enemy == other.current_enemy,
            self.get_relative_cooldown(0) == other.get_relative_cooldown(0),
            self.get_relative_cooldown(1) == other.get_relative_cooldown(1),
            self.mw == other.mw,
            self.get_relative_events() == other.get_relative_events(),
        ]
        return all(conditions)

    def __hash__(self):
        strs = (
            self.players_positions[0],
            self.players_positions[1],
            self.players_positions[2],
            self.current_player,
            self.current_enemy,
            self.get_relative_cooldown(0),
            self.get_relative_cooldown(1),
            frozenset(self.mw),
            str(self.get_relative_events())
        )

        return hash(strs)
