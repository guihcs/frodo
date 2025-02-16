from board3 import Board3, sqr_distance
from controller3 import ActionController
import time
from heapq import heappush, heappop
from tqdm.auto import tqdm
from mcts import search


board = Board3(walk_time=500)
board.set_player(1, 0)
board.set_enemy(2, 0)
board.set_todd(1, 2)
board.player_throw_mw(2, 1)
# board.swap_enemy()
board.step(2000)
# board.swap_enemy()
print(board)


start = time.time()
path = search(board, max_iterations=100000, time_step=1000)
if path:
    print(len(path))
print(f'Time: {time.time() - start}')

for b, a in path:
    print(b)
    print(ActionController(b).explain_action(a))