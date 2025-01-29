from tree import Tree
import torch
from board import Board
from agent import Agent

def get_best_action(board, memory, agent, timestep=200, max_depth=5, max_iter=100, use_tree=True):
    agent.eval()
    if use_tree:
        root = Tree(board, memory=memory)

        res, iterations = root.search(agent, max_depth=max_depth, max_iter=max_iter, timestep=timestep)

        n = res
        i = 0
        while n.parent is not None and n.parent.parent is not None:
            n = n.parent
            i += 1

        return n.action, iterations

    else:

        with torch.no_grad():
            value, policy = agent(board.to_tensor().unsqueeze(0).unsqueeze(0), memory.unsqueeze(0))
            return policy.exp().argmax().item(), 1


def evaluate(agent, n=100, max_steps=20, board_stack=2, timestep=200, max_depth=5, max_iter=100, use_tree=True):
    agent.eval()
    v = 0
    fh = []
    iterations = 0
    for _ in range(n):
        board = Board()
        # board.board = [[0, 1, 0],
        #                [0, 2, 0],
        #                [1, 1, 3]]

        bs = board.to_tensor().unsqueeze(0).repeat(board_stack, 1)
        ah = []
        for _ in range(max_steps):
            a, it = get_best_action(board, bs, agent, timestep=timestep, max_depth=max_depth, max_iter=max_iter,
                                    use_tree=use_tree)
            iterations += it

            be = board.to_tensor()
            with torch.no_grad():
                value, pl = agent(be.unsqueeze(0).unsqueeze(0), bs.unsqueeze(0))
            board.do_action(a)
            be = board.to_tensor().unsqueeze(0)
            bs = torch.cat([bs[1:], be])

            if pl.exp().argmax() == a:
                ah.append(1)
            else:
                ah.append(0)
            if board.is_win():
                fh.extend(ah)
                v += 1
                break
            elif board.board[1][1] == 4 and board.board[0][1] != 3:
                break

            board.step(timestep)

    return v / n, sum(fh) / (len(fh) if len(fh) > 0 else 1), iterations





if __name__ == '__main__':
    board_stack = 5
    time_step = 1000
    max_iter = 5000
    max_depth = 15
    max_steps = 60
    agent = Agent(45, 60, Board().get_total_actions(), max_len=board_stack, nhead=4, num_layers=1)
    agent.load_state_dict(torch.load('models/mw.pt'))

    # 33260
    # 63800
    print(evaluate(agent, n=100, max_steps=max_steps, board_stack=board_stack, timestep=time_step, max_iter=max_iter,
                   max_depth=max_depth, use_tree=True))
